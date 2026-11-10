import numpy as np

from rl.binpacker import *
from rl.conveyor import Conveyor, rotated_sizes
from rl.geometry import Cuboid
from rl.SpacePartitioner import SpacePartitioner


class Env:
    def __init__(self, verbose):
        self.verbose = verbose
        self._state = None

    def reset(self):
        raise Exception("not implemented")

    def state(self, step=False):
        raise Exception("not implemented")

    def step(self, action):
        raise Exception("not implemented")

    def actions(self):
        raise Exception("not implemented")


# State layout (all uint8 to keep replay memory compact):
#   items:   (k, 3) item sizes (w, h, d); empty conveyor slots are all-zero rows
#   h_maps:  (n_bins, D, W) height maps
#   actions: (n_actions, ACTION_COLS) rows, see the column indices below
# An action is the row index into `actions`.
A_ITEM, A_BIN = 0, 1
A_X, A_Y, A_Z, A_W, A_H, A_D = 2, 3, 4, 5, 6, 7
A_SX, A_SY, A_SZ, A_SW, A_SH, A_SD = 8, 9, 10, 11, 12, 13
ACTION_COLS = 14
UINT8_MAX = np.iinfo(np.uint8).max


def indices(actions):
    return list(range(len(actions)))


def placement(actions, action):
    """Decode an action row into (bin, (x, y, z), (w, h, d), split)."""
    row = [int(v) for v in actions[action]]
    return (
        row[A_BIN],
        (row[A_X], row[A_Y], row[A_Z]),
        (row[A_W], row[A_H], row[A_D]),
        Cuboid(*row[A_SX : A_SD + 1]),
    )


class MultiBinPackerEnv(Env):
    def __init__(
        self,
        n_bins,
        size,
        k=1,
        max_bins=None,
        max_items=None,
        replace="all",
        verbose=False,
        prealloc_bins=0,
        prealloc_items=0,
        shuffle=False,
        use_rotate=True,
        use_skip=True,
    ):
        super().__init__(verbose)
        if max(size) > UINT8_MAX or k > UINT8_MAX + 1 or n_bins > UINT8_MAX + 1:
            raise ValueError(
                f"uint8 state requires bin size <= {UINT8_MAX}, k <= {UINT8_MAX + 1} "
                f"and n_bins <= {UINT8_MAX + 1}"
            )
        self.n_bins = n_bins
        self.size = size
        self.k = k

        self.packers = [SpacePartitioner(size) for _ in range(n_bins)]
        self.conveyor = Conveyor(
            k,
            max_items=max_items,
            prealloc_bins=prealloc_bins,
            prealloc_items=prealloc_items,
            shuffle=shuffle,
        )

        self.max_bins = -1 if max_bins is None else max_bins
        self.used_bins = n_bins

        self.used_packers = [] + self.packers
        self.replace = replace  # {'max', 'min', 'all'}

        self.use_rotate = use_rotate
        self.use_skip = use_skip

    def reset(self):
        for packer in self.packers:
            packer.reset()
        self.conveyor.reset()

        self._state = None
        self.used_bins = self.n_bins
        self.used_packers = []
        return self.state()

    def state(self, step=False):
        if step or self._state is None:
            items = self._items(self.conveyor.peek())
            h_maps = np.array(self._height_maps(), dtype=np.uint8)
            actions = self.actions(
                items, h_maps, rotate=self.use_rotate, skip=self.use_skip
            )

            self._state = (items, h_maps, actions)

        return self._state

    def _items(self, items):
        sizes = np.array(
            [(0, 0, 0) if item is None else tuple(item) for item in items],
            dtype=np.int64,
        ).reshape(-1, 3)
        if sizes.size and (sizes.max() > UINT8_MAX or sizes.min() < 0):
            raise ValueError(f"item sizes must be within [0, {UINT8_MAX}]: {sizes}")
        return sizes.astype(np.uint8)

    def _height_maps(self):
        return [packer.height_map.copy() for packer in self.packers]

    def placeable_coords(self, packer, h_map, size):
        xz = []
        splits = {}
        for split in packer.free_splits:
            if (split.top < self.size[1]) or (not split.fit(size)):
                continue
            x, y, z = split.coord
            xz.append((x, z))
            splits[(x, z)] = split
        xz = set(xz)

        w, h, d = size
        xyz = []
        for x, z in xz:
            region = h_map[z : z + d, x : x + w]
            y = int(np.amax(region))
            # placement and stability constraints
            if np.count_nonzero(region == y) / (d * w) > 0.5:
                xyz.append((x, y, z, splits[(x, z)]))

        return xyz

    def actions(self, items, h_maps, rotate, skip):
        rows = []

        for i_item, item in enumerate(items):
            if not item.any():
                continue

            for i_bin, packer in enumerate(self.packers):
                for size in rotated_sizes(tuple(int(v) for v in item), rotate):
                    for x, y, z, split in self.placeable_coords(
                        packer, h_maps[i_bin], size
                    ):
                        rows.append((i_item, i_bin, x, y, z, *size, *split))

            # always pick the first available item
            if not skip:
                break

        return np.array(rows, dtype=np.uint8).reshape(-1, ACTION_COLS)

    def step(self, action):
        items, h_maps, actions = self.state()

        row = [int(v) for v in actions[action]]
        i, j = row[A_ITEM], row[A_BIN]
        x, y, z, w, h, d = row[A_X : A_D + 1]

        packer = self.packers[j]
        cuboid = Cuboid(x, y, z, w, h, d)
        if not packer.add(cuboid):
            raise Exception(f"invalid space {cuboid}")
        self.conveyor.grab(i)

        #         if self.verbose:
        #             print(f'action: {action}, item: {i}, bin: {j}, rotated item: {(w, h, d)}, coord: {(x, y, z)}')
        #             sns.heatmap(self.p_map(cuboid), vmin=0.0, vmax=1.0, cmap='coolwarm')
        #             plt.show()

        next_state = self.state(step=True)

        # reward shaping
        items, h_maps, actions = next_state

        h_map = h_maps[j].astype(np.int64)

        volume = np.sum([split.volume for split in packer.splits])
        pyramid = volume / np.sum(h_map)
        compactness = volume / np.prod((packer.size[0], np.amax(h_map), packer.size[2]))
        reward = (pyramid + compactness) / 2

        done = len(actions) == 0

        if done:
            if self.max_bins != -1 and self.used_bins + 1 > self.max_bins:
                for i, packer in enumerate(
                    packer for packer in self.packers if packer.space_utilization() != 0
                ):
                    self.used_packers.append(packer)
                    loc = self.packers.index(packer)
                    if self.verbose:
                        print(
                            f"bin {self.used_bins - self.n_bins + i}, loc: {loc}, space util: \
                                {packer.space_utilization() * 100:.2f}, packed items: {len(packer.splits)}"
                        )
                done = True
            else:
                if self.replace == "max":
                    loc = np.argmax(
                        [packer.space_utilization() for packer in self.packers]
                    )
                    packer = self.packers[loc]
                    if self.verbose:
                        print(
                            f"bin {self.used_bins - self.n_bins}, loc: {loc}, \
                                space util: {packer.space_utilization() * 100:.2f}, packed items: {len(packer.splits)}"
                        )

                    self.used_packers.append(self.packers[loc])
                    self.packers[loc] = SpacePartitioner(self.size)
                    self.packers[loc].reset()
                    added = 1
                    self.used_bins += 1
                elif self.replace == "all":
                    added = 0
                    while True:
                        loc = np.argmax(
                            [packer.space_utilization() for packer in self.packers]
                        )
                        packer = self.packers[loc]

                        if packer.space_utilization() == 0:
                            break
                        if self.max_bins != -1 and self.used_bins + 1 > self.max_bins:
                            break
                        if self.verbose:
                            print(
                                f"bin {self.used_bins - self.n_bins}, loc: {loc}, space util: \
                                    {packer.space_utilization() * 100:.2f}, packed items: {len(packer.splits)}"
                            )

                        self.used_packers.append(self.packers[loc])
                        self.packers[loc] = SpacePartitioner(self.size)
                        self.packers[loc].reset()
                        added += 1
                        self.used_bins += 1
                else:
                    raise Exception("not implemented")

                next_state = self.state(step=True)
                items, h_maps, actions = next_state
                done = len(actions) == 0
                if done:
                    for i, packer in enumerate(
                        packer
                        for packer in self.packers
                        if packer.space_utilization() != 0
                    ):
                        self.used_packers.append(packer)
                        loc = self.packers.index(packer)
                        if self.verbose:
                            print(
                                f"bin {self.used_bins - self.n_bins + i + 1}, loc: {loc}, space util: \
                                    {packer.space_utilization() * 100:.2f}, packed items: {len(packer.splits)}"
                            )
                    self.used_bins -= len(
                        [
                            packer
                            for packer in self.packers
                            if packer.space_utilization() == 0
                        ]
                    )
            #                 if not done:
            #                     self.used_bins += added
            #                     pass
            if self.verbose:
                print()

        return next_state, reward, done

    def p_map(self, i_bin, cuboid):
        x, y, z, w, h, d = (int(v) for v in cuboid)

        W, H, D = self.packers[i_bin].size
        mask = np.zeros((W, D))
        mask[z : z + d, x : x + w] = h / H

        return mask

    def i_map(self, i_bin, items):
        W, H, D = self.packers[i_bin].size

        # empty slots are zero rows, so they map to zeros as before
        masks = np.zeros((self.k, 3))
        masks[: len(items)] = np.asarray(items, dtype=np.float64) / (W, H, D)

        return masks
