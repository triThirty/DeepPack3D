import collections
import itertools

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    concatenate,
)
from tensorflow.keras.regularizers import l2

from rl.env import *


def q_net(k=1):
    weight_decay = 0.0005
    hmap_in = Input((32, 32, 1))
    amap_in = Input((32, 32, 1))
    imap_in = Input((k, 3))
    imap_x = Flatten()(imap_in)
    const_in = Input((32, 32, 1))

    x = concatenate([hmap_in, amap_in, const_in], axis=-1)

    x = Conv2D(
        64,
        11,
        strides=1,
        activation="relu",
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        128,
        9,
        strides=1,
        activation="relu",
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        256,
        7,
        strides=1,
        activation="relu",
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        512,
        5,
        strides=1,
        activation="relu",
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        1024,
        3,
        strides=1,
        activation="relu",
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        2048,
        2,
        strides=1,
        activation="relu",
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)

    emb = Dense(
        256, kernel_regularizer=l2(weight_decay), kernel_initializer="he_uniform"
    )(imap_x)

    x = concatenate([x, emb], axis=-1)

    x = Dense(
        1000,
        activation="relu",
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)

    x = Dense(
        100,
        activation="relu",
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)

    x = Dense(1, activation="linear")(x)

    outputs = x
    model = Model([const_in, hmap_in, amap_in, imap_in], outputs)
    return model


class Agent:
    def __init__(
        self,
        env=MultiBinPackerEnv(
            n_bins=2, max_bins=-1, size=(32, 32, 32), k=10, verbose=True
        ),
        train=True,
        verbose=True,
        visualize=False,
        batch_size=32,
    ):
        self.env = env

        self.gamma = 0.95

        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.99

        # self.ep_history = []

        self.warmup_epochs = 20
        self.warmup_lr = 1e-5
        self.learning_rate = 1e-3
        self.lr_min = 1e-5
        self.lr_drop = 10000
        self.epoch = 0
        self.update_epochs = 10

        self.batch_size = batch_size

        self.__train = train

        self.verbose = verbose
        self.visualize = visualize

        if self.__train:
            self.q_net = q_net(k=env.k - 1)
            self.q_net_target = q_net(k=env.k - 1)
            self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=self.warmup_lr)
            self.memory = collections.deque(maxlen=100000)
        else:
            self.q_net = None
            self.q_net_target = None
            self.q_optimizer = None
            self.memory = None

    def select(self, state):
        items, h_maps, actions = state
        action_space = indices(actions)

        #         print('actions: ', len(action_space))

        r = np.random.random()
        if r < self.eps:
            action = action_space[np.random.choice(len(action_space))]
        else:
            q = self.Q(state)
            action = action_space[np.argmax(q)]
            r = np.max(q)

        return action, r

    def Q_inputs(self, state, action=None):
        W, H, D = self.env.size

        items, h_maps, actions = state
        rows = actions if action is None else actions[[action]]
        rows = rows.astype(np.int64)

        item_idx, bin_idx = rows[:, A_ITEM], rows[:, A_BIN]
        x, y, z, w, h, d = (rows[:, c] for c in range(A_X, A_D + 1))

        # height map of the target bin with the placed item's top raised
        base = h_maps[bin_idx].astype(np.float64)
        zs = np.arange(base.shape[1])
        xs = np.arange(base.shape[2])
        in_z = (zs >= z[:, None]) & (zs < (z + d)[:, None])
        in_x = (xs >= x[:, None]) & (xs < (x + w)[:, None])
        mask = in_z[:, :, None] & in_x[:, None, :]
        amap_in = np.where(mask, (y + h)[:, None, None], base) / H

        hmap_in = np.empty_like(amap_in)
        hmap_in[:] = amap_in.max(axis=(1, 2), initial=0.0)[:, None, None]

        # item maps of the target bin without the selected item
        imaps = np.stack(
            [self.env.i_map(b, items) for b in range(len(self.env.packers))]
        )
        others = np.arange(len(items) - 1)
        others = others + (others >= item_idx[:, None])
        imap_in = imaps[bin_idx[:, None], others]

        hmap_in = hmap_in[..., None]
        amap_in = amap_in[..., None]
        const_in = np.ones(hmap_in.shape)

        return [const_in, hmap_in, amap_in, imap_in]

    def Q(self, state, action=None):
        const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)

        batch_size = self.batch_size
        sections = np.cumsum(
            [self.batch_size] * int(np.ceil(const_in.shape[0] / batch_size) - 1)
        )
        batches = map(
            lambda data: map(lambda x: x.copy(), np.split(data, sections, axis=0)),
            (const_in, hmap_in, amap_in, imap_in),
        )

        outputs = []
        for const_in, hmap_in, amap_in, imap_in in zip(*batches):
            # print(const_in.shape)
            q = self.q_net([const_in, hmap_in, amap_in, imap_in])
            #             print('const_in.shape')
            outputs.append(q)
        q = np.concatenate(outputs, axis=0)
        #         print(q.shape)
        return q

    def lr_scheduler(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr
        else:
            lr = self.learning_rate * (0.5 ** (epoch / self.lr_drop))
        return max(self.lr_min, lr)

    def train(self, history):
        q_inputs = []
        q_targets = []

        for state, action, next_state, reward, done in history:
            const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
            q_inputs.append([const_in[0], hmap_in[0], amap_in[0], imap_in[0]])
            if done:
                q_target = reward
            else:
                #                 print(np.amax(self.Q(next_state)))
                q_target = reward + self.gamma * np.amax(self.Q(next_state))
            #                 print(q_target)
            q_targets.append([q_target])

        const_in, hmap_in, amap_in, imap_in = zip(*q_inputs)
        q_inputs = [
            np.asarray(const_in),
            np.asarray(hmap_in),
            np.asarray(amap_in),
            np.asarray(imap_in),
        ]
        q_targets = np.asarray(q_targets)
        #         print('q_targets', q_targets)
        #         print([result for result in map(lambda inps: (inps.shape, np.amin(inps), np.amax(inps), np.mean(inps)), q_inputs)], q_targets)
        return self.fit(q_inputs, q_targets)

    def fit(self, q_inputs, q_targets):
        with tf.GradientTape() as tape:
            #             print(q_inputs)
            q = self.q_net_target(q_inputs)
            # print(tf.keras.losses.MeanSquaredError()(q_targets, q))
            loss = tf.reduce_mean(tf.square(q_targets - q))
            loss = tf.keras.losses.MeanSquaredError()(q_targets, q)
        #             print('q', q)
        #             print('loss', loss)

        #         print(list(zip(q, q_targets)))
        grad = tape.gradient(loss, self.q_net_target.trainable_variables)

        #         gradient clipping
        #         if self.epoch < self.warmup_epochs:
        #             grad = [tf.clip_by_value(value, -1e-5, 1e-5) for value in grad]

        self.q_optimizer.apply_gradients(
            zip(grad, self.q_net_target.trainable_variables)
        )

        self.q_optimizer.learning_rate.assign(self.lr_scheduler(self.epoch))

        self.epoch += 1

        #         print([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        if self.epoch % self.update_epochs == 0:
            print("update")
            self.q_net.set_weights(
                [
                    a * 0.5 + b * (1 - 0.5)
                    for a, b in zip(
                        self.q_net.get_weights(), self.q_net_target.get_weights()
                    )
                ]
            )
        return loss

    def run(self, max_ep=1, verbose=False, train=None):
        if train is None:
            train = self.__train

        iters = (
            (i for i, _ in enumerate(iter(bool, True)))
            if max_ep == -1
            else range(max_ep)
        )

        for ep in iters:
            if verbose:
                print(f"ep {ep}:")

            state = self.env.reset()
            ep_reward = 0

            history = []

            for step in itertools.count():
                if verbose:
                    print(f"\nstep {step}")

                items, h_map, actions = state
                if len(actions) == 0:
                    raise Exception("0 actions")
                    # break
                action, r = self.select(state)

                if verbose:
                    print(f"possible actions: {len(actions)}")
                    print(f"action: {action}")
                    print(f"placement: {placement(actions, action)}")

                yield placement(actions, action)

                next_state, reward, done = self.env.step(action)

                history.append((state, action, next_state, reward, done))

                if self.visualize:
                    for i, packer in enumerate(self.env.packers):
                        packer.render().savefig(f"./outputs/{ep}_{step}_{i}.jpg")

                ep_reward += reward
                if done:
                    break
                state = next_state

            loss = None
            if train:
                self.memory.extend(history)
                if len(self.memory) > 1000:
                    print("update model")
                    history = [
                        self.memory[i] for i in np.random.choice(len(self.memory), 16)
                    ]
                    loss = self.train(history)

            # self.ep_history.append(
            #     (
            #         [packer.space_utilization() for packer in self.env.used_packers],
            #         self.env.used_bins,
            #         ep_reward,
            #     )
            # )

            yield None

            utils = [
                round(packer.space_utilization() * 100, 2)
                for packer in self.env.used_packers
            ]
            if self.verbose:
                print(
                    f"Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}, memory: {len(self.memory) if self.memory is not None else None}, eps: {self.eps:.2f}, loss: {loss}, lr: {self.q_optimizer.learning_rate.numpy() if self.q_optimizer is not None else None}"
                )


class HeuristicAgent:
    def __init__(
        self,
        heuristic,
        env=MultiBinPackerEnv(
            n_bins=2, max_bins=-1, size=(32, 32, 32), k=10, verbose=True
        ),
        verbose=True,
        visualize=False,
    ):
        self.env = env

        self.heuristic = heuristic
        self.ep_history = []

        self.verbose = verbose
        self.visualize = visualize

    def select(self, state):
        # state = (items, h_map, actions)

        items, h_map, actions = state
        #         print(len(indices(actions)))
        action = self.heuristic(actions)

        return action

    def run(self, max_ep=1, verbose=False):
        iters = (
            (i for i, _ in enumerate(iter(bool, True)))
            if max_ep == -1
            else range(max_ep)
        )

        for ep in iters:
            if verbose:
                print(f"ep {ep}:")

            state = self.env.reset()
            ep_reward = 0

            history = []

            for step in itertools.count():
                if verbose:
                    print(f"\nstep {step}")

                items, h_map, actions = state
                if len(actions) == 0:
                    raise Exception("0 actions")

                action = self.select(state)

                if verbose:
                    print(f"action: {action}")

                next_state, reward, done = self.env.step(action)

                history.append((state, action, next_state, reward, done))

                if verbose:
                    print(f"actions: {actions}")
                    print(f"reward: {reward}, done: {done}")
                    print(f"placement: {placement(actions, action)}")

                yield placement(actions, action)

                ep_reward += reward

                if self.visualize:
                    for i, packer in enumerate(self.env.packers):
                        packer.render().savefig(f"./outputs/{ep}_{step}_{i}.jpg")

                if done:
                    break

                state = next_state

            self.ep_history.append(
                (
                    [packer.space_utilization() for packer in self.env.used_packers],
                    self.env.used_bins,
                    ep_reward,
                )
            )

            yield None

            utils = [
                round(packer.space_utilization() * 100, 2)
                for packer in self.env.used_packers
            ]
            if self.verbose:
                print(
                    f"Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}"
                )


def gp_actions_data(actions):
    return [
        [int(v) for v in row[A_X : A_SD + 1]] + [a] for a, row in enumerate(actions)
    ]


def _first_by(*keys):
    """Index of the lexicographically smallest key tuple, ties broken by action order."""
    order = np.lexsort((np.arange(len(keys[0])),) + tuple(reversed(keys)))
    return int(order[0])


def _columns(actions):
    rows = np.asarray(actions, dtype=np.int64)
    return {name: rows[:, c] for name, c in (
        ("x", A_X), ("y", A_Y), ("z", A_Z), ("w", A_W), ("h", A_H), ("d", A_D),
        ("sw", A_SW), ("sh", A_SH), ("sd", A_SD),
    )}


def bottom_left(actions):
    c = _columns(actions)
    return _first_by(c["y"] + c["h"], c["x"] + c["w"], c["z"] + c["d"])


def best_short_side_fit(actions):
    c = _columns(actions)
    return _first_by(np.minimum(c["sw"] - c["w"], c["sh"] - c["h"]))


def best_area_fit(actions):
    c = _columns(actions)
    volume = c["sw"] * c["sh"] * c["sd"]
    return _first_by(volume, np.minimum(c["sw"] - c["w"], c["sh"] - c["h"]))


def best_long_side_fit(actions):
    c = _columns(actions)
    return _first_by(np.maximum(c["sw"] - c["w"], c["sh"] - c["h"]))
