import numpy as np
from deap import gp
from rl.env import indices


def GP_action_selector(state_map, actions, individual, toolbox):
    const_in, hmap_in, amap_in, imap_in = state_map
    state_data = [
        [a.squeeze(-1), b.squeeze(-1), c.squeeze(-1), np.mean(d)]
        for a, b, c, d in zip(const_in, hmap_in, amap_in, imap_in)
    ]

    actions_values = []
    func = toolbox.compile(expr=individual)
    for action_data in state_data:
        action_value = func(*action_data)
        actions_values.append(action_value)
    action_position = np.argmax(actions_values)
    action_space = indices(actions)
    i, j, k = action_space[action_position]
    return (i, j, k)
