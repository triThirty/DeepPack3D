import numpy as np
from rl.agent import gp_actions_data


def GP_action_selector(
    actions, individual
):  # genetic programming evolved sequencing rule
    actions_data = gp_actions_data(actions)

    actions_values = []
    for action_data in actions_data:
        action_value = treeNode_S(individual, 0, action_data)
        actions_values.append(action_value)
    action_position = np.argmax(actions_values)
    i, j, k = actions_data[action_position][-1]
    return (i, j, k)


def treeNode_S(tree, index, data):
    if tree[index].arity == 2:
        if tree[index].name == "add":
            return treeNode_S(tree, index + 1, data) + treeNode_S(tree, index + 2, data)
        elif tree[index].name == "subtract":
            return treeNode_S(tree, index + 1, data) - treeNode_S(tree, index + 2, data)
        elif tree[index].name == "multiply":
            return treeNode_S(tree, index + 1, data) * treeNode_S(tree, index + 2, data)
        elif tree[index].name == "protected_div":
            return protected_div(
                treeNode_S(tree, index + 1, data), treeNode_S(tree, index + 2, data)
            )
        elif tree[index].name == "maximum":
            return np.maximum(
                treeNode_S(tree, index + 1, data), treeNode_S(tree, index + 2, data)
            )
        elif tree[index].name == "minimum":
            return np.minimum(
                treeNode_S(tree, index + 1, data), treeNode_S(tree, index + 2, data)
            )
    elif tree[index].arity == 1:
        if tree[index].name == "lf":  # add by mengxu 2022.11.08
            ref = treeNode_S(tree, index + 1, data)
            if isinstance(ref, (np.int64, np.float64, float, int)):
                return 1 / (1 + np.exp(-ref))
            else:
                for i in range(len(ref)):
                    ref[i] = 1 / (1 + np.exp(-ref[i]))
                    # print(ref[i])
                return ref
    elif tree[index].arity == 0:
        if tree[index].name == "X":
            return data[0]
        elif tree[index].name == "Y":
            return data[1]
        elif tree[index].name == "Z":
            return data[2]
        elif tree[index].name == "W":
            return data[3]
        elif tree[index].name == "H":
            return data[4]
        elif tree[index].name == "D":
            return data[5]
        elif tree[index].name == "S_X":
            return data[6]
        elif tree[index].name == "S_Y":
            return data[7]
        elif tree[index].name == "S_Z":
            return data[8]
        elif tree[index].name == "S_W":
            return data[9]
        elif tree[index].name == "S_H":
            return data[10]
        elif tree[index].name == "S_D":
            return data[11]
        # elif tree[index].name == "WKR":
        #     return data[6]
        # elif tree[index].name == "NOR":
        #     return data[7]
        # elif tree[index].name == "TIS":
        #     return data[8]
        # elif tree[index].name == "SLACK":
        #     return data[9]

        # return tree[index].value


def protected_div(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x
