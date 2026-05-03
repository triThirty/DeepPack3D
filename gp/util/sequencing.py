import numpy as np


def GP_evolve_S(data, tree_S):  # genetic programming evolved sequencing rule
    new_data = []
    # new_data.append(np.array([data[0] for i in range(len(data[3]))]))
    # new_data.append(np.array([data[1] for i in range(len(data[3]))]))
    # new_data.append(np.array([data[2] for i in range(len(data[3]))]))
    # for i in range(3, len(data)):
    #     new_data.append(data[i])
    individualvalue = treeNode_S(
        tree_S, 0, new_data
    )  # todo: actually, this should be used for sequencing rule
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return 0  # todo: need to check if this is right!!! by mengxu 2022.10.15
    job_position = individualvalue.argmin()
    return job_position


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
        if tree[index].name == "NIQ":
            return data[0]
        elif tree[index].name == "WIQ":
            return data[1]
        elif tree[index].name == "MWT":
            return data[2]
        elif tree[index].name == "PT":
            return data[3]
        elif tree[index].name == "NPT":
            return data[4]
        elif tree[index].name == "OWT":
            return data[5]
        elif tree[index].name == "WKR":
            return data[6]
        elif tree[index].name == "NOR":
            return data[7]
        elif tree[index].name == "TIS":
            return data[8]
        elif tree[index].name == "SLACK":
            return data[9]

        # return tree[index].value
