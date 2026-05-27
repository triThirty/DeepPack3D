import copy
import random
import json

import numpy as np
from deap import gp, creator, base
from deap import tools

from . import saveFile


def protected_div(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def init_primitives(pset):
    # --- Matrix-to-Matrix Primitives ---
    # These let the tree process, blend, or compare the image inputs first
    pset.addPrimitive(np.add, [np.ndarray, np.ndarray], np.ndarray, name="MatAdd")
    pset.addPrimitive(np.subtract, [np.ndarray, np.ndarray], np.ndarray, name="MatSub")
    pset.addPrimitive(np.absolute, [np.ndarray], np.ndarray, name="MatAbs")

    # --- Matrix-to-Scalar Bridge Primitives ---
    # These collapse a 2D image matrix down into a single float constant
    pset.addPrimitive(np.mean, [np.ndarray], float, name="MatMean")
    pset.addPrimitive(np.std, [np.ndarray], float, name="MatStd")
    pset.addPrimitive(
        lambda img: float(np.max(img)), [np.ndarray], float, name="MatMax"
    )

    # --- Scalar-to-Scalar Primitives ---
    # Once converted to a float, the tree can perform standard scalar math
    pset.addPrimitive(np.add, [float, float], float, name="ScalarAdd")
    pset.addPrimitive(np.subtract, [float, float], float, name="ScalarSub")
    pset.addPrimitive(np.multiply, [float, float], float, name="ScalarMul")
    pset.addPrimitive(protected_div, [float, float], float, name="ScalarDiv")
    pset.addPrimitive(np.maximum, [float, float], float, name="ScalarMax")
    pset.addPrimitive(np.minimum, [float, float], float, name="ScalarMin")

    # --- Constant Terminals ---
    # Add random floating-point constants directly into the tree mix
    pset.addEphemeralConstant("rand101", lambda: random.uniform(-1.0, 1.0), float)

    pset.addTerminal("constance", np.ndarray, "Constant")  #  bin space coordinate x
    pset.addTerminal("hmap", np.ndarray, "Height_Map")  # bin space coordinate y
    pset.addTerminal("amap", np.ndarray, "Action_Map")  # bin space coordinate z
    pset.addTerminal(
        "imap", np.ndarray, "Item_Map"
    )  # the width of the item to be packed


def init_toolbox(toolbox, pset, config):
    weights = (1.0,)
    creator.create("FitnessMax", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMax, pset=pset)

    toolbox.register(
        "expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6
    )  # original max = 6, modified by mengxu 2022.10.15 to check
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    N_TREES = (
        config.N_TREES
    )  # todo: only for test, need to be the same with original GPFC.py
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.tree, n=N_TREES
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("expr_mut", gp.genFull, min_=2, max_=8)

    toolbox.register("mate", lim_xmate)
    toolbox.register("mutate", lim_xmut, expr=toolbox.expr_mut)
    toolbox.register("select", tools.selTournament, tournsize=config.TOURNAMENT_SIZE)


# the following is modified by mengxu
def xmate(ind1, ind2):
    i1 = random.randrange(len(ind1))
    ind1[i1], ind2[i1] = gp.cxOnePoint(ind1[i1], ind2[i1])

    # exchange the other tree
    # i2 = 1 - i1  # only for individual with two tree
    # ind1[i2], ind2[i2] = ind2[i2], ind1[i2]
    return ind1, ind2


def maxheight(v):
    return max(i.height for i in v)


def wrap(func, *args, **kwargs):
    # MAX_HEIGHT = 8 #todo: only for test, need to be the same with original GPFC.py
    keep_inds = [copy.deepcopy(ind) for ind in args]
    new_inds = list(func(*args, **kwargs))
    for i, ind in enumerate(new_inds):
        if maxheight(ind) > 8:
            new_inds[i] = random.choice(keep_inds)
    return new_inds


def lim_xmate(ind1, ind2):
    return wrap(xmate, ind1, ind2)


def xmut(ind, expr):
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)
    ind[i1] = indx[0]
    return (ind,)


def lim_xmut(ind, expr):
    res = wrap(xmut, ind, expr=expr)
    return res


def init_stats():
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.MultiStatistics(fitness=fitness_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats


def record(
    halloffame,
    population,
    gen,
    stats,
    logbook,
    verbose,
    config,
    min_fitness,
    best_ind_all_gen,
):
    if halloffame is not None:
        halloffame.clear()
        halloffame.update(population)

    pop_fit = [ind.fitness.values[0] for ind in population]
    best_index = np.argmin(pop_fit)
    best_ind_all_gen.append(population[best_index])
    p_one = population[best_index]
    saveFile.save_individual_each_gen_to_txt(config, p_one, gen)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=gen, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    min_fitness.append(p_one.fitness.values[0])


def load_individual_from_gen_json_format(config):
    path = saveFile.formula_base_dir.substitute(**config)
    with open(
        path,
        "r",
    ) as fileName_individual:
        dict = json.load(fileName_individual)

    return dict
