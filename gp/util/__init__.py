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
    # add function
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)

    # terminals for sequencing and routing in my paper
    # pset.addTerminal(str("NOI"))  # number of items to be packed
    pset.addTerminal(str("X"))  #  bin space coordinate x
    pset.addTerminal(str("Y"))  # bin space coordinate y
    pset.addTerminal(str("Z"))  # bin space coordinate z
    # pset.addTerminal(str("HMAP"))  # acurrent height map of the bin
    pset.addTerminal(str("W"))  # the width of the item to be packed
    pset.addTerminal(str("D"))  # the depth of the item to be packed
    pset.addTerminal(str("H"))  # the height of the item to be packed
    # pset.addTerminal(str("COMPACTNESS"))  # the current compactness of the bin
    # pset.addTerminal(str(""))  # add by mengxu


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
