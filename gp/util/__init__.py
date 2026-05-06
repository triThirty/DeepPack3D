import copy
import random
import numpy as np

from collections import defaultdict
from deap import gp, creator
from deap import tools
from functools import partial


def protected_div(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def init_toolbox(toolbox, pset):
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMin,
        num_calculation=int,
        pset=pset,
    )

    toolbox.register(
        "expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6
    )  # original max = 6, modified by mengxu 2022.10.15 to check
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.tree, n=N_TREES
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("expr_mut", gp.genFull, min_=2, max_=8)

    toolbox.register("mate", lim_xmate)
    toolbox.register("mutate", lim_xmut, expr=toolbox.expr_mut)

    toolbox.register("score_mate", newlim_xmate)
    toolbox.register("score_mutate", newlim_xmut, expr=toolbox.expr_mut)


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
    # pass
    # REP.init_toolbox(toolbox, pset)
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)
    toolbox.register(
        "expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6
    )  # original max = 6, modified by mengxu 2022.10.15 to check
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    N_TREES = 1  # todo: only for test, need to be the same with original GPFC.py
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.tree, n=N_TREES
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("expr_mut", gp.genFull, min_=2, max_=8)

    toolbox.register("mate", tools.crossover.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut)

    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=config.TOURNAMENT_SIZE,
        elitism=config.ELITISM,
    )


def init_stats():
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.MultiStatistics(fitness=fitness_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats
