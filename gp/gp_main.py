# import simpy
import time
from deap import base, creator, gp, tools
import numpy as np

from .util import init_primitives

# import MTGP_KNN.multi_tree as REP
# from MTGP import ea_simple_elitism
# from util.ParallelToolbox import ParallelToolbox
# import util.saveFile as saveFile

# from functools import partial

# import util.job_creation as job_creation
# import util.agent_machine as agent_machine
# import util.agent_workcenter as agent_workcenter
# import util.sequencing as sequencing
# import util.routing as routing
# import util.multi_tree as mt
# from util.selection import (
#     selElitistAndTournament,
# )
# from util.shopfloor import evaluate


def init_toolbox(toolbox, pset, config):
    # pass
    # REP.init_toolbox(toolbox, pset)
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)
    toolbox.register(
        "expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6
    )  # original max = 6, modified by mengxu 2022.10.15 to check
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    N_TREES = 2  # todo: only for test, need to be the same with original GPFC.py
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


def GPFC_main(config):
    num_features = 0  # the initial number of terminals is 0, then I will add more terminals into the pset
    pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset.context["array"] = np.array
    init_primitives(pset)
    weights = (-1.0,)
    creator.create("FitnessMin", base.Fitness, weights=weights)
    # set up toolbox
    toolbox = base.Toolbox()  # base.Toolbox()
    init_toolbox(toolbox, pset, config)

    # TODO: implement the evalutate function
    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=config.POP_SIZE)
    stats = init_stats()
    hof = tools.HallOfFame(1)

    # TODO: implement the evolutionary process


def main(config):
    # saveFile.clear_individual_each_gen_to_txt(config)
    start = time.time()
    # min_fitness, p_one, best_ind_all_gen, all_individuals = GPFC_main(config)
    GPFC_main(config)
    end = time.time()
    running_time = end - start
    # saveFile.save_each_gen_best_individual_json_format(config, best_ind_all_gen)
    # saveFile.save_each_gen_best_individual_meng(config, best_ind_all_gen)
    # print(min_fitness)
    print("Training time: " + str(running_time))
    print("Training end!")
