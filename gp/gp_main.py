# import simpy
import time
from deap import base, creator, gp, tools
import numpy as np

from .util import init_primitives, init_stats, init_toolbox, saveFile
from .evaluate import evaluate
from .evolution import evolve


def GPFC_main(config, agent):
    num_features = 0
    pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset.context["array"] = np.array
    init_primitives(pset)

    toolbox = base.Toolbox()  # base.Toolbox()
    init_toolbox(toolbox, pset, config)
    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=config.POP_SIZE)
    stats = init_stats()
    hof = tools.HallOfFame(1)

    pop, logbook, max_fitness, best_ind_all_gen, all_individuals = evolve(
        pop, agent, toolbox, stats, hof, config
    )
    best = hof[0]
    return max_fitness, best, best_ind_all_gen, all_individuals


def main(config, agent):
    # saveFile.clear_individual_each_gen_to_txt(config)
    start = time.time()
    max_fitness, best, best_ind_all_gen, all_individuals = GPFC_main(config, agent)
    # GPFC_main(config, agent)
    end = time.time()
    running_time = end - start
    saveFile.save_each_gen_best_individual_json_format(config, best_ind_all_gen)
    saveFile.save_each_gen_best_individual_meng(config, best_ind_all_gen)
    # print(max_fitness)
    print("Training time: " + str(running_time))
    print("Training end!")
