import itertools
import copy
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from .util.sequencing import GP_action_selector

from rl.geometry import Cuboid


def _evaluate_individual(agent, individual):
    evaluated_agent = copy.deepcopy(agent)
    state = evaluated_agent.env.reset()

    for step in itertools.count():
        items, h_map, actions = state
        action = GP_action_selector(actions, individual[0])
        next_state, _, done = evaluated_agent.env.step(action)

        if done:
            break
        for i, packer in enumerate(evaluated_agent.env.packers):
            packer.render().savefig(f"./outputs/gp_{step}_{i}.jpg")
        state = next_state

    return round(evaluated_agent.env.used_packers[0].space_utilization() * 100, 2)


def evaluate(population, agent):
    fitness_list = []

    # with ThreadPoolExecutor(max_workers=1) as executor:
    # futures = [
    #     executor.submit(_evaluate_individual, agent, individual)
    #     for individual in population
    # ]

    # fitness_list = [future.result() for future in futures]
    for individual in population:
        fitness = _evaluate_individual(agent, individual)
        fitness_list.append(fitness)

    return fitness_list
