import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from .util.sequencing import treeNode_S, GP_evolve_S

from rl.env import indices


def _evaluate_individual(agent, individual):
    evaluated_agent = copy.deepcopy(agent)
    state = evaluated_agent.env.reset()

    items, h_map, actions = state
    if len(actions) == 0:
        raise Exception("0 actions")
    # TODO: use gp to decide the action
    # action, r = self.select(state)
    # TODO: use gp to decide the action

    # next_state, reward, done = evaluated_agent.env.step(action)

    return 0


def evaluate(population, agent):
    fitness_list = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(_evaluate_individual, agent, individual)
            for individual in population
        ]

        fitness_list = [future.result() for future in futures]

    return fitness_list
