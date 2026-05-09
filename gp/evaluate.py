import itertools
import copy

from .util.sequencing import GP_action_selector


def _evaluate_individual(agent, individual):
    evaluated_agent = copy.deepcopy(agent)
    state = evaluated_agent.env.reset()

    for step in itertools.count():
        items, h_map, actions = state
        action = GP_action_selector(actions, individual[0])
        next_state, _, done = evaluated_agent.env.step(action)

        if done:
            break
        state = next_state

    return round(evaluated_agent.env.used_packers[0].space_utilization() * 100, 2)


def evaluate(population, agent):
    fitness_list = []

    for individual in population:
        fitness = _evaluate_individual(agent, individual)
        fitness_list.append(fitness)

    return fitness_list
