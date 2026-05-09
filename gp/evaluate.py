import itertools

from .util.sequencing import GP_action_selector


def _evaluate_individual(agent, individual):
    state = agent.env.reset()

    for step in itertools.count():
        items, h_map, actions = state
        action = GP_action_selector(actions, individual[0])
        next_state, _, done = agent.env.step(action)

        if done:
            break
        state = next_state

    return round(agent.env.used_packers[0].space_utilization() * 100, 2)


def evaluate(population, agent):
    """Evaluate population sequentially with minimal memory overhead."""
    return [_evaluate_individual(agent, individual) for individual in population]
