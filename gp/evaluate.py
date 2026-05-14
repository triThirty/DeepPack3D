from .util.sequencing import GP_action_selector

from rl.split_gen import reset_rng


def _evaluate_once(agent, individual, gen, config):
    reset_rng(seed=config.seed * gen * 27)
    state = agent.env.reset()

    step = 0
    while True:
        items, h_map, actions = state
        # print(f"the available cuboid space: {len(actions[0][0])}")
        action = GP_action_selector(actions, individual[0])
        next_state, _, done = agent.env.step(action)
        # for i, packer in enumerate(agent.env.packers):
        #     packer.render().savefig(f"./outputs/{step}_{i}.jpg")
        step += 1

        if done:
            break
        state = next_state

    return agent.env.used_packers[0].space_utilization() * 100


def _evaluate_individual(agent, individual, gen, config, n_runs=1):
    total_fitness = 0.0

    for _ in range(n_runs):
        total_fitness += _evaluate_once(agent, individual, gen, config)

    return round(total_fitness / n_runs, 2)


def evaluate(population, agent, gen, config):

    fitness_list = []
    for individual in population:
        fitness = _evaluate_individual(agent, individual, gen, config)
        fitness_list.append(fitness)

    return fitness_list
