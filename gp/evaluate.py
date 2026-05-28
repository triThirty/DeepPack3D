from .util.sequencing import GP_action_selector

from rl.split_gen import reset_rng


def _evaluate_once(agent, individual, gen, config, toolbox):
    reset_rng(seed=config.seed * gen)
    state = agent.env.reset()

    # step = 0
    return_v = 0
    while True:
        items, h_map, actions = state
        state_map = agent.Q_inputs(state)

        action = GP_action_selector(state_map, actions, individual[0], toolbox)
        next_state, reward, done = agent.env.step(action)
        return_v += reward
        # for i, packer in enumerate(agent.env.packers):
        #     packer.render().savefig(f"./outputs/{step}_{i}.jpg")
        # print(f"gen: {gen}, step: {step}, action: {action}, reward: {reward}, return: {return_v}")
        # step += 1

        if done:
            break
        state = next_state

    # return agent.env.used_packers[0].space_utilization() * 100
    return return_v


def _evaluate_individual(agent, individual, gen, config, toolbox, n_runs=1):
    total_fitness = 0.0

    for _ in range(n_runs):
        total_fitness += _evaluate_once(agent, individual, gen, config, toolbox)

    return round(total_fitness / n_runs, 4)


def evaluate(population, agent, gen, config, toolbo):

    fitness_list = []
    for i, individual in enumerate(population):
        # print(i)
        fitness = _evaluate_individual(agent, individual, gen, config, toolbo)
        fitness_list.append(fitness)

    return fitness_list
