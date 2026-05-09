import random

from deap import tools

from .util import record


def varAnd(population, toolbox, cxpb, mutpb, reppb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb = cxpb / (cxpb + mutpb + reppb)
    new_mutpb = mutpb / (cxpb + mutpb + reppb) + new_cxpb
    i = 1
    while i < len(offspring):
        randomValue = random.random()
        if randomValue < new_cxpb:  # crossover
            if offspring[i - 1] == offspring[i]:
                (offspring[i - 1],) = toolbox.mutate(offspring[i - 1])
                (offspring[i],) = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(
                    offspring[i - 1], offspring[i]
                )
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        elif new_cxpb <= randomValue < new_mutpb:  # mutation
            (offspring[i - 1],) = toolbox.mutate(offspring[i - 1])
            del offspring[i - 1].fitness.values
            i = i + 1
        else:
            i = i + 1
    return offspring


def evolve(population, agent, toolbox, stats, hof, config):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    max_fitness = []
    best_ind_all_gen = []
    all_individuals = []
    # Begin the generational process
    for gen in range(1, config.NGEN + 1):
        # Step 3: Full Fitness Evaluation
        # fitnesses = toolbox.multiProcess(
        fitnesses = toolbox.evaluate(population, agent)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (fit,)
        # Step 3: Full Fitness Evaluation

        record(
            hof,
            population,
            gen,
            stats,
            logbook,
            True,
            config,
            max_fitness,
            best_ind_all_gen,
        )

        # Step 5-7: Produce Offspring from population in intermediate population
        parents = toolbox.select(population, len(population))  # Select parents
        elitism_pop = tools.selBest(population, config.ELITISM)  # Select elitism
        pop_intermediate = []
        while len(pop_intermediate) < len(population):
            offspring_intermediate = varAnd(
                parents, toolbox, config.CXPB, config.MUTPB, config.REPPB
            )
            pop_intermediate.extend(offspring_intermediate)
            del offspring_intermediate
        pop_intermediate[:] = pop_intermediate[: len(population) - config.ELITISM]
        # Step 5-7: Produce Offspring from population in intermediate population

        # Replace the current population by the offspring
        population[:] = elitism_pop + pop_intermediate

    return population, logbook, max_fitness, best_ind_all_gen, all_individuals
