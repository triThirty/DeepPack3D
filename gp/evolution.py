from deap import tools


def evolve(population, agent, toolbox, stats, hof, config):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    min_fitness = []
    best_ind_all_gen = []
    all_individuals = []
    # Begin the generational process
    for gen in range(1, config.NGEN + 1):
        # Step 3: Full Fitness Evaluation
        # fitnesses = toolbox.multiProcess(
        fitnesses = toolbox.evaluate(population, agent)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        # Step 3: Full Fitness Evaluation

        # record(
        #     halloffame,
        #     population,
        #     gen,
        #     stats,
        #     logbook,
        #     verbose,
        #     config,
        #     min_fitness,
        #     best_ind_all_gen,
        # )

        # Step 5-7: Produce Offspring from population in intermediate population
        parents = toolbox.select(population, len(population))  # Select parents
        elitism_pop = tools.selBest(population, elitism)  # Select elitism
        pop_intermediate = []
        while len(pop_intermediate) < len(population):
            offspring_intermediate = varAnd(parents, toolbox, cxpb, mutpb, reppb)
            pop_intermediate.extend(offspring_intermediate)
            del offspring_intermediate
        pop_intermediate[:] = pop_intermediate[: len(population) - elitism]
        # Step 5-7: Produce Offspring from population in intermediate population

        # Replace the current population by the offspring
        population[:] = elitism_pop + pop_intermediate

    return population, logbook, min_fitness, best_ind_all_gen, all_individuals
