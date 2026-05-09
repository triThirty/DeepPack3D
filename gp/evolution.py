import random
import gc
import time

from deap import tools

from .util import record


def varAnd(population, toolbox, cxpb, mutpb, reppb):
    """Create offspring from population with minimal memory duplication."""
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
    """Evolve population with improved memory management."""
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    max_fitness = []
    best_ind_all_gen = []

    # Begin the generational process
    for gen in range(1, config.NGEN + 1):
        # Step 3: Full Fitness Evaluation
        start_time = time.time()
        print("----------------------------------")
        print(f"Evaluating generation {gen}...")
        fitnesses = toolbox.evaluate(population, agent)
        print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (fit,)
        del fitnesses  # Release memory

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

        # Generate exactly needed offspring (avoid over-allocation)
        offspring_needed = len(population) - config.ELITISM
        pop_intermediate = []
        while len(pop_intermediate) < offspring_needed:
            batch_size = min(len(parents), offspring_needed - len(pop_intermediate))
            parent_batch = parents[:batch_size]
            offspring_batch = varAnd(
                parent_batch, toolbox, config.CXPB, config.MUTPB, config.REPPB
            )
            pop_intermediate.extend(offspring_batch)
            del offspring_batch  # Release memory immediately

        # Replace the current population by the offspring
        population[:] = elitism_pop + pop_intermediate[:offspring_needed]
        del parents, elitism_pop, pop_intermediate  # Release memory

        # Periodic garbage collection to reclaim memory
        if gen % 2 == 0:
            gc.collect()

    return population, logbook, max_fitness, best_ind_all_gen, []
