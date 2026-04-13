# ==========================================================
# IMPORTS MODULE HERE
# ==========================================================
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# ==========================================================
# CONSTANT PARAMETERS
# ==========================================================
POP_SIZE = 100
GENERATIONS = 300
DIMENSION = 10
LOWER_BOUND = 0
UPPER_BOUND = 30
LIMIT = 5 
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
DISTRIBUTION_INDEX = 20
MUTATION_DISTRIBUTION_INDEX = 20

# ==========================================================
# INITIAL POPULATION (Real Values)
# ==========================================================
def generate_initial_population(pop_size, dimension):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(POP_SIZE, DIMENSION))


# ==========================================================
# FITNESS FUNCTION (MINIMIZATION)
# ==========================================================
def fitness(chromosome):
    return np.sum(chromosome ** 2)


# ==========================================================
# SELECT A PARENT 
# ==========================================================
def binary_tournament_selection(population, fitness, k=2, problem="min"):
    Np = len(population)
    mating_pool = []
    
    # Track how many times each individual is selected
    selection_count = np.zeros(Np, dtype=int)

    while len(mating_pool) < Np:
        # Get valid candidates (selected less than 2 times)
        valid_indices = np.where(selection_count < 2)[0]

        # If not enough candidates for tournament, break
        if len(valid_indices) < k:
            break

        # Step 1: Randomly pick k valid individuals
        indices = np.random.choice(valid_indices, k, replace=False)

        # Step 2: Get their fitness
        selected_fitness = fitness[indices]

        # Step 3: Select winner
        if problem == "max":
            winner_index = indices[np.argmax(selected_fitness)]
        else:
            winner_index = indices[np.argmin(selected_fitness)]

        # Step 4: Add to mating pool
        mating_pool.append(population[winner_index])
        selection_count[winner_index] += 1

    return np.array(mating_pool)


# ==========================================================
# SIMULATED BINARY CROSSOVER (SBX)
# ==========================================================
def crossover(p1, p2):
    # If random number >= crossover rate → children = parents
    if np.random.rand() >= CROSSOVER_RATE:
        return p1[:], p2[:]

    child1 = []
    child2 = []

    for x1, x2 in zip(p1, p2):
        u = np.random.rand()
        # Compute beta
        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (DISTRIBUTION_INDEX + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1.0 / (DISTRIBUTION_INDEX + 1))

        # Generate children
        c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
        c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)

        child1.append(c1)
        child2.append(c2)
    
    # Keep within bounds
    child1 = np.clip(child1, LOWER_BOUND, UPPER_BOUND)
    child2 = np.clip(child2, LOWER_BOUND, UPPER_BOUND)

    return np.array(child1), np.array(child2)


# ==========================================================
# POLYNOMIAL MUTATION
# ==========================================================
def mutate(chromosome):
    chromosome = chromosome.copy()

    if np.random.rand() >= MUTATION_RATE:
        return chromosome
    else:
        r = np.random.rand(DIMENSION)
        for i in range(len(chromosome)):
            if r[i] < 0.5:
                delta = (2 * r[i]) ** (1.0 / (MUTATION_DISTRIBUTION_INDEX + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r[i])) ** (1.0 / (MUTATION_DISTRIBUTION_INDEX + 1))

            # Apply mutation
            chromosome[i] = chromosome[i] + delta * (UPPER_BOUND - LOWER_BOUND)


    # Keep within bounds
    chromosome = np.clip(chromosome, LOWER_BOUND, UPPER_BOUND)

    return chromosome


# ==========================================================
# REAL-CODED GENETIC ALGORITHM (RCGA)
# ===========================================================
def real_coded_genetic_algorithm():
    # Initialize random population
    population = generate_initial_population(POP_SIZE, DIMENSION)

    # Evaluate fitness of the population
    fitness_values = np.array([
        fitness(ind) for ind in population
    ])

    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx]
    best_fitness = fitness_values[best_idx]
    # Best Fitness per generation
    best_fitness_per_gen = []


    for gen in range(GENERATIONS):
        mating_pool = binary_tournament_selection(population,fitness_values)
        offspring_population = []

        for i in range(POP_SIZE // 2):
            p1, p2 = mating_pool[np.random.choice(len(mating_pool), 2, replace=False)]
            
            c1, c2 = crossover(p1, p2)

            # offsprings are added
            offspring_population.append(c1)
            offspring_population.append(c2)

        # MUTATION
        for i in range(len(offspring_population)):
            offspring_population[i] = mutate(offspring_population[i])


        # Evaluate offspring fitness 
        offspring_fitness = [fitness(ind) for ind in offspring_population]

        # Combine
        combined_population = list(population) + offspring_population
        combined_fitness = list(fitness_values) + offspring_fitness

        # Sort
        sorted_indices = np.argsort(combined_fitness)

        # Select next generation
        population = np.array([combined_population[i] for i in sorted_indices[:POP_SIZE]])
        fitness_values = np.array([combined_fitness[i] for i in sorted_indices[:POP_SIZE]])

        # Best of this generation
        gen_best_fitness = fitness_values[0]
        best_fitness_per_gen.append(gen_best_fitness)

        # Update global best
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = population[0]

        # print(f"Generation {gen+1}: Best Fitness = {gen_best_fitness}")

    return best_solution, best_fitness, best_fitness_per_gen

# ==========================================================
# MULTIPLE RUNS
# ==========================================================
def solve_square_function():

    num_runs = 20

    all_histories = []
    all_best_sol = []
    all_best_costs = []
    all_times = []

    for run in range(num_runs):

        start_time = time.perf_counter()

        best_sol, best_cost, history = real_coded_genetic_algorithm()

        end_time = time.perf_counter()

        run_time = end_time - start_time

        all_histories.append(history)
        all_best_sol.append(best_sol)
        all_best_costs.append(best_cost)
        all_times.append(run_time)

        print(f"Run {run+1}: Best Cost = {best_cost:.6f}, Time = {run_time:.4f} sec")

    all_histories = np.array(all_histories)
    avg_curve = np.mean(all_histories, axis=0)

    # ==================================================
    # PLOT
    # ==================================================
    plt.figure()

    for i, history in enumerate(all_histories):
        plt.plot(history, alpha=0.6, label=f"Run {i+1}")


    plt.plot(avg_curve, linewidth=2, label="Average")

    plt.xlabel("Generation")
    plt.ylabel("Best Cost (Min)")
    plt.title("RCGA Convergence (Sphere Function)")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/RCGA_convergence.png", dpi=300)
    plt.show()


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    solve_square_function()