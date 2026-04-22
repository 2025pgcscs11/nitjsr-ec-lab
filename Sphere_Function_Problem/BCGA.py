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
GENERATIONS = 200
DIMENSION = 10
LOWER_BOUND = 0
UPPER_BOUND = 30
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population():
    # Number of bits needed to represent values up to UPPER_BOUND
    num_bits = int(np.ceil(np.log2(UPPER_BOUND)))

    # Generate random integers in [0, 1]
    population = np.random.randint(0, 2, size=(POP_SIZE, num_bits * DIMENSION))
   
    return np.array(population)


# ==========================================================
# DECODE CHROMOSOME AND EXTRACT AGENTS
# ==========================================================
def decode_chromosome(chromosome):
    num_bits = int(np.ceil(np.log2(UPPER_BOUND)))
    agents = []

    for j in range(len(chromosome) // num_bits):
        start = j * num_bits
        end = start + num_bits

        value = 0
        for bit in chromosome[start:end]:
            value = (value << 1) | bit

        # Keep within bounds
        precision = (UPPER_BOUND - LOWER_BOUND) / (2 ** num_bits -1)
        value = LOWER_BOUND + precision * value

        agents.append(value)

    return agents


# ==========================================================
# FITNESS FUNCTION (MINIMIZATION)
# ==========================================================
def fitness(chromosome):
    decoded = np.array(decode_chromosome(chromosome))
    return np.sum(decoded ** 2)


# ==========================================================
# SELECT A PARENT 
# ==========================================================
def binary_tournament_selection(population, fitness_values, k=2, problem="min"):
    Np = len(population)
    mating_pool = []
    selection_count = np.zeros(Np, dtype=int)
    
    while len(mating_pool) < Np:
        # Get indices that haven't reached max selections (2)
        available = np.where(selection_count < 2)[0]
        
        if len(available) < k:
            # Reset selection counts if not enough individuals
            selection_count = np.zeros(Np, dtype=int)
            available = np.where(selection_count < 2)[0]
        
        # Tournament selection
        participants = np.random.choice(available, k, replace=False)
        
        if problem == "min":
            winner = participants[np.argmin(fitness_values[participants])]
        else:
            winner = participants[np.argmax(fitness_values[participants])]
        
        mating_pool.append(population[winner])
        selection_count[winner] += 1
    
    return np.array(mating_pool)

# def k_tournament_selection(population, fitness_values, k=2, problem="min"):
#     Np = len(population)
#     mating_pool = []

#     selection_count = np.zeros(Np, dtype=int)

#     for _ in range(Np):
#         # pick k participants who have been selected < k times
#         participants = []

#         while len(participants) < k:
#             idx = np.random.randint(0, Np)
#             if selection_count[idx] < k and idx not in participants:
#                 participants.append(idx)
#                 selection_count[idx] += 1

#         participants = np.array(participants)

#         # choose winner
#         if problem == "min":
#             winner = participants[np.argmin(fitness_values[participants])]
#         else:
#             winner = participants[np.argmax(fitness_values[participants])]

#         mating_pool.append(population[winner])

#     return np.array(mating_pool)


# ==========================================================
# CROSSOVER ON TWO PARENTS (RANDOM BIT POINTS)
# ==========================================================
def crossover(p1, p2):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, len(p1) - 2)
        return (
        np.concatenate((p1[:point], p2[point:])),
        np.concatenate((p2[:point], p1[point:]))
        )
    return p1.copy(), p2.copy()


# ==========================================================
# MUTATION IN A CHROMOSOME (BIT-WISE)
# ==========================================================
def mutate(chromosome):
    chromosome = chromosome.copy()  
    for i in range(len(chromosome)):
        if np.random.rand() < MUTATION_RATE:
            chromosome[i] ^= 1
    return chromosome


# ==========================================================
# BINARY-CODED GENETIC ALGORITHM (BCGA)
# ==========================================================
def binary_coded_genetic_algorithm():
    # Initialize random population
    population = generate_initial_population()

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

        # CROSSOVER
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
        gen_best_fitness = combined_fitness[sorted_indices[0]]
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
    results = []
    num_runs = 20

    all_histories = []
    all_best_sol = []
    all_best_values = []
    all_times = []

    for run in range(num_runs):

        start_time = time.perf_counter()

        best_sol, best_value, history = binary_coded_genetic_algorithm()

        end_time = time.perf_counter()

        run_time = end_time - start_time

        all_histories.append(history)
        all_best_sol.append(best_sol)
        all_best_values.append(best_value)
        all_times.append(run_time)

        print(f"Run {run+1}: Best value = {best_value:.6f}, Time = {run_time:.4f} sec")

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
    plt.ylabel("Best value (Min)")
    plt.title("BCGA Convergence (Sphere Function)")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/BCGA_convergence.png", dpi=300)
    plt.show()

    # Store results
    results.append({
        "histories": all_histories,
        "best_cost_per_run": all_best_values,
        "best_solution_per_run": all_best_sol,
        "time_per_run": all_times,
    })

    return results



# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    all_results = solve_square_function()

    for result in all_results:

        best_values = np.array(result["best_cost_per_run"])
        times = np.array(result["time_per_run"])

        print("\n===== FINAL SUMMARY (SPHERE FUNCTION) =====")

        print(f"Number of runs        : {len(best_values)}")
        print(f"Best fitness (min)    : {np.min(best_values):.6f}")
        print(f"Worst fitness         : {np.max(best_values):.6f}")
        print(f"Average fitness       : {np.mean(best_values):.6f}")
        print(f"Std deviation         : {np.std(best_values):.6f}")

        print(f"\nAverage time/run      : {np.mean(times):.4f} sec")
        print(f"Total execution time  : {np.sum(times):.4f} sec")