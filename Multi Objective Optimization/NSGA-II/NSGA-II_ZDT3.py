# ==========================================================
# IMPORTS MODULE HERE
# ==========================================================
import numpy as np
import time
import matplotlib.pyplot as plt

# ==========================================================
# CONSTANT PARAMETERS
# ==========================================================
POP_SIZE = 100
GENERATIONS = 200
DIMENSION = 30
LOWER_BOUND = 0
UPPER_BOUND = 1
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
SBX_DISTRIBUTION_INDEX = 15
POLYNOMIAL_MUTATION_DISTRIBUTION_INDEX = 20

# ==========================================================
# INITIAL POPULATION (Real Values)
# ==========================================================
def generate_initial_population():
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(POP_SIZE, DIMENSION))


# ==========================================================
# FITNESS FUNCTION 
# ==========================================================
def fitness(chromosome):
    chromosome = np.array(chromosome)
    n = len(chromosome)
    
    # f1
    f1 = chromosome[0]
    
    # g(x)
    g = 1 + (9 / (n - 1)) * np.sum(chromosome[1:])
    
    # h(f1, g)
    h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    
    # f2
    f2 = g * h
    
    return f1, f2


# ==========================================================
# WHETHER A DOMINATES B 
# ==========================================================
def dominates(p, q):
    """Return True if p dominates q (minimization)."""
    p, q = np.asarray(p), np.asarray(q)
    return bool(np.all(p <= q) and np.any(p < q))


# ==========================================================
# RANK CALCULATION
# ==========================================================
def fast_non_dominated_sort(fitness_values):
    """
    fitness_values: numpy array of shape (N, M) — objective values, not decision vectors.
    returns: list of fronts [[indices], [indices], ...]
    """
    N = len(fitness_values)
    
    S = [[] for _ in range(N)]   # S[p] = solutions dominated by p
    n = [0] * N                  # n[p] = number of solutions dominating p
    rank = [0] * N
    
    fronts = [[]]  # F1

    # 🔹 Stage 1
    for p in range(N):
        S[p] = []
        n[p] = 0
        
        for q in range(N):
            if dominates(fitness_values[p], fitness_values[q]):
                S[p].append(q)
            elif dominates(fitness_values[q], fitness_values[p]):
                n[p] += 1
        
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    # 🔹 Stage 2
    i = 0
    while fronts[i]:
        Q = []
        
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        
        i += 1
        fronts.append(Q)

    # Remove last empty front
    fronts.pop()

    return fronts, rank


# ==========================================================
# CROWDING DISTANCE
# ==========================================================
def crowding_distance(front, fitness_values):
    """
    front: list/array of indices
    fitness_values: numpy array of shape (N, M) — objective values
    
    returns: numpy array of crowding distances (aligned with 'front')
    """
    if len(front) == 0:
        return np.array([])

    front = np.array(front)
    F = fitness_values[front]   # shape (r, M)
    
    r, M = F.shape

    # A single-member front trivially gets infinite distance
    if r == 1:
        return np.array([np.inf])

    distance = np.zeros(r)
    
    for m in range(M):
        # Sort indices based on objective m
        sorted_idx = np.argsort(F[:, m])
        sorted_F = F[sorted_idx]
        
        # Assign infinity to boundary points
        distance[sorted_idx[0]] = np.inf
        distance[sorted_idx[-1]] = np.inf
        
        f_min = sorted_F[0, m]
        f_max = sorted_F[-1, m]
        
        if f_max == f_min:
            continue
        
        # Vectorized computation for middle points
        prev_vals = sorted_F[:-2, m]
        next_vals = sorted_F[2:, m]
        
        norm = (f_max - f_min)
        increments = (next_vals - prev_vals) / norm
        
        # Add to distances (skip first & last)
        distance[sorted_idx[1:-1]] += increments
    
    return distance


# ==========================================================
# CROWDED BINARY TOURNAMENT
# ==========================================================
def crowded_binary_tournament(pop_size, ranks, cd):
    """
    Each tournament draws two *distinct* competitors to ensure
    genuine head-to-head comparison.
    """
    N = len(ranks)

    # Draw two distinct competitors per tournament
    i_arr = np.random.randint(0, N, pop_size)
    j_arr = np.random.randint(0, N, pop_size)
    collisions = i_arr == j_arr
    j_arr[collisions] = (j_arr[collisions] + np.random.randint(1, N, collisions.sum())) % N

    winners = np.where(
        ranks[i_arr] < ranks[j_arr], i_arr,
        np.where(
            ranks[j_arr] < ranks[i_arr], j_arr,
            np.where(
                cd[i_arr] > cd[j_arr], i_arr,
                np.where(
                    cd[j_arr] > cd[i_arr], j_arr,
                    np.where(np.random.rand(pop_size) < 0.5, i_arr, j_arr)
                )
            )
        )
    )
    
    return winners


# ==========================================================
# SELECTION
# ==========================================================
def selection(population, fitness_values):
    """
    NSGA-II Selection using:
    - Non-dominated sorting (rank)
    - Crowding distance
    - Binary tournament

    Returns:
        mating_pool (numpy array of selected individuals)
    """

    # Non-dominated sorting
    fronts, rank = fast_non_dominated_sort(fitness_values)

    # Crowding distance
    cd = np.zeros(len(population))
    for front in fronts:
        cd[front] = crowding_distance(front, fitness_values)

    # Tournament selection
    selected_indices = crowded_binary_tournament(
        pop_size=len(population),
        ranks=np.array(rank),
        cd=cd
    )

    mating_pool = population[selected_indices]

    return mating_pool


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
            beta = (2 * u) ** (1.0 / (SBX_DISTRIBUTION_INDEX + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1.0 / (SBX_DISTRIBUTION_INDEX + 1))

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
                delta = (2 * r[i]) ** (1.0 / (POLYNOMIAL_MUTATION_DISTRIBUTION_INDEX + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r[i])) ** (1.0 / (POLYNOMIAL_MUTATION_DISTRIBUTION_INDEX + 1))

            # Apply mutation
            chromosome[i] = chromosome[i] + delta * (UPPER_BOUND - LOWER_BOUND)


    # Keep within bounds
    chromosome = np.clip(chromosome, LOWER_BOUND, UPPER_BOUND)

    return chromosome


# ==========================================================
# VARIATION
# ==========================================================
def variation(mating_pool):
    offspring = []

    # Iterate over consecutive pairs; stop one short to avoid overrun on odd-length pools
    for i in range(0, len(mating_pool) - 1, 2):
        p1 = mating_pool[i]
        p2 = mating_pool[i + 1]

        c1, c2 = crossover(p1, p2)
        offspring.append(mutate(c1))
        offspring.append(mutate(c2))

    # Trim to exactly POP_SIZE regardless of pool parity
    return np.array(offspring[:POP_SIZE])


def survivor_selection(population, fitness_values, offspring, offspring_fitness):
    """
    NSGA-II Survivor Selection

    Combines parent + offspring and selects next generation using:
    - Non-dominated sorting (rank)
    - Crowding distance (diversity)

    Returns:
        new_population, new_fitness
    """

    # Combine populations
    combined_population = np.vstack((population, offspring))
    combined_fitness = np.vstack((fitness_values, offspring_fitness))

    # Non-dominated sorting
    fronts, _ = fast_non_dominated_sort(combined_fitness)

    new_population = []
    new_fitness = []

    # Fill next generation
    for front in fronts:
        if len(new_population) + len(front) <= POP_SIZE:
            new_population.extend(combined_population[front])
            new_fitness.extend(combined_fitness[front])
        else:
            # Compute crowding distance for this front
            cd_front = crowding_distance(front, combined_fitness)

            # Sort by descending crowding distance
            sorted_idx = np.argsort(-cd_front)

            remaining = POP_SIZE - len(new_population)

            selected = [front[i] for i in sorted_idx[:remaining]]

            new_population.extend(combined_population[selected])
            new_fitness.extend(combined_fitness[selected])
            break

    return np.array(new_population), np.array(new_fitness)


# ===========================================================
# NON-DOMINATED SORTING GENETIC ALGORITHM (NSGA - II)
# ===========================================================
def non_dominated_sorting_genetic_algorithm():
    # Initial random population
    population = generate_initial_population()
    fitness_values = np.array([fitness(ind) for ind in population])

    # Store initial population (for plotting)
    initial_fitness = fitness_values.copy()

    for gen in range(GENERATIONS):
        mating_pool = selection(population, fitness_values)

        offspring = variation(mating_pool)
        offspring_fitness = np.array([fitness(ind) for ind in offspring])

        population, fitness_values = survivor_selection(
            population, fitness_values,
            offspring, offspring_fitness
        )

    return initial_fitness, fitness_values


# ==================================================
# PLOT
# ==================================================
def plot_nsga2_results(initial_fitness, final_fitness):

    # Extract Pareto front
    fronts, _ = fast_non_dominated_sort(final_fitness)
    pareto = final_fitness[fronts[0]]

    # True Pareto front (ZDT3)
    # ZDT3 — filter to only the non-dominated portion of the curve
    true_f1 = np.linspace(0, 1, 1000)
    true_f2 = 1 - np.sqrt(true_f1) - true_f1 * np.sin(10 * np.pi * true_f1)

    # Keep only non-dominated points from the curve
    mask = np.ones(len(true_f1), dtype=bool)
    for i in range(len(true_f1)):
        for j in range(len(true_f1)):
            if i != j and true_f1[j] <= true_f1[i] and true_f2[j] <= true_f2[i]:
                if true_f1[j] < true_f1[i] or true_f2[j] < true_f2[i]:
                    mask[i] = False
                    break

    true_f1 = true_f1[mask]
    true_f2 = true_f2[mask]

    plt.figure(figsize=(12, 5))

    # LEFT
    plt.subplot(1, 2, 1)
    plt.scatter(initial_fitness[:, 0], initial_fitness[:, 1], s=15, alpha=0.6, label='Initial population')
    plt.scatter(true_f1, true_f2, s=3, c='red', label='True Pareto front')
    plt.title("Initial Population")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(alpha=0.3)
    plt.legend()

    # RIGHT
    plt.subplot(1, 2, 2)
    plt.scatter(pareto[:, 0], pareto[:, 1], s=20, c='green', alpha=0.7, label='NSGA-II Pareto front (non-dominated)')
    plt.scatter(true_f1, true_f2, s=3, c='red', label='True Pareto front')
    plt.title("Obtained Non-Dominated Solutions")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(alpha=0.3)
    plt.legend()

    # MAIN TITLE
    plt.suptitle("NSGA-II Performance on ZDT3", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    

# ==========================================================
# MULTIPLE RUNS
# ==========================================================
def run_nsga2_analysis():
    num_runs = 1

    for run in range(num_runs):
        
        start_time = time.perf_counter()

        initial_fitness, final_fitness = non_dominated_sorting_genetic_algorithm()
        
        end_time = time.perf_counter()

        run_time = end_time - start_time
        
        plot_nsga2_results(initial_fitness, final_fitness)

        print(f"Run {run+1}: Time = {run_time:.4f} sec")



# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    run_nsga2_analysis()