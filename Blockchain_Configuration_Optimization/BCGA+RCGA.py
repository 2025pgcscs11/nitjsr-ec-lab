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
GENERATIONS = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
SBX_DISTRIBUTION_INDEX = 20
POLYNOMIAL_MUTATION_INDEX = 20 


# ==========================================================
# INITIAL POPULATION (BINARY + REAL ENCODED)
# ==========================================================
def generate_initial_population(params):
    N = params["N"]
    t_min = params["t_min"]
    t_max = params["t_max"]
    population = []
    for _ in range(POP_SIZE):
        # Binary part: N bits for validators
        validator_bits = np.random.randint(0, 2, size=N)

        # Real part: single float for T
        T = np.random.uniform(t_min, t_max)
        
        # Chromosome = concatenation of binary array and float
        chromosome = np.concatenate([validator_bits.astype(float), [T]])

        population.append(chromosome)
    
    return np.array(population)


# ==========================================================
# DECODE CHROMOSOME
# ==========================================================
def decode_chromosome(chromosome, params):
    N = params["N"]
    t_min = params["t_min"]
    t_max = params["t_max"]

    # First N entries are binary (0/1) for validators
    validator_bits = chromosome[:N]
    selected = np.where(validator_bits >= 0.5)[0]   # threshold 0.5
    
    # Last entry is T (real)
    T = chromosome[-1]
    T = int(round(np.clip(T, t_min, t_max)))
    
    return selected, T


# ==========================================================
# FITNESS FUNCTION (Minimization)
# ==========================================================
def fitness(chromosome, params):
    selected, T = decode_chromosome(chromosome, params)

    phi = params["phi"]
    f = len(selected)

    if f < params["f_min"]:
        return 1e9 

    # COST
    cost = sum(phi[i] for i in selected) / T

    # SECURITY
    security = params["tau"] * (f ** params["i"])

    # LATENCY
    latency = (
        (T * params["X"]) / params["jd"]
        + max(params["Z"] / phi[i] for i in selected)
        + params["mu"] * (T * params["X"]) * f
        + params["delta"] / params["ju"]
    )

    # NORMALIZATION
    # ----- Log normalization for Cost -----
    cost_log_actual = np.log(cost)
    cost_log_min = np.log(params["cost_min"])
    cost_log_max = np.log(params["cost_max"])
    cost_n = (cost_log_actual - cost_log_min) / (cost_log_max - cost_log_min + 1e-9)

    # ----- Log normalization for Latency -----
    lat_log_actual = np.log(latency)
    lat_log_min = np.log(params["lat_min"])
    lat_log_max = np.log(params["lat_max"])
    lat_n = (lat_log_actual - lat_log_min) / (lat_log_max - lat_log_min + 1e-9)

    # ----- Log normalization for Security (then invert) -----
    sec_log_actual = np.log(security)
    sec_log_min = np.log(params["sec_min"])
    sec_log_max = np.log(params["sec_max"])
    sec_raw = (sec_log_actual - sec_log_min) / (sec_log_max - sec_log_min + 1e-9)
    sec_n = 1 - sec_raw   # because security is to be maximized

    # Finally, the overall objective (to minimize)
    obj = params["w1"] * lat_n + params["w2"] * sec_n + params["w3"] * cost_n

    return obj


# ==========================================================
# SELECT A PARENT 
# ==========================================================
def binary_tournament_selection(population, fitness_values, k=2, problem="min"):
    population = np.asarray(population)
    fitness_values = np.asarray(fitness_values)
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
        indices = np.random.choice(valid_indices, k, replace=False).astype(int)

        # Step 2: Get their fitness
        selected_fitness = fitness_values[indices]

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
# BINARY CROSSOVER (for validator bits) – e.g., single-point
# ==========================================================
def binary_crossover(parent1_bits, parent2_bits):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, len(parent1_bits))
        child1_bits = np.concatenate((parent1_bits[:point], parent2_bits[point:]))
        child2_bits = np.concatenate((parent2_bits[:point], parent1_bits[point:]))
        return child1_bits, child2_bits
    return parent1_bits.copy(), parent2_bits.copy()


# ==========================================================
# BINARY MUTATION (for validator bits)
# ==========================================================
def binary_mutation(bits):
    for i in range(len(bits)):
        if np.random.rand() < MUTATION_RATE:
            bits[i] = 1 - bits[i]
    return bits


# ==========================================================
# REAL SBX CROSSOVER (for T)
# ==========================================================
def real_sbx_crossover(t1, t2, t_min, t_max):
    if np.random.rand() >= CROSSOVER_RATE:
        return t1, t2
    u = np.random.rand()
    if u <= 0.5:
        beta = (2 * u) ** (1.0 / (SBX_DISTRIBUTION_INDEX + 1))
    else:
        beta = (1 / (2 * (1 - u))) ** (1.0 / (SBX_DISTRIBUTION_INDEX + 1))
    child1 = 0.5 * ((1 + beta) * t1 + (1 - beta) * t2)
    child2 = 0.5 * ((1 - beta) * t1 + (1 + beta) * t2)
    child1 = np.clip(child1, t_min, t_max)
    child2 = np.clip(child2, t_min, t_max)
    return child1, child2


# ==========================================================
# REAL POLYNOMIAL MUTATION (for T)
# ==========================================================
def real_polynomial_mutation(t, t_min, t_max):
    if np.random.rand() >= MUTATION_RATE:
        return t
    delta = 0.0
    r = np.random.rand()
    if r < 0.5:
        delta = (2 * r) ** (1.0 / (POLYNOMIAL_MUTATION_INDEX + 1)) - 1
    else:
        delta = 1 - (2 * (1 - r)) ** (1.0 / (POLYNOMIAL_MUTATION_INDEX + 1))
    t_new = t + delta * (t_max - t_min)
    return np.clip(t_new, t_min, t_max)


# ==========================================================
# CROSSOVER (handles both parts)
# ==========================================================
def crossover(parent1, parent2, params):
    if np.random.rand() >= CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    
    N = params["N"]
    t_min = params["t_min"]
    t_max = params["t_max"]
    # Split: first N bits = validator part, last element = T
    p1_bits = parent1[:N]
    p2_bits = parent2[:N]
    p1_T = parent1[-1]
    p2_T = parent2[-1]
    # Binary crossover for validator bits
    c1_bits, c2_bits = binary_crossover(p1_bits, p2_bits)
    # Real crossover for T
    c1_T, c2_T = real_sbx_crossover(p1_T, p2_T, t_min, t_max)
    # Rebuild children
    child1 = np.concatenate([c1_bits, [c1_T]])
    child2 = np.concatenate([c2_bits, [c2_T]])
    return child1, child2


# ==========================================================
# MUTATION (handles both parts)
# ==========================================================
def mutate(chromosome, params):
    N = params["N"]
    t_min = params["t_min"]
    t_max = params["t_max"]
    # Mutate validator bits
    mutated_bits = binary_mutation(chromosome[:N].copy())
    # Mutate T
    mutated_T = real_polynomial_mutation(chromosome[-1], t_min, t_max)
    return np.concatenate([mutated_bits, [mutated_T]])


# ==========================================================
# GENETIC ALGORITHM MAIN LOOP (Hybrid)
# ==========================================================
def hybrid_genetic_algorithm(params):
    # Generate Initial Population
    population = generate_initial_population(params)

    # Evaluate fitness values
    fitness_values = [fitness(chrom, params) for chrom in population]
    
    # Store best chromosome and its fitness value
    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx].copy()
    best_fitness = fitness_values[best_idx]

     # Best Fitness per generation
    best_fitness_per_gen = []


    for gen in range(GENERATIONS):
        mating_pool = binary_tournament_selection(population, fitness_values)
        offspring = []
        for i in range(POP_SIZE // 2):
            # SELECTION
            p1, p2 = mating_pool[np.random.choice(len(mating_pool), 2, replace=False)]
            
            # CROSSOVER
            c1, c2 = crossover(p1, p2, params)
            
            # MUTATION
            c1 = mutate(c1, params)
            c2 = mutate(c2, params)

            # Offsprings are added
            offspring.extend([c1, c2])

        # Evaluate offspring fitness 
        offspring_fitness = [fitness(ind, params) for ind in offspring]

        # Combine and select best POP_SIZE
        combined_pop = list(population) + offspring
        combined_fit = list(fitness_values) + offspring_fitness

        # Sort
        sorted_idx = np.argsort(combined_fit)
        population = [combined_pop[i] for i in sorted_idx[:POP_SIZE]]
        fitness_values = [combined_fit[i] for i in sorted_idx[:POP_SIZE]]
        
        # Best of this generation
        gen_best = min(fitness_values)
        
        best_fitness_per_gen.append(gen_best)
        
        # Update global best
        if gen_best < best_fitness:
            best_fitness = gen_best
            best_solution = population[0].copy()
        

        # val, T =decode_chromosome(population[0],params)
        # print(f"Generation {gen+1}:  Best Fitness = {gen_best_fitness}    Number of Validators = {len(val)}   Number of Transactions = {T}")
    
    return best_solution, best_fitness, best_fitness_per_gen


# ==========================================================
# READ BCO FILE
# ==========================================================
def read_bco_file(filename):
    instances = []

    with open(filename, 'r') as f:
        data = list(map(float, f.read().split()))

    idx = 0
    P = int(data[idx])
    idx += 1

    for _ in range(P):
        f_min = int(data[idx])
        f_max = int(data[idx + 1])
        t_min = int(data[idx + 2])
        t_max = int(data[idx + 3])
        idx += 4

        phi = data[idx: idx + f_max]
        idx += f_max

        # Validation added
        if len(phi) != f_max:
            raise ValueError("Mismatch in phi length and f_max")

        instance = {
            "f_min": f_min,
            "f_max": f_max,
            "t_min": t_min,
            "t_max": t_max,
            "phi": phi,
            "N": len(phi)
        }

        instances.append(instance)

    return instances



# ==================================================================
# ITERATE OVER ALL INSTANCES IN A FILE AND APPLY GENETIC ALGORITHM
# ==================================================================
def solve_gap_file(filename):
    instances = read_bco_file(filename)
    results = []

    print(f"\n===== Solving file: {filename} =====\n")

    for idx, instance in enumerate(instances, start=1):
        print(f"\nSetting {idx}:")

        params = instance

        # CONSTANTS
        params.update({
            "tau": 1,
            "i": 4,
            "delta": 0.5,
            "jd": 7.2,
            "ju": 7.3,
            "Z": 100,
            "X": 0.5,
            "mu": 0.007,
            "w1": 0.33,
            "w2": 0.33,
            "w3": 0.34
        })

        # cost min, max calculation
        phi = np.array(params["phi"])
        # Sort phi
        phi_sorted = np.sort(phi)
        # COST
        cost_max = np.sum(phi) / params["t_min"]
        cost_min = np.sum(phi_sorted[:params["f_min"]]) / params["t_max"]

        
        # security min,max calculation
        tau = params["tau"]
        q = params["i"]

        sec_max = tau * (params["f_max"] ** q)
        sec_min = tau * (params["f_min"] ** q)

        
       # latency min, max calculation
        X = params["X"]
        rd = params["jd"]
        ru = params["ju"]
        K = params["Z"]
        psi = params["mu"]
        O = params["delta"]

        # LATENCY MAX
        lat_max = (
            (params["t_max"] * X) / rd
            + K / np.min(phi)
            +   psi * (params["t_max"] * X) * params["f_max"]
            + O / ru
        )

        # LATENCY MIN
        phi_desc = np.sort(phi)[::-1]  # descending
        vth_largest = phi_desc[params["f_min"] - 1]

        lat_min = (
            (params["t_min"] * X) / rd
        + K / vth_largest
        + psi * (params["t_min"] * X) * params["f_min"]
        + O / ru
        )

        # all max, min values are added
        params["cost_min"] = cost_min
        params["cost_max"] = cost_max

        params["sec_min"] = sec_min
        params["sec_max"] = sec_max

        params["lat_min"] = lat_min
        params["lat_max"] = lat_max

        
        num_runs = 20
        all_histories = []
        all_best_sol = []
        all_best_utilitys = []
        all_times = []

        # Run GA multiple times
        for run in range(num_runs):
            start_time = time.perf_counter()

            best_assignment, best_utility, fitness_per_gen = hybrid_genetic_algorithm(params)

            end_time = time.perf_counter()

            run_time = end_time - start_time

            all_histories.append(fitness_per_gen)
            all_best_sol.append(best_assignment)
            all_best_utilitys.append(best_utility)
            all_times.append(run_time)

            val, T =decode_chromosome(best_assignment,params)
            print(f"  Run {run+1}: Best Utility = {best_utility}, No. of Validators = {len(val)}, No. of Transactions = {T}, Best Time = {run_time:.4f} sec")

        # Convert to numpy array for easier computation
        all_histories = np.array(all_histories)

        # Compute average convergence
        avg_fitness = np.mean(all_histories, axis=0)

        # ==========================
        # Plot for THIS instance
        # ==========================
        plt.figure()

        # Plot all runs (light)
        for i, history in enumerate(all_histories):
            plt.plot(history, alpha=0.4, label=f"Run {i+1}")

        # Plot average (bold)
        plt.plot(avg_fitness, linewidth=2, label="Average")

        plt.xlabel("GENERATION")
        plt.ylabel("BEST UTILITY VALUE")
        plt.title(f"CONVERGENCE GRAPH || HYBRID GENETIC ALGORITHM (BCGA + RCGA) || BLOCKCHAIN CONFIGURATION OPTIMIZATION || {os.path.splitext(os.path.basename(filename))[0]} || Setting {idx}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # os.makedirs("plots", exist_ok=True)
        # plt.savefig(f"plots/{os.path.splitext(os.path.basename(filename))[0]}_setting_{idx}_Hybrid_GA_convergence.png", dpi=300)
        plt.show()

        # Store results
        results.append({
            "histories": all_histories,
            "best_utility_per_run": all_best_utilitys,
            "best_solution_per_run": all_best_sol,
            "time_per_run": all_times,
            "params" : params
        })

    return results

# ==========================================================
# ITERATE OVER ALL FILES
# ==========================================================
def solve_multiple_files(file_list,base_dir="bco_dataset"):
    all_results = {}
    
    # Absolute path of current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Full path to dataset folder
    dataset_dir = os.path.join(script_dir, base_dir)

    for file in file_list:
        file_path = os.path.join(dataset_dir,file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GAP file not found: {file_path}")
        
        all_results[file] = solve_gap_file(file_path)

    return all_results


# ==========================================================
# ALL FILE NAMES
# ==========================================================
files = [
    "bco1.txt",
]


# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__": 
    all_results = solve_multiple_files(files)

    for file, instances in all_results.items():
        print(f"\n===== SUMMARY FOR FILE: {file} =====")

        for idx, instance in enumerate(instances, start=1):

            utilities = np.array(instance["best_utility_per_run"])
            times = np.array(instance["time_per_run"])
            solutions = np.array(instance["best_solution_per_run"])
            params = instance["params"]
            print(f"\n--- Setiing {idx} ---")

            # Compute stats
            avg_utility = np.mean(utilities)
            std_utility = np.std(utilities)
            best_utility_index = np.argmin(utilities)
            best_solution = solutions[best_utility_index]
            best_utility = utilities[best_utility_index]
            worst_utility = np.max(utilities)
            number_of_validators, T = decode_chromosome(best_solution,params)
            avg_time = np.mean(times)
            total_time = np.sum(times)

            # Print
            print(f"Number of Validators   : {len(number_of_validators)}")
            print(f"Number of Transactions : {T}")
            print(f"Average utility        : {avg_utility:} ± {std_utility:}")
            print(f"Best utility           : {best_utility:}")
            print(f"Worst utility          : {worst_utility:}")
            print(f"Average time           : {avg_time:.4f} s")
            print(f"Total time             : {total_time:.4f} s")