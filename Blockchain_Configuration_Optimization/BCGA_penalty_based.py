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
POP_SIZE = 300
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# ==========================================================
# INITIAL POPULATION (BINARY ENCODED)
# ==========================================================
def generate_initial_population(params):
    N = params["N"]
    t_min = params["t_min"]
    t_max = params["t_max"]

    num_bits_T = int(np.ceil(np.log2(t_max)))

    population = []

    for _ in range(POP_SIZE):
        validators = np.random.randint(0, 2, size=N)

        T = np.random.randint(t_min, t_max + 1)
        T_bits = list(map(int, format(T, f'0{num_bits_T}b')))

        chromosome = np.concatenate([validators, T_bits])
        population.append(chromosome)

    return np.array(population)


# ==========================================================
# DECODE CHROMOSOME
# ==========================================================
def decode_chromosome(chromosome, params):
    N = params["N"]
    t_min = params["t_min"]
    t_max = params["t_max"]

    num_bits_T = int(np.ceil(np.log2(t_max)))

    validators = chromosome[:N]
    T_bits = chromosome[N:]

    T = 0
    for bit in T_bits:
        T = (T << 1) | bit

    T = t_min + ((t_max - t_min) / (2 ** num_bits_T -1)) * T

    selected = np.where(validators == 1)[0]

    return selected, T


# ==========================================================
# FITNESS FUNCTION (Minimization with Penalty)
# ==========================================================
def fitness(chromosome, params):
    selected, T = decode_chromosome(chromosome, params)

    phi = params["phi"]
    f = len(selected)

    if f == 0:
        return -1e9  # invalid

    # COST
    cost = sum(phi[i] for i in selected) * T

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
    cost_n = (cost - params["cost_min"]) / (params["cost_max"] - params["cost_min"] + 1e-9)
    lat_n = (latency - params["lat_min"]) / (params["lat_max"] - params["lat_min"] + 1e-9)

    sec_raw = (security - params["sec_min"]) / (params["sec_max"] - params["sec_min"] + 1e-9)
    sec_n = 1 - sec_raw

    obj = params["w1"] * lat_n + params["w2"] * sec_n + params["w3"] * cost_n

    return obj


# ==========================================================
# SELECT A PARENT 
# ==========================================================
def tournament_selection(population, fitness_values, k=3, minimize=True):
    pop_size = len(population)                     # use current size
    k = min(k, pop_size)                            # ensure k ≤ pop_size
    if pop_size == 0:
        raise ValueError("Population is empty – cannot select parents.")
    competitors = np.random.choice(pop_size, k, replace=False)

    best_index = competitors[0]
    for idx in competitors[1:]:
        if minimize:
            if fitness_values[idx] < fitness_values[best_index]:
                best_index = idx
        else:
            if fitness_values[idx] > fitness_values[best_index]:
                best_index = idx

    return population[best_index].copy()


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
# BINARY-CODED GENETIC ALGORITHM
# ==========================================================
def binary_coded_genetic_algorithm(params):
    # Generate Initial Population
    population = generate_initial_population(params)

    # Evaluate fitness values
    fitness_values = [fitness(chromosome, params) for chromosome in population]

    # Store best chromosome and its fitness value
    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx]
    best_fitness = fitness_values[best_idx]
    
    # Best Fitness per generation
    best_fitness_per_gen = []


    for gen in range(GENERATIONS):
        offspring_population = []

        # CROSSOVER
        for i in range(POP_SIZE // 2):
            p1 = tournament_selection(population, fitness_values)
            p2 = tournament_selection(population, fitness_values)

            c1, c2 = crossover(p1, p2)

            # offsprings are added
            offspring_population.append(c1)
            offspring_population.append(c2)
        

        # MUTATION
        for i in range(len(offspring_population)):
            offspring_population[i] = mutate(offspring_population[i])
        
    
        # Evaluate offspring fitness 
        offspring_fitness = [fitness(ind, params) for ind in offspring_population]

        # Combine
        combined_population = list(population) + offspring_population
        combined_fitness = list(fitness_values) + offspring_fitness

        # Sort
        sorted_indices = np.argsort(combined_fitness)

        # Select next generation
        population = [combined_population[i] for i in sorted_indices[:POP_SIZE]]
        fitness_values = [combined_fitness[i] for i in sorted_indices[:POP_SIZE]]

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
        print(f"\nInstance {idx}:")

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
        cost_max = np.sum(phi) * params["t_max"]
        cost_min = np.sum(phi_sorted[:params["f_min"]]) * params["t_min"]

        
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
        all_best_costs = []
        all_times = []

        # Run GA multiple times
        for run in range(num_runs):
            start_time = time.perf_counter()

            best_assignment, best_cost, fitness_per_gen = binary_coded_genetic_algorithm(params)

            end_time = time.perf_counter()

            run_time = end_time - start_time

            all_histories.append(fitness_per_gen)
            all_best_sol.append(best_assignment)
            all_best_costs.append(best_cost)
            all_times.append(run_time)


            print(f"  Run {run+1}: Best Utility = {best_cost}, Time = {run_time:.4f} sec")

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

        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title(f"CONVERGENCE PLOT || BCGA || {os.path.splitext(os.path.basename(filename))[0]} || Instance {idx}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{os.path.splitext(os.path.basename(filename))[0]}_instance_{idx}_BCGA_penalty_convergence.png", dpi=300)
        plt.show()

        # Store results
        results.append({
            "histories": all_histories,
            "best_cost_per_run": all_best_costs,
            "best_solution_per_run": all_best_sol,
            "time_per_run": all_times,
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
        file_path = os.path.join( dataset_dir,file)
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
    # "bco2.txt", 
]


# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__": 
    all_results = solve_multiple_files(files)

    for file, instances in all_results.items():
        print(f"\n===== SUMMARY FOR FILE: {file} =====")

        for idx, instance in enumerate(instances, start=1):

            utilities = np.array(instance["best_cost_per_run"])
            times = np.array(instance["time_per_run"])
            solutions = instance["best_solution_per_run"]
            params = instance["params"]
            print(f"\n--- Instance {idx} ---")

            # Compute stats
            avg_utility = np.mean(utilities)
            std_utility = np.std(utilities)
            best_utility_index = np.argmin(utilities)
            best_solution = solutions[best_utility_index]
            best_utility = utilities[best_utility_index]
            worst_utility = np.max(utilities)
            validators, T = decode_chromosome(best_solution,params)
            avg_time = np.mean(times)
            total_time = np.sum(times)

            # Print
            print(f"Validators         : {validators},Transactions:{T}")
            print(f"Average utility    : {avg_utility:.2f} ± {std_utility:.2f}")
            print(f"Best utility       : {best_utility:.2f}")
            print(f"Worst utility      : {worst_utility:.2f}")
            print(f"Average time       : {avg_time:.4f} s")
            print(f"Total time         : {total_time:.4f} s")