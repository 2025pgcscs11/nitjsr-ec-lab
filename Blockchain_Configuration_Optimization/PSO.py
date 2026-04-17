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
ITERATIONS = 200
INTERTIA = 0.7
C1 = 1.5
C2 = 1.5


# ==========================================================
# INITIAL POPULATION (REALENCODED)
# ==========================================================
def generate_initial_population(params):
    N = params["N"]
    t_min = params["t_min"]
    t_max = params["t_max"]


    population = []

    for _ in range(POP_SIZE):
        validators = np.random.rand(N)   # [0,1]
        T = np.random.uniform(t_min, t_max)

        particle = np.concatenate([validators, [T]])
        population.append(particle)

    return np.array(population)


# ==========================================================
# INITIAL VELOCITY
# ==========================================================
def generate_initial_velocity(params):
    N = params["N"]
    t_min = params["t_min"]
    t_max = params["t_max"]

    velocities = np.zeros((POP_SIZE, N + 1))

    # Validators velocity → small range
    velocities[:, :N] = np.random.uniform(-0.1, 0.1, size=(POP_SIZE, N))

    # Transaction velocity → scaled to its range
    velocities[:, N] = np.random.uniform(
        -0.1 * (t_max - t_min),
         0.1 * (t_max - t_min),
        size=POP_SIZE
    )

    return velocities


# ==========================================================
# DECODE particle
# ==========================================================
def decode_particle(particle, params):
    N = params["N"]
    t_min = params["t_min"]
    t_max = params["t_max"]

    v = particle[:params["N"]]
    T = int(round(particle[-1]))

    # enforce bounds
    T = np.clip(T, t_min, t_max)

    selected = np.where(v >= 0.5)[0]

    return selected, T


# ==========================================================
# FITNESS FUNCTION (Minimization with Penalty)
# ==========================================================
def fitness(particle, params):
    selected, T = decode_particle(particle, params)

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
# PARTICLE SWARM OPTIMIZATION
# ==========================================================
def particle_swarm_optimization(params):
    # Generate Initial Population and velocity
    population = generate_initial_population(params)
    velocity = generate_initial_velocity(params)

    # Evaluate fitness values
    fitness_values = [fitness(particle, params) for particle in population]

    p_best = population.copy()
    f_p_best = fitness_values.copy()

    # Initialize global best 
    g_best_index = np.argmin(f_p_best)
    g_best = p_best[g_best_index].copy()
    f_g_best = f_p_best[g_best_index]

    # Best Fitness per iteration (ONLY iteration best)
    best_fitness_per_gen = []

    for gen in range(ITERATIONS):

        for i in range(POP_SIZE):

            r1 = np.random.rand()
            r2 = np.random.rand()

            # velocity update
            velocity[i] = (
                INTERTIA * velocity[i]
                + C1 * r1 * (p_best[i] - population[i])
                + C2 * r2 * (g_best - population[i])
            )

            # Position update
            population[i] += velocity[i]

            # Position bounds
            N = params["N"]
            t_min = params["t_min"]
            t_max = params["t_max"]
            
            population[i][:N] = np.clip(population[i][:N], 0.0, 1.0)
            population[i][N] = np.clip(population[i][N], t_min, t_max)

            # Velocity bounds
            velocity[i][:N] = np.clip(velocity[i][:N], -0.2, 0.2)
            velocity[i][N] = np.clip(velocity[i][N], -0.2 * (t_max - t_min), 0.2 * (t_max - t_min))

            # Fitness
            fitness_values[i] = fitness(population[i], params)

            # Personal best update
            if fitness_values[i] < f_p_best[i]:
                p_best[i] =population[i].copy()
                f_p_best[i] = fitness_values[i]

        # Iteration best
        gen_best = np.min(f_p_best)
        best_fitness_per_gen.append(gen_best)

        # val, T =decode_chromosome(population[0],params)
        # print(f"Iteration {gen+1}:  Best Fitness = {gen_best_fitness}  Number of Validators = {len(val)}   Number of Transactions = {T}")
        
        # Global best update 
        best_index = np.argmin(f_p_best)
        if f_p_best[best_index] < f_g_best:
            g_best = p_best[best_index].copy()
            f_g_best = f_p_best[best_index]


    return g_best, f_g_best, best_fitness_per_gen


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

            best_assignment, best_utility, fitness_per_gen = particle_swarm_optimization(params)

            end_time = time.perf_counter()

            run_time = end_time - start_time

            all_histories.append(fitness_per_gen)
            all_best_sol.append(best_assignment)
            all_best_utilitys.append(best_utility)
            all_times.append(run_time)

            val, T =decode_particle(best_assignment,params)
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

        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title(f"CONVERGENCE PLOT || PSO || {os.path.splitext(os.path.basename(filename))[0]} || Setting {idx}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{os.path.splitext(os.path.basename(filename))[0]}_setting_{idx}_PSO_convergence.png", dpi=300)
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
            number_of_validators, T = decode_particle(best_solution,params)
            avg_time = np.mean(times)
            total_time = np.sum(times)

            # Print
            print(f"Number of Validators   : {len(number_of_validators)}")
            print(f"Number of Transactions : {T}")
            print(f"Average utility        : {avg_utility:.2f} ± {std_utility:.2f}")
            print(f"Best utility           : {best_utility:.2f}")
            print(f"Worst utility          : {worst_utility:.2f}")
            print(f"Average time           : {avg_time:.4f} s")
            print(f"Total time             : {total_time:.4f} s")