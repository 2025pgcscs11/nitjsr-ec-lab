
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
POP_SIZE = 500
ITERATIONS = 200
SCALING_FACTOR = 0.85
CROSSOVER_RATE = 0.8


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population(pop_size, m, n):
    # Each chromosome: length n (jobs), values in [0, m-1]
    return np.random.randint(0, m-1, size=(pop_size, n))


# ==========================================================
# GENERATE FEASIBLE SOLUTION
# ==========================================================
def generate_feasible_solution(C, R, B):
    n = len(C[0])
    m = len(C)

    assignment = np.full(n, -1)
    resource_used = [0] * m

    # Assign jobs one by one
    for j in range(n):
        best_agent = None
        best_profit = -float('inf')

        for a in range(m):
            if resource_used[a] + R[a][j] <= B[a]:
                if C[a][j] > best_profit:
                    best_profit = C[a][j]
                    best_agent = a

        if best_agent is not None:
            assignment[j] = best_agent
            resource_used[best_agent] += R[best_agent][j]
        else:
            # fallback: assign randomly (rare case)
            assignment[j] = np.random.randint(m)

    return assignment


# ==========================================================
# FITNESS FUNCTION (Maximization with Penalty)
# ==========================================================
def fitness(vector, C, R, B):
    cost = 0

    m = len(B)
    resource_used = [0] * m
    
    agents = vector.astype(int)

    for j, agent in enumerate(agents):
        cost += C[agent][j]
        resource_used[agent] += R[agent][j]


    return cost


# ==========================================================
# CHECK FEASIBILITY OF EACH trial
# ==========================================================
def is_feasible(trial, R, B):
    m = len(R)        # number of agents
    n = len(R[0])     # number of jobs

    agents = trial.astype(int)

    resource_used = np.zeros(m)

    # Compute resource usage
    for j, agent in enumerate(agents):
        resource_used[agent] += R[agent][j]

        # Early stopping (optimization)
        if resource_used[agent] > B[agent]:
            return False

    return True


# ==========================================================
# DIFFERENTIAL EVOLUTION BASED OPTIMIZATION
# ==========================================================
def differential_evolution_based_optimization(C, R, B):

    m = len(C)      # number of Agents
    n = len(C[0])   # number of Jobs

    # Initialize random target vector
    target_vector = generate_initial_population(POP_SIZE, m, n)
    donar_vector = np.zeros((POP_SIZE, n))
    trial_vector = np.zeros((POP_SIZE, n))

    # Evaluate fitness of the target vector
    fitness_values = np.array([
        fitness(target_vector[i], C, R, B)
        for i in range(POP_SIZE)
    ])

    # Initialize global best
    best_index = np.argmax(fitness_values)
    g_best = target_vector[best_index].copy()
    f_g_best = fitness_values[best_index]

    # Store iteration best
    best_fitness_per_gen = []

    for t in range(ITERATIONS):

        for i in range(POP_SIZE):
            # Generate random number array
            r1, r2, r3 = np.random.choice(POP_SIZE, 3, replace=False)

            # Generate Donar Vector (mutation)
            donar_vector[i] = target_vector[r1] + SCALING_FACTOR * (target_vector[r2] - target_vector[r3])

            # Generate Trial Vector (crossover)
            del_ = np.random.randint(n)
            r = np.random.rand()

            for j in range(n):
                if np.random.rand() <= CROSSOVER_RATE or j == del_:
                    trial_vector[i][j] = donar_vector[i][j]
                else:
                    trial_vector[i][j] = target_vector[i][j]

        for i in range(POP_SIZE):
            # Bound
            trial_vector[i] = np.clip(trial_vector[i], 0, m - 1)

            # Discretize
            trial_vector[i] = np.round(trial_vector[i])

            # Replace infeasible solution
            if not is_feasible(trial_vector[i],R,B):
                trial_vector[i] = generate_feasible_solution(trial_vector[i],C,R,B)

            # Selection
            temp = fitness(trial_vector[i], C, R, B)
            if temp > fitness_values[i]:
                target_vector[i] = trial_vector[i]
                fitness_values[i] = temp 

        # Iteration best
        iteration_best = np.max(fitness_values)
        best_fitness_per_gen.append(iteration_best)

        # Global best update
        best_index = np.argmax(fitness_values)
        if fitness_values[best_index] > f_g_best:
            g_best = target_vector[best_index].copy()
            f_g_best = fitness_values[best_index]

        # print(f"Iteration {t+1}: Iteration Best = {iteration_best}, Global Best = {f_g_best}")

    return g_best, f_g_best, best_fitness_per_gen


# ================================================================
# GENERATE COST MATRIX, RESOURCE MATRIX, CAPACITY VECTOR FROM FILE
# ================================================================
def read_gap_file(filename):

    instances = []

    with open(filename, 'r') as f:
        data = list(map(int, f.read().split()))

    idx = 0
    P = data[idx]
    idx += 1

    for _ in range(P):

        m = data[idx]
        n = data[idx + 1]
        idx += 2

        C = []
        for _ in range(m):
            C.append(data[idx:idx+n])
            idx += n

        R = []
        for _ in range(m):
            R.append(data[idx:idx+n])
            idx += n

        B = data[idx:idx+m]
        idx += m

        instances.append((C, R, B))

    return instances


# ==================================================================
# ITERATE OVER ALL INSTANCES IN A FILE AND APPLY GENETIC ALGORITHM
# ==================================================================
def solve_gap_file(filename):
    instances = read_gap_file(filename)
    results = []

    print(f"\n===== Solving file: {filename} =====\n")

    for idx, (C, R, B) in enumerate(instances, start=1):
        print(f"\nInstance {idx}:")

        num_runs = 20
        all_histories = []
        all_best_sol = []
        all_best_costs = []
        all_times = []

        # Run GA multiple times
        for run in range(num_runs):
            start_time = time.perf_counter()

            best_assignment, best_cost, fitness_per_gen = differential_evolution_based_optimization(C, R, B)

            end_time = time.perf_counter()

            run_time = end_time - start_time

            all_histories.append(fitness_per_gen)
            all_best_sol.append(best_assignment)
            all_best_costs.append(best_cost)
            all_times.append(run_time)

            print(f"  Run {run+1}: Best Cost = {best_cost}, Time = {run_time:.4f} sec")

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
        plt.title(f"CONVERGENCE PLOT || DE(replacement based) || {os.path.splitext(os.path.basename(filename))[0]} || Instance {idx}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{os.path.splitext(os.path.basename(filename))[0]}_instance_{idx}_DE_replacement_convergence.png", dpi=300)
        plt.show()

        # Store results
        results.append({
            "histories": all_histories,
            "best_costs": all_best_costs,
            "avg_fitness": avg_fitness
        })

    return results


# ==========================================================
# ITERATE OVER ALL FILES
# ==========================================================
def solve_multiple_files(file_list,base_dir="gap_dataset"):
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
    "gap_sample_data_txt.txt",
    # "gap1.txt",
    # "gap2.txt",
    # "gap3.txt",
    # "gap4.txt",
    # "gap5.txt",
    # "gap6.txt",
    # "gap7.txt",
    # "gap8.txt",
    # "gap9.txt",
    # "gap10.txt",
    # "gap11.txt",
    "gap12.txt",
]


# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__": 
    solve_multiple_files(files)
