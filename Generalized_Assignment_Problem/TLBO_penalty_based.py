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
TEACHING_FACTOR = 2


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population(pop_size, m, n):
    # Each student: length n (jobs), values in [0, m-1]
    return np.random.randint(0, m, size=(pop_size, n))


# ==========================================================
# FITNESS FUNCTION (Maximization with Penalty)
# ==========================================================
def fitness(student, C, R, B, penalty_weight=1000):
    cost = 0
    penalty = 0

    m = len(B)
    resource_used = [0] * m
    
    agents = student.astype(int)

    for j, agent in enumerate(agents):
        cost += C[agent][j]
        resource_used[agent] += R[agent][j]

    for a in range(m):
        if resource_used[a] > B[a]:
            penalty += (resource_used[a] - B[a])

    return cost - penalty_weight * penalty


# ==========================================================
# TEACHING LEARNING BASED OPTIMIZATION
# ==========================================================
def teaching_learning_based_optimization(C, R, B):
    m = len(C)      # number of Agents
    n = len(C[0])   # number of Jobs

    # Initialize random population
    population = generate_initial_population(POP_SIZE, m, n)

    # Evaluate fitness of the population
    fitness_values = np.array([
        fitness(population[i], C, R, B)
        for i in range(POP_SIZE)
    ])

    # Global best initialization
    best_index = np.argmin(fitness_values)
    best_solution = population[best_index].copy()
    best_fitness = fitness_values[best_index]

    # Best Fitness per iteration
    best_fitness_per_gen = []

    for gen in range(ITERATIONS):

        for i in range(POP_SIZE):
            ##################################
            #        TEACHING PHASE          #
            ##################################

            # Generate random number array
            r1 = np.random.rand()
            r2 = np.random.rand()

            # Find X_best
            x_best_index = np.argmax(fitness_values)
            x_best = population[x_best_index].copy()

            # Determine X_mean
            x_mean = np.mean(population, axis=0)

            # Calculate x_new
            x_new = population[i] + r1 * (x_best - TEACHING_FACTOR * x_mean)

            # Bound x_new
            x_new = np.clip(np.round(x_new),0,m - 1)

            # Calculate fitness of x_new
            f_x_new = fitness(x_new,C,R,B)

            # Compare with the past fitness
            if f_x_new > fitness_values[i]:
                population[i] = x_new.copy()
                fitness_values[i] = f_x_new
            

            ##################################
            #        LEARNER PHASE           #
            ##################################

            # Select a random partner solution other than current solution and its fitness value
            x_p_index = np.random.choice(
                np.delete(np.arange(population.shape[0]), i)
            )
            x_p = population[x_p_index]
            f_x_p = fitness_values[x_p_index]

            # Calculate x_new
            if f_x_p > fitness_values[i]:
                x_new = population[i] + r2 * (population[i] - x_p)
            else:
                x_new = population[i] - r2 * (population[i] - x_p)

            # Bound x_new
            x_new = np.clip(np.round(x_new),0,m - 1)

            # Calculate fitness of x_new
            f_x_new = fitness(x_new,C,R,B)

            # Compare with the past fitness
            if f_x_new > fitness_values[i]:
                population[i] = x_new.copy()
                fitness_values[i] = f_x_new

        #  Iteration best
        gen_best_index = np.argmax(fitness_values)
        gen_best_solution = population[gen_best_index]
        gen_best_fitness = fitness_values[gen_best_index]

        best_fitness_per_gen.append(gen_best_fitness)
        
        # print(f"Generation {gen+1}: Best Fitness = {f_g_best}")

        # Global best update
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = gen_best_solution.copy()


    return best_solution, best_fitness, best_fitness_per_gen


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

            best_assignment, best_cost, fitness_per_gen = teaching_learning_based_optimization(C, R, B)

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
        plt.title(f"CONVERGENCE PLOT || TLBO(penalty based) || {os.path.splitext(os.path.basename(filename))[0]} || Instance {idx}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{os.path.splitext(os.path.basename(filename))[0]}_instance_{idx}_TLBO_penalty_convergence.png", dpi=300)
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
