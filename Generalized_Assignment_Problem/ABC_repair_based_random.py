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
ITERATIONS = 100
LIMIT = 10


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population(pop_size, m, n):
    # Each food_source: length n (location), values in [0, m-1]
    return np.random.randint(0, m, size=(pop_size, n))


# ==========================================================
# REPAIR FOOD SOURCE USING RANDOM APPROACH
# ==========================================================
def repair_food_source_random(food_source,C,R,B):
    n = len(C[0])   # number of Jobs
    m = len(C)      # number of Agents

    agents = food_source.astype(int)

    resource_used = np.zeros(m)

    for j, agent in enumerate(agents):
        resource_used[agent] += R[agent][j]

    for a in range(m):

        while resource_used[a] > B[a]:

            jobs = [j for j in range(n) if agents[j] == a]

            if not jobs:
                break

            j = np.random.choice(jobs)

            feasible_agents = []

            for b in range(m):

                if b != a and resource_used[b] + R[b][j] <= B[b]:
                    feasible_agents.append(b)

            if feasible_agents:

                new_agent = np.random.choice(feasible_agents)

                resource_used[a] -= R[a][j]
                resource_used[new_agent] += R[new_agent][j]

                agents[j] = new_agent

            else:
                break

    return agents



# ==========================================================
# OBJECTIVE FUNCTION (Maximization with Penalty)
# ==========================================================
def fitness(food_source, C, R):
    cost = 0

    m = len(C)
    resource_used = [0] * m
    
    agents = food_source.astype(int)

    for j, agent in enumerate(agents):
        cost += C[agent][j]
        resource_used[agent] += R[agent][j]


    return cost


# ==========================================================
# CHECK FEASIBILITY OF EACH FOOD SOURCE
# ==========================================================
def is_feasible(food_source, R, B):
    m = len(R)        # number of agents
    n = len(R[0])     # number of jobs

    agents = food_source.astype(int)

    resource_used = np.zeros(m)

    # Compute resource usage
    for j, agent in enumerate(agents):
        resource_used[agent] += R[agent][j]

        # Early stopping (optimization)
        if resource_used[agent] > B[agent]:
            return False

    return True


# ==========================================================
# ARTIFICIAL BEE COLONY OPTIMIZATION (ABC)
# ==========================================================
def artificial_bee_colony_opimization(C, R, B):
    m = len(C)      # number of Agents
    n = len(C[0])   # number of Jobs

    # Initialize random population
    population = generate_initial_population(POP_SIZE, m, n)

    # Evaluate fitness of the population
    fitness_values = np.array([
        fitness(population[i],C,R)
        for i in range(POP_SIZE)
    ])

    # Initial food_source vector of the popultion
    trial_vector = np.zeros(POP_SIZE)

    # Global best initialization
    best_index = np.argmax(fitness_values)
    best_solution = population[best_index].copy()
    best_fitness = fitness_values[best_index].copy()

    # Best Objective Function Value per iteration
    best_fitness_value_per_gen = []

    for gen in range(ITERATIONS):

        ##################################
        #     Employed Bee Phase         #
        ##################################
        for i in range(POP_SIZE):
            # Select a random variable from 0 to n
            j = np.random.randint(0,n)

            # Select a random partner solution other than current solution
            k = i
            while k == i:
                k = np.random.randint(0, POP_SIZE)

            # Select a random number between (-1,1)
            phi = np.random.uniform(-1,1)

            # Modify and Bound the jth variable
            x_new = population[i].copy()
            x_new[j] = x_new[j] + phi * (x_new[j] - population[k][j])
            x_new  = np.clip(np.round(x_new), 0, m - 1)

            # Repair infeasible solution
            if not is_feasible(x_new,R,B):
                x_new = repair_food_source_random(x_new,C,R,B)

            # Evaluate the objective function and fitness of newly generated solution
            f_new = fitness(x_new,C,R)

            # Greedy selection and food_source vector updation
            if f_new > fitness_values[i]:
                population[i] = x_new
                fitness_values[i] = f_new
                trial_vector[i] = 0
            else:
                trial_vector[i] += 1


        ####################################
        #       PROBABILITY CALCULATION    #
        #################################### 
        prob = 0.9 * (fitness_values / np.max(fitness_values)) + 0.1
        

        ##################################
        #     Onlooker Bee Phase         #
        ##################################
        p = 0
        q = 1

        while p < POP_SIZE:
            # Generate random number (0,1)
            r = np.random.rand()

            if r < prob[q]:
                # Select a random varible from 0 to n
                j = np.random.randint(0,n)
                
                # Select a random partner solution other than current solution
                k = q
                while k == q:
                    k = np.random.randint(0, POP_SIZE)

                # Select a random number between (-1,1)
                phi = np.random.uniform(-1,1)

                # Modify and Bound the jth variable
                x_new = population[q].copy()
                x_new[j] = x_new[j] + phi * (x_new[j] - population[k][j])
                x_new  = np.clip(np.round(x_new), 0, m - 1)

                # Repair infeasible solution
                if not is_feasible(x_new,R,B):
                    x_new = repair_food_source_random(x_new,C,R,B)

                # Evaluate the objective function and fitness of newly generated solution
                f_new = fitness(x_new,C,R)


                # Greedy selection and food_source vector updation
                if f_new > fitness_values[q]:
                    population[q] = x_new
                    fitness_values[q] = f_new
                    trial_vector[q] = 0
                else:
                    trial_vector[q] = trial_vector[q] + 1

                # Increment p
                p = p + 1
                
            # Increment q
            q = (q + 1) % POP_SIZE


        ##################################
        #   Memorize the Best solution   #
        ##################################
        current_best_index = np.argmax(fitness_values)
        current_best_fitness = fitness_values[current_best_index]

        # Update only if better
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_index].copy()

        # Store global best (not current best)
        best_fitness_value_per_gen.append(best_fitness)
        # print(f"Generation {gen+1}: Best Objective Function Value = {best_fitness}")
        

        ##################################
        #     Scout Bee Phase            #
        ##################################
        for i in range(POP_SIZE):
            if trial_vector[i] > LIMIT: 
                # Generate a random solution
                x_new = np.random.randint(0, m, n)
                
                # Repair infeasible solution
                if not is_feasible(x_new, R, B):
                    x_new = repair_food_source_random(x_new, C, R, B)

                # Evaluate the objective function and fitness of newly generated solution and assign it
                f_new = fitness(x_new,C,R)


                population[i] = x_new
                fitness_values[i] = f_new

                # Reset food_source vector
                trial_vector[i] = 0


    return best_solution, best_fitness, best_fitness_value_per_gen


# ================================================================
# GENERATE COST MATRIX, RESOURCE MATRIX, CAPACITY VECTOR FROM FILE
# ================================================================
def read_gap_file(filename):

    instances = []

    with open(filename, 'r') as f:
        data = list(map(int, f.read().split()))

    idx = 0
    # P = data[idx]
    P = 1               # overriding actual instance value to 1
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

            best_assignment, best_cost, fitness_per_gen = artificial_bee_colony_opimization(C, R, B)

            end_time = time.perf_counter()

            run_time = end_time - start_time

            feasible = is_feasible(best_assignment,R,B)

            all_histories.append(fitness_per_gen)
            all_best_sol.append(best_assignment)
            all_best_costs.append(best_cost)
            all_times.append(run_time)

            print(f"  Run {run+1}: Best Cost = {best_cost}, Time = {run_time:.4f} sec, Feasible = {feasible}")

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
        plt.title(f"CONVERGENCE PLOT || ABC(penalty based) || {os.path.splitext(os.path.basename(filename))[0]} || Instance {idx}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{os.path.splitext(os.path.basename(filename))[0]}_instance_{idx}_ABC_penalty_convergence.png", dpi=300)
        plt.show()

        # Store results
        results.append({
            "histories": all_histories,
            "best_cost_per_run": all_best_costs,
            "best_solution_per_run": all_best_sol,
            "time_per_run": all_times,
            "R": R,
            "B": B
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
    # "gap_sample_data_txt.txt",
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
    all_results = solve_multiple_files(files)

    for file, instances in all_results.items():
        print(f"\n===== SUMMARY FOR FILE: {file} =====")

        for idx, instance in enumerate(instances, start=1):

            profits = np.array(instance["best_cost_per_run"])
            times = np.array(instance["time_per_run"])
            solutions = instance["best_solution_per_run"]
            R = instance["R"]
            B = instance["B"]

            # Get indices of feasible solutions
            feasible_indices = [
                i for i, sol in enumerate(solutions)
                if is_feasible(sol, R, B)
            ]

            feasible_count = len(feasible_indices)

            print(f"\n--- Instance {idx} ---")

            if feasible_count == 0:
                print("No feasible solutions found.")
                continue

            # Filter only feasible runs
            feasible_profits = profits[feasible_indices]
            feasible_times = times[feasible_indices]

            # Compute stats
            avg_profit = np.mean(feasible_profits)
            std_profit = np.std(feasible_profits)
            best_profit = np.max(feasible_profits)
            worst_profit = np.min(feasible_profits)

            avg_time = np.mean(feasible_times)
            total_time = np.sum(feasible_times)

            # Print
            print(f"Feasible runs     : {feasible_count}/{len(profits)}")
            print(f"Average profit    : {avg_profit:.2f} ± {std_profit:.2f}")
            print(f"Best profit       : {best_profit:.2f}")
            print(f"Worst profit      : {worst_profit:.2f}")
            print(f"Average time      : {avg_time:.4f} s")
            print(f"Total time        : {total_time:.4f} s")
