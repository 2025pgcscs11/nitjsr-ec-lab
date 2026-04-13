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
def generate_initial_population(m, n):
    # Number of bits needed to represent values up to m-1
    num_bits = int(np.ceil(np.log2(m)))

    # Generate random integers in [0, 1]
    population = np.random.randint(0, 2, size=(POP_SIZE, num_bits * n))
   
    return np.array(population)


# ==========================================================
# DECODE CHROMOSOME AND EXTRACT AGENTS
# ==========================================================
def decode_chromosome(chromosome, m):
    num_bits = int(np.ceil(np.log2(m)))
    agents = []

    for j in range(len(chromosome) // num_bits):
        start = j * num_bits
        end = start + num_bits

        value = 0
        for bit in chromosome[start:end]:
            value = (value << 1) | bit

        # Keep within bounds
        value = value % m
        # value = min(value, m - 1)
        agents.append(value)

    return agents


# ==========================================================
# ENCODE CHROMOSOME INTO BINARY
# ==========================================================
def encode_chromosome(assignment, m):
    num_bits = int(np.ceil(np.log2(m)))
    chromosome = []

    for agent in assignment:
        for i in reversed(range(num_bits)):
            chromosome.append((agent >> i) & 1)

    return chromosome


# ==========================================================
# REPAIR CHROMOSOME USING COST  APPROACH
# ==========================================================
def repair_chromosome_greedy_cost(chromosome, C, R, B):
    n = len(C[0])   # number of jobs
    m = len(C)      # number of agents

    agents = decode_chromosome(chromosome, m)

    resource_used = np.zeros(m)

    # Compute initial resource usage
    for j, agent in enumerate(agents):
        resource_used[agent] += R[agent][j]

    # Repair overloaded agents
    for a in range(m):

        while resource_used[a] > B[a]:

            jobs = [j for j in range(n) if agents[j] == a]

            if not jobs:
                break

            best_move = None
            best_gain = -float('inf')

            # Try all jobs assigned to agent a
            for j in jobs:

                current_profit = C[a][j]

                # Try assigning job j to other agents
                for b in range(m):

                    if b == a:
                        continue

                    # Check feasibility
                    if resource_used[b] + R[b][j] <= B[b]:

                        new_profit = C[b][j]
                        gain = new_profit - current_profit
                        # gain = new_profit / R[a][j] - current_profit / R[b][j]

                        if gain > best_gain:
                            best_gain = gain
                            best_move = (j, b)

            # Apply best move
            if best_move is not None:
                j, new_agent = best_move
                old_agent = agents[j]

                resource_used[old_agent] -= R[old_agent][j]
                resource_used[new_agent] += R[new_agent][j]

                agents[j] = new_agent
            else:
                # No feasible improvement possible
                break

    return encode_chromosome(agents, m)


# ==========================================================
# FITNESS FUNCTION (Maximization with Penalty)
# ==========================================================
def fitness(chromosome, C, R, B, penalty_weight=0):
    cost = 0

    m = len(C)      # number of Agents

    resource_used = [0] * m

    agents = decode_chromosome(chromosome, m)

    for j, agent in enumerate(agents):
        cost += C[agent][j]
        resource_used[agent] += R[agent][j]


    return cost


# ==========================================================
# CHECK FEASIBILITY OF EACH CHROMOSOME
# ==========================================================
def is_feasible(chromosome, R, B):
    m = len(R)        # number of agents
    n = len(R[0])     # number of jobs

    agents = decode_chromosome(chromosome, m)

    resource_used = np.zeros(m)

    # Compute resource usage
    for j, agent in enumerate(agents):
        resource_used[agent] += R[agent][j]

        # Early stopping (optimization)
        if resource_used[agent] > B[agent]:
            return False

    return True


# ==========================================================
# SELECT A PARENT 
# ==========================================================
def tournament_selection(population, fitness_values, k=3, minimize=False):
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
# CROSSOVER ON TWO PARENTS (RANDOM GENE POINTS)
# ==========================================================
# def crossover(p1, p2, m):
#     num_bits = int(np.ceil(np.log2(m)))
#     if random.random() < CROSSOVER_RATE:
#         num_genes = len(p1) // num_bits
#         point = random.randint(1, num_genes - 1)
#         bit_point = point * num_bits
#         return (
#             p1[:bit_point] + p2[bit_point:],
#             p2[:bit_point] + p1[bit_point:]
#         )
#     return p1[:], p2[:]


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
# MUTATION IN A CHROMOSOME (GENE-WISE)
# ==========================================================
# def mutate(chromosome, m):
#     num_bits = int(np.ceil(np.log2(m)))
#     # Decode chromosome into agent list
#     agents = decode_chromosome(chromosome, num_bits)

#     mutated_chromosome = []

#     for j in range(len(agents)):
#         if random.random() < MUTATION_RATE:
#             agents[j] = random.randint(0, m-1)

#         # Convert back to binary
#         binary_str = format(agents[j], f'0{num_bits}b')
#         mutated_chromosome.extend([int(bit) for bit in binary_str])

#     return mutated_chromosome


# ==========================================================
# BINARY-CODED GENETIC ALGORITHM
# ==========================================================
def binary_coded_genetic_algorithm(C, R, B):
    m = len(C)      # number of Agents
    n = len(C[0])   # number of Jobs

    # Generate Initial Population
    population = generate_initial_population(m,n)

    # Store best chromosome
    best_solution = None
    # Store best chromosome's fitness value
    best_fitness = float('-inf')
     # Best Fitness per generation
    best_fitness_per_gen = []

    # Evaluate fitness values
    fitness_values = [fitness(chromosome, C, R, B) for chromosome in population]


    for _ in range(GENERATIONS):
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
            temp = mutate(offspring_population[i])

            # Repair infeasible solution
            if is_feasible(temp,R,B):
                offspring_population[i] = temp
            else:
                offspring_population[i] = repair_chromosome_greedy_cost(temp,C,R,B)



        # Evaluate offspring fitness 
        offspring_fitness = [fitness(ind, C, R, B) for ind in offspring_population]

        # Combine
        combined_population = list(population) + offspring_population
        combined_fitness = list(fitness_values) + offspring_fitness

        # Sort
        sorted_indices = np.argsort(combined_fitness)[::-1]

        # Select next generation
        population = [combined_population[i] for i in sorted_indices[:POP_SIZE]]
        fitness_values = [combined_fitness[i] for i in sorted_indices[:POP_SIZE]]

        # Best of this generation
        gen_best_fitness = combined_fitness[sorted_indices[0]]
        best_fitness_per_gen.append(gen_best_fitness)

        # Update global best
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_solution = population[0]

        # print(f"Generation {gen+1}: Best Fitness = {gen_best_fitness}")

    return best_solution, best_fitness, best_fitness_per_gen

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

        # Cost matrix
        C = []
        for _ in range(m):
            C.append(data[idx:idx+n])
            idx += n

        # Resource matrix
        R = []
        for _ in range(m):
            R.append(data[idx:idx+n])
            idx += n

        # Capacities
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

            best_assignment, best_cost, fitness_per_gen = binary_coded_genetic_algorithm(C, R, B)

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

        plt.xlabel("Generation / Iteration")
        plt.ylabel("Best Cost")
        plt.title(f"CONVERGENCE GRAPH || BINARY CODED GENETIC ALGORITHM || REAPAIR BASED || {os.path.splitext(os.path.basename(filename))[0]} || Instance {idx}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{os.path.splitext(os.path.basename(filename))[0]}_instance_{idx}_BCGA_repair_cost_convergence.png", dpi=300)
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
    "gap12.txt"
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