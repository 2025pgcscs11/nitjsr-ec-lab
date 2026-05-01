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
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# ==========================================================
# INITIAL POPULATION (BINARY ENCODED)
# ==========================================================
def generate_initial_population(m, n):
    # Number of bits needed to represent values up to m-1
    num_bits = int(np.ceil(np.log2(m))) if m > 1 else 1

    # Generate random integers in [0, 1]
    population = np.random.randint(0, 2, size=(POP_SIZE, num_bits * n))
   
    return np.array(population)


# ==========================================================
# DECODE CHROMOSOME AND EXTRACT AGENTS
# ==========================================================
def decode_chromosome(chromosome, m):
    num_bits = int(np.ceil(np.log2(m))) if m > 1 else 1
    agents = []

    for j in range(len(chromosome) // num_bits):
        start = j * num_bits
        end = start + num_bits

        value = 0
        for bit in chromosome[start:end]:
            value = (value << 1) | bit

        # Keep within bounds
        if value >= m:
            value = np.random.randint(0, m)

        agents.append(value)

    return agents


# ==========================================================
# ENCODE CHROMOSOME INTO BINARY
# ==========================================================
def encode_chromosome(assignment, m):
    """
    Converts a list of agent assignments into a binary chromosome.
    Each agent index is encoded MSB-first to match decode_chromosome.
    """
    num_bits = int(np.ceil(np.log2(m))) if m > 1 else 1
    chromosome = []

    for agent in assignment:
        for i in reversed(range(num_bits)):
            chromosome.append((agent >> i) & 1)

    return np.array(chromosome, dtype=np.int8)


# ==========================================================
# REPAIR CHROMOSOME FOR MAXIMIZATION GAP
# ==========================================================
def repair_chromosome_greedy_cost(chromosome, C, R, B):
    """
    Repairs an infeasible chromosome for the MAXIMIZATION GAP problem.

    Strategy:
    - For each overloaded agent a, move jobs away to restore feasibility.
    - Job selection priority: lowest profit-to-resource ratio at agent a
      (cheapest jobs to keep → move them first to preserve high-value ones).
    - Move destination: agent b that causes least profit loss (max gain).
    - Repeats until fully feasible or no further progress is possible.
    """
    n = len(C[0])   # number of jobs
    m = len(C)      # number of agents

    C = np.array(C)     # ensure numpy for fast indexing
    R = np.array(R)
    B = np.array(B)

    agents = decode_chromosome(chromosome, m)
    agents = np.array(agents)

    # Compute initial resource usage per agent
    resource_used = np.zeros(m)
    for j in range(n):
        resource_used[agents[j]] += R[agents[j], j]

    # Early exit if already feasible
    if np.all(resource_used <= B):
        return chromosome.copy()

    max_iterations = n * m      # safety cap
    iteration = 0
    changed = True

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1

        for a in range(m):

            while resource_used[a] > B[a]:

                # Jobs currently assigned to overloaded agent a
                jobs_at_a = np.where(agents == a)[0]

                if len(jobs_at_a) == 0:
                    break

                # Priority: move cheapest jobs first
                # Sort by profit-to-resource ratio ascending
                # (lowest ratio = least valuable per unit resource = move first)
                ratios = np.array([
                    C[a, j] / R[a, j] if R[a, j] > 0 else float('inf')
                    for j in jobs_at_a
                ])
                priority_order = jobs_at_a[np.argsort(ratios)]

                best_move = None
                best_gain = -float('inf')   # maximize: least profit loss

                for j in priority_order:
                    for b in range(m):
                        if b == a:
                            continue

                        # Check if move is feasible for agent b
                        if resource_used[b] + R[b, j] <= B[b]:

                            # Maximization: prefer move with highest gain
                            # (or least loss if all negative)
                            gain = C[b, j] - C[a, j]

                            if gain > best_gain:
                                best_gain = gain
                                best_move = (j, b)

                    # Early exit: found a profitable move, no need to
                    # check remaining jobs in priority order
                    if best_move is not None and best_gain >= 0:
                        break

                if best_move is not None:
                    j, new_agent = best_move

                    # Apply move
                    resource_used[a]         -= R[a, j]
                    resource_used[new_agent] += R[new_agent, j]
                    agents[j]                 = new_agent

                    changed = True

                else:
                    # No feasible move exists for agent a
                    # (chromosome is irreparable for this agent)
                    break

    return encode_chromosome(agents.tolist(), m)


# ==========================================================
# FITNESS FUNCTION
# ==========================================================
def fitness(chromosome, C, R):
    cost = 0
    m = len(C)      # number of Agents

    agents = decode_chromosome(chromosome, m)

    for j, agent in enumerate(agents):
        cost += C[agent][j]

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
# BINARY TOURNAMNET SELECTION 
# ==========================================================
def binary_tournament_selection(population, fitness_values, k=2, problem="max"):
    fitness_values = np.array(fitness_values)  
    Np = len(population)
    
    if Np == 0:
        raise ValueError("Population is empty.")
    
    mating_pool = []
    selection_count = np.zeros(Np, dtype=int)

    while len(mating_pool) < Np:
        available = np.where(selection_count < 2)[0]

        # If not enough available, fall back to full population
        if len(available) < k:
            available = np.arange(Np)

        participants = np.random.choice(available, k, replace=False)

        if problem == "min":
            winner = participants[np.argmin(fitness_values[participants])]
        else:
            winner = participants[np.argmax(fitness_values[participants])]

        mating_pool.append(population[winner].copy())
        selection_count[winner] += 1

    return np.array(mating_pool)


# ==========================================================
# CROSSOVER ON TWO PARENTS (RANDOM BIT POINTS)
# ==========================================================
def crossover(p1, p2):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, len(p1) - 1)
        return (
        np.concatenate((p1[:point], p2[point:])),
        np.concatenate((p2[:point], p1[point:]))
        )
    return p1.copy(), p2.copy()


# ==========================================================
# CROSSOVER ON TWO PARENTS (RANDOM GENE POINTS)
# ==========================================================
# def crossover(p1, p2, m):
#     num_bits = int(np.ceil(np.log2(m))) if m > 1 else 1

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
#      num_bits = int(np.ceil(np.log2(m))) if m > 1 else 1

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

    # Evaluate fitness values
    fitness_values = [fitness(chromosome, C, R) for chromosome in population]

    # Store best chromosome
    best_idx = np.argmax(fitness_values)
    best_solution  = population[best_idx]
    best_fitness = fitness_values[best_idx]

    # Best Fitness so far per gen
    best_fitness_so_far_per_gen = []

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

        # Repair infeasible solution
        offspring_population = [
            repair_chromosome_greedy_cost(c, C, R, B)
            if not is_feasible(c, R, B)
            else c
            for c in offspring_population
        ]

        # Evaluate offspring fitness 
        offspring_fitness = [fitness(ind, C, R) for ind in offspring_population]

        # Combine + elitist selection
        combined = list(zip(
            list(population) + offspring_population,
            list(fitness_values) + offspring_fitness
        ))
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:POP_SIZE]

        # Paired
        population, fitness_values = zip(*combined)
        population = list(population)
        fitness_values = list(fitness_values)

        # Update best
        gen_best = fitness_values[0]
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_solution = population[0].copy()

        best_fitness_so_far_per_gen.append(best_fitness)

        # print(f"Generation {gen+1}: Best Fitness So Far = {best_fitness}")

    return best_solution, best_fitness, best_fitness_so_far_per_gen


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

        plt.xlabel("GENERATION")
        plt.ylabel("BEST COST")
        plt.title(f"CONVERGENCE GRAPH || BINARY CODED GENETIC ALGORITHM || REAPAIR BASED (GREEDY) || {os.path.splitext(os.path.basename(filename))[0]} || Instance {idx}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # os.makedirs("plots", exist_ok=True)
        # plt.savefig(f"plots/{os.path.splitext(os.path.basename(filename))[0]}_instance_{idx}_BCGA_repair_cost_convergence.png", dpi=300)
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
    "gap8.txt",
    # "gap9.txt",
    "gap10.txt",
    # "gap11.txt",    
    # "gap12.txt"
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