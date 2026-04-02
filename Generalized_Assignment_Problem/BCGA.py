# ==========================================================
# IMPORTS MODULE HERE
# ==========================================================
import random
import os
import numpy as np

# ==========================================================
# CONSTANT PARAMETERS
# ==========================================================
POP_SIZE = 200
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# ==========================================================
# INITIAL POPULATION (BINARY ENCODED)
# ==========================================================
def generate_initial_population(m, n):
    # Generate random integers in [0, m-1]
    population_int = np.random.randint(0, m, size=(POP_SIZE, n))
    
    # Number of bits needed to represent values up to m-1
    num_bits = int(np.ceil(np.log2(m)))
    
    population_bin = []

    for individual in population_int:
        chromosome = []
        
        for value in individual:
            # Convert integer to fixed-length binary string
            binary_str = format(value, f'0{num_bits}b')
            
            # Append bits to chromosome
            chromosome.extend([int(bit) for bit in binary_str])
        
        population_bin.append(chromosome)

    return np.array(population_bin)

# ==========================================================
# DECODE CHROMOSOME AND EXTRACT AGENTS
# ==========================================================
def decode_chromosome(chromosome, m):
    num_bits = int(np.ceil(np.log2(m)))
    n = len(chromosome) // num_bits
    agents = []

    for j in range(len(chromosome) // num_bits):
        start = j * num_bits
        end = start + num_bits

        value = 0
        for bit in chromosome[start:end]:
            value = (value << 1) | bit

        agents.append(value)

    return agents


# ==========================================================
# FITNESS FUNCTION (Maximization with Penalty)
# ==========================================================
def fitness(chromosome, C, R, B, penalty_weight=10):
    cost = 0
    penalty = 0

    m = len(C)

    resource_used = [0] * m

    agents = decode_chromosome(chromosome, m)

    for j, agent in enumerate(agents):
        cost += C[agent][j]
        resource_used[agent] += R[agent][j]

    for a in range(m):
        if resource_used[a] > B[a]:
            penalty += (resource_used[a] - B[a])

    return cost - penalty_weight * penalty


# ==========================================================
# CHECK FEASIBILITY OF EACH CHROMOSOME
# ==========================================================
def feasibility_check(chromosome, R, B):
    m = len(B)       # number of agents

    agents = decode_chromosome(chromosome, m)
    usage = [0] * m

    # ---- Boundry Check ---- 
    for j, agent in enumerate(agents):  
        # ---- Boundry Check ----   
        if agent < 0 or agent >= m:
            return None

        # ---- Capacity check ----
        usage[agent] += R[agent][j]
        if usage[agent] > B[agent]:
            return None

    # ✔ Feasible
    return chromosome

# ==========================================================
# SELECT A PARENT 
# ==========================================================
def tournament_selection(population, fitness_values, k=3, minimize=False):
    pop_size = len(population)                     # use current size
    k = min(k, pop_size)                            # ensure k ≤ pop_size
    if pop_size == 0:
        raise ValueError("Population is empty – cannot select parents.")
    competitors = random.sample(range(pop_size), k)

    best_index = competitors[0]
    for idx in competitors[1:]:
        if minimize:
            if fitness_values[idx] < fitness_values[best_index]:
                best_index = idx
        else:
            if fitness_values[idx] > fitness_values[best_index]:
                best_index = idx

    return population[best_index]

# ==========================================================
# CROSSOVER ON TWO PARENTS (RANDOM BIT POINTS)
# ==========================================================
def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(p1) - 2)
        return (
        np.concatenate((p1[:point], p2[point:])),
        np.concatenate((p2[:point], p1[point:]))
        )
    return p1[:], p2[:]

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
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
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
def binary_coded_genetic_algorithm(fitness, lb, ub, POP_SIZE, GENERATIONS, n, CROSSOVER_RATE, MUTATION_RATE, k):
    # Generate Initial Population
    population = generate_initial_population(POP_SIZE,)

    # Store best chromosome
    best_solution = None
    # Store best chromosome's fitness value
    best_fitness = float('-inf')

    for gen in range(GENERATIONS):
        new_population = []

        fitness_values = 2
        # [fitness(chromosome, C, R, B) for chromosome in population]
        
        for i in range(POP_SIZE // 2):
            p1 = tournament_selection(population, fitness_values)
            p2 = tournament_selection(population, fitness_values)

            c1, c2 = crossover(p1, p2)

            c1 = mutate(c1)
            c2 = mutate(c2)

            new_population.append(c1)      # now added
            new_population.append(c2)      # now added
            

        # Combine parents and offspring
        combined_population = population + new_population

        # Sort all individuals by fitness (descending) and keep the best POP_SIZE
        # combined_fitness = [fitness(ind, C, R, B) for ind in combined_population]
        # sorted_indices = np.argsort(combined_fitness)[::-1]          # descending order
        # population = [combined_population[i] for i in sorted_indices[:POP_SIZE]]
        
        for chrom in population:
            f = fitness(chrom)
            if f > best_fitness:
                best_fitness = f
                best_solution = chrom

        print(f"Generation {gen+1}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness


# ===============================================================
# GENERATE COST MATRIX, RESOURCE MATRIX, CAPACITY VECTOR FROM FILE
# ===============================================================
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
        print(f"Instance {idx}:")

        lb = 0
        ub = len(C[0]) - 1
        n  = int(np.ceil(np.log2(ub)))
        k  = 3

        best_assignment, best_cost = binary_coded_genetic_algorithm(fitness, lb, ub, POP_SIZE, GENERATIONS, n, CROSSOVER_RATE, MUTATION_RATE, k)

        print(f"  Genetic Algorithm: Best Cost = {best_cost}")

        results.append({
            "Genetic Algorithm": (best_assignment, best_cost)
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
    "gap_sample_data_txt.txt"
    # "gap1.txt",
    # "gap2.txt", "gap3.txt","gap4.txt",
    # "gap5.txt","gap6.txt","gap7.txt","gap8.txt",
    # "gap9.txt","gap10.txt","gap11.txt",    
    # "gap12.txt"
]


# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__": 
    solve_multiple_files(files)