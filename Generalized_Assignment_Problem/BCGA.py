import random
import os

POP_SIZE = 10
GENERATIONS = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1


def generate_initial_population(C, R, B, pop_size):
    """
    Generate ONLY feasible initial population for GAP
    """

    m = len(C)       # number of agents
    n = len(C[0])    # number of jobs

    population = []

    while len(population) < POP_SIZE:

        chromosome = [0] * (m * n)
        remaining_capacity = B[:]

        feasible = True

        # Assign jobs one by one
        for j in range(n):

            feasible_agents = [
                i for i in range(m)
                if R[i][j] <= remaining_capacity[i]
            ]

            # If no feasible agent exists → discard chromosome
            if not feasible_agents:
                feasible = False
                break

            chosen_agent = random.choice(feasible_agents)

            chromosome[chosen_agent * n + j] = 1
            remaining_capacity[chosen_agent] -= R[chosen_agent][j]

        # ✔ Keep only feasible chromosomes
        if feasible:
            population.append(chromosome)

    return population


def fitness(chromosome, C, R, B, penalty_weight=1000):
    m = len(C)
    n = len(C[0])

    total_profit = 0
    penalty = 0

    for i in range(m):
        used = 0
        for j in range(n):
            index = i * n + j
            if chromosome[index] == 1:
                total_profit += R[i][j]
                used += C[i][j]

        # capacity violation
        if used > B[i]:
            penalty += max(0,(used - B[i]))

    return total_profit - penalty_weight * penalty



def feasibility_check(chromosome, R, B):
    """
    Feasibility checker for GAP.
    Returns chromosome if feasible, otherwise returns None.
    """

    m, n = len(B), len(R[0])

    # ---------- Check assignment constraint ----------
    for j in range(n):
        assigned = sum(chromosome[i*n + j] for i in range(m))
        if assigned != 1:
            return None

    # ---------- Check capacity constraint ----------
    usage = [0] * m

    for i in range(m):
        for j in range(n):
            if chromosome[i*n + j] == 1:
                usage[i] += R[i][j]
                if usage[i] > B[i]:
                    return None

    # ✔ Feasible
    return chromosome


def tournament_selection(population, C,R,B,pool_size = 6, k=3):
    """
    population        : list of chromosomes
    C                 : fitness matrix / data
    pool_size         : number of individuals to select
    k                 : tournament size / maximum participation
    """

    n = len(population)
    participation_count = [0] * n
    mating_pool = []

    while len(mating_pool) < pool_size:

        # Select only individuals that haven't exceeded participation
        eligible_indices = [
            i for i in range(n)
            if participation_count[i] < k
        ]

        # Stop if not enough eligible individuals
        if len(eligible_indices) < k:
            break

        # Randomly choose k eligible individuals
        selected_indices = random.sample(eligible_indices, k)

        # Update participation count
        for idx in selected_indices:
            participation_count[idx] += 1

        # Find best among selected
        best_idx = selected_indices[0]
        best_fitness = fitness(population[best_idx], C,R,B)

        for idx in selected_indices[1:]:
            f = fitness(population[idx], C,R,B)
            if f > best_fitness:   
                best_idx = idx
                best_fitness = f

        mating_pool.append(population[best_idx].copy())

    
    mating_pool.sort(reverse=True)

    return mating_pool[0].copy()


def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(p1) - 2)
        return (
            p1[:point] + p2[point:],
            p2[:point] + p1[point:]
        )
    return p1[:], p2[:]


def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] ^= 1
    return chromosome

# def mutate(chromosome):
#     for i in range(len(chromosome)):
#         if random.random() < MUTATION_RATE:
#             chromosome[i] ^= 1
#     return chromosome

def mutate(chromosome,C):
    m = len(C)       # number of agents
    n = len(C[0])
    for j in range(n):
        if random.random() < MUTATION_RATE:
            # remove current assignment
            for i in range(m):
                chromosome[i*n + j] = 0
            
            # assign to random agent
            new_agent = random.randint(0, m-1)
            chromosome[new_agent*n + j] = 1

    return chromosome


def genetic_algorithm(C, R, B):
    population = generate_initial_population(C, R, B)
    best_solution = None
    best_fitness = float('-inf')

    for gen in range(GENERATIONS):
        new_population = []

        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population, C,R,B)
            p2 = tournament_selection(population, C,R,B)

            c1, c2 = crossover(p1, p2)

            # Child 1
            c1 = mutate(c1)
            c1 = feasibility_check(c1, R, B)
            c1 = mutate(c1,C)
            c1 = repair(c1, R, B)
            if c1 is not None:
                new_population.append(c1)

            # Child 2
            c2 = mutate(c2)
            c2 = feasibility_check(c2, R, B)
            c2 = mutate(c2,C)
            c2 = repair(c2, R, B)
            if c2 is not None:
                new_population.append(c2)

        population = new_population[:POP_SIZE]

        for chrom in population:
            f = fitness(chrom, C,R,B)
            if f > best_fitness:
                best_fitness = f
                best_solution = chrom

        print(f"Generation {gen+1}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness


def read_gap_file(filename):
    """
    Reads a GAP data file and returns a list of problem instances.
    Each instance is a tuple: (C, R, B)
    """
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


def solve_gap_file(filename):
    """
    Iterate over all instances in the file and run BCGA on ech instance of GAP problem 
    """
    instances = read_gap_file(filename)
    results = []

    print(f"\n===== Solving file: {filename} =====\n")

    for idx, (C, R, B) in enumerate(instances, start=1):
        print(f"Instance {idx}:")

        best_assignment, best_cost = genetic_algorithm(C, R, B)

        print(f"  Genetic Algorithm: Best Cost = {best_cost}")

        results.append({
            "Genetic Algorithm": (best_assignment, best_cost)
        })

    return results

def read_gap_file(filename):
    """
    Reads a GAP data file and returns a list of problem instances.
    Each instance is a tuple: (C, R, B)
    """
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


def solve_multiple_files(file_list,base_dir="gap_dataset"):
    """
    Generates the files path and iterate over each file 
    """
    all_results = {}
    
    # Absolute path of current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Full path to dataset folder
    dataset_dir = os.path.join(script_dir, base_dir)
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


# All files name
files = [   
    "gap1.txt",
    "gap2.txt",
    "gap3.txt",
    "gap4.txt",
    "gap5.txt",
    "gap6.txt",
    "gap7.txt",
    "gap8.txt",
    "gap9.txt",
    "gap10.txt",
    "gap11.txt",
]

# Execution starts here
if __name__ == "__main__": 
    solve_multiple_files(files)

solve_multiple_files(files)