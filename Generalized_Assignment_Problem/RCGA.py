import random
import os

POP_SIZE = 100
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
DISTRIBUTIOIN_INDEX = 20
MUTATION_DISTRIBUTION_INDEX = 20
UPPER_LIMIT = 30
LOWER_LIMIT = 0

def generate_initial_population(C, R, B, pop_size):
    """
    Generate ONLY feasible initial population for GAP
    """

    m = len(C)       # number of agents
    n = len(C[0])    # number of jobs

    population = []

    while len(population) < pop_size:

        chromosome = [0] * (m * n)
        remaining_capacity = B[:]

        feasible = True

        # Assign jobs one by one
        for j in range(n):

            feasible_agents = [
                i for i in range(m)
                if R[i][j] <= remaining_capacity[i]
            ]

            # ❌ If no feasible agent exists → discard chromosome
            if not feasible_agents:
                feasible = False
                break

            chosen_agent = random.choice(feasible_agents)

            chromosome[j] = chosen_agent
            remaining_capacity[chosen_agent] -= R[chosen_agent][j]

        # ✔ Keep only feasible chromosomes
        if feasible:
            population.append(chromosome)

    return population



def decode_gene(gene, m, lower=0, upper=30):
    """
    Convert real gene value back to agent index.
    """

    # Scale back to [0, m-1]
    agent = int(round((gene - lower) / (upper - lower) * (m - 1)))

    # Clamp to valid range
    agent = max(0, min(m - 1, agent))

    return agent


def fitness(chromosome, C, lower=0, upper=30):
    """
    Compute total profit for RCGA encoding.
    chromosome[j] = scaled real value representing agent.
    """

    m = len(C)
    n = len(C[0])

    total_profit = 0

    for j in range(n):
        agent = decode_gene(chromosome[j], m, lower, upper)
        total_profit += C[agent][j]

    return total_profit

def repair(chromosome, R, B, lower=0, upper=30):
    """
    Check capacity feasibility for RCGA encoding.
    Returns chromosome if feasible, otherwise None.
    """

    m = len(B)
    n = len(chromosome)

    usage = [0] * m

    for j in range(n):
        agent = decode_gene(chromosome[j], m, lower, upper)
        usage[agent] += R[agent][j]

        if usage[agent] > B[agent]:
            return None

    return chromosome


def select(population, C):
    a, b = random.sample(population, 2)
    return a if fitness(a, C) > fitness(b, C) else b


def crossover(p1, p2):
    """
    Simulated Binary Crossover (SBX)

    p1, p2 : parent chromosomes (list of real values)
    CROSSOVER_RATE : probability of crossover
    nc : distribution index (controls spread)
    """

    # If random number >= crossover rate → children = parents
    if random.random() >= CROSSOVER_RATE:
        return p1[:], p2[:]

    child1 = []
    child2 = []

    for x1, x2 in zip(p1, p2):

        u = random.random()

        # Compute beta
        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (DISTRIBUTIOIN_INDEX + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1.0 / (DISTRIBUTIOIN_INDEX + 1))

        # Generate children
        c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
        c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)

        child1.append(c1)
        child2.append(c2)

    return child1, child2


def mutate(chromosome):
    """
    Polynomial Mutation

    chromosome : list of real values
    """

    for i in range(len(chromosome)):

        if random.random() < MUTATION_RATE:

            r = random.random()

            if r < 0.5:
                delta = (2 * r) ** (1.0 / (MUTATION_DISTRIBUTION_INDEX + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r)) ** (1.0 / (MUTATION_DISTRIBUTION_INDEX + 1))

            # Apply mutation
            chromosome[i] = chromosome[i] + delta * (UPPER_LIMIT - LOWER_LIMIT)

            # Keep within bounds
            chromosome[i] = max(LOWER_LIMIT, min(UPPER_LIMIT, chromosome[i]))

    return chromosome

# def mutate(chromosome,C):
#     m = len(C)       # number of agents
#     n = len(C[0])
#     for j in range(n):
#         if random.random() < MUTATION_RATE:
#             # remove current assignment
#             for i in range(m):
#                 chromosome[i*n + j] = 0
            
#             # assign to random agent
#             new_agent = random.randint(0, m-1)
#             chromosome[new_agent*n + j] = 1

#     return chromosome


def genetic_algorithm(C, R, B):
    population = generate_initial_population(C, R, B, POP_SIZE)
    best_solution = None
    best_fitness = float('-inf')

    for gen in range(GENERATIONS):
        new_population = []

        while len(new_population) < POP_SIZE:
            p1 = select(population, C)
            p2 = select(population, C)

            c1, c2 = crossover(p1, p2)

            # Child 1
            c1 = mutate(c1)
            c1 = repair(c1, R, B)
            if c1 is not None:
                new_population.append(c1)

            # Child 2
            c2 = mutate(c2)
            c2 = repair(c2, R, B)
            if c2 is not None:
                new_population.append(c2)

        population = new_population[:POP_SIZE]

        for chrom in population:
            f = fitness(chrom, C)
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




def solve_multiple_files(file_list,base_dir="gap_dataset"):
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



files = [
    "gap1.txt",
    "gap2.txt", "gap3.txt","gap4.txt",
    "gap5.txt","gap6.txt","gap7.txt","gap8.txt",
    "gap9.txt","gap10.txt","gap11.txt",
    "gap12.txt"
]


solve_multiple_files(files)