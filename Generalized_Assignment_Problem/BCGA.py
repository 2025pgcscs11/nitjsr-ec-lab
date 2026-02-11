import random
import os

POP_SIZE = 10
GENERATIONS = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

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

            chromosome[chosen_agent * n + j] = 1
            remaining_capacity[chosen_agent] -= R[chosen_agent][j]

        # ✔ Keep only feasible chromosomes
        if feasible:
            population.append(chromosome)

    return population



def fitness(chromosome, C):
    """
    Compute fitness (total profit) of a chromosome for GAP

    chromosome : binary list of length m*n
    C : cost matrix (m x n)
    """

    m = len(C)
    n = len(C[0])

    total_profit = 0

    for i in range(m):          # agents
        for j in range(n):      # jobs
            index = i * n + j
            total_profit += C[i][j] * chromosome[index]

    return total_profit

def repair(chromosome, R, B):
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


def select(population, C):
    a, b = random.sample(population, 2)
    return a if fitness(a, C) > fitness(b, C) else b


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




def solve_multiple_files(file_list,base_dir="./gap_dataset"):
    all_results = {}
    # Directory where BCGA.py is located
    # script_dir = os.path.dirname(os.path.abspath(__file__))

    for file in file_list:
        file_path = os.path.join(base_dir,file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GAP file not found: {file_path}")
        
        all_results[file] = solve_gap_file(file_path)

    return all_results



files = [
    "gap1.txt",
    # "gap2.txt", "gap3.txt","gap4.txt",
    # "gap5.txt","gap6.txt","gap7.txt","gap8.txt",
    # "gap9.txt","gap10.txt","gap11.txt",
    "gap12.txt"
]


solve_multiple_files(files)