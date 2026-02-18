import random
import os

POP_SIZE = 100
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
PENALTY_WEIGHT = 1000


# ==========================================================
# INITIAL POPULATION (Binary Encoding)
# ==========================================================
def generate_initial_population(m, n):
    population = []

    for _ in range(POP_SIZE):
        chromosome = [0] * (m * n)

        # Each job assigned to exactly one agent
        for j in range(n):
            agent = random.randint(0, m - 1)
            chromosome[agent * n + j] = 1

        population.append(chromosome)

    return population


# ==========================================================
# FITNESS FUNCTION (Maximization with Penalty)
# ==========================================================
def fitness(chromosome, C, R, B):
    """
    Maximization: Profit - Penalty
    """

    m = len(C)
    n = len(C[0])

    total_profit = 0
    penalty = 0
    usage = [0] * m

    for i in range(m):
        for j in range(n):
            if chromosome[i*n + j] == 1:
                total_profit += C[i][j]     # PROFIT
                usage[i] += R[i][j]

    # Capacity violation penalty
    for i in range(m):
        if usage[i] > B[i]:
            penalty += (usage[i] - B[i])

    return total_profit - PENALTY_WEIGHT * penalty



# ==========================================================
# TOURNAMENT SELECTION
# ==========================================================
def tournament_selection(population, C, R, B, k=3):
    selected = random.sample(population, k)
    return max(selected, key=lambda chrom: fitness(chrom, C, R, B))


# ==========================================================
# ONE-POINT CROSSOVER (Job Based)
# ==========================================================
def crossover(p1, p2, m, n):
    if random.random() > CROSSOVER_RATE:
        return p1[:], p2[:]

    point = random.randint(1, n - 1)

    child1 = [0] * (m * n)
    child2 = [0] * (m * n)

    for j in range(n):
        for i in range(m):
            if j < point:
                child1[i*n + j] = p1[i*n + j]
                child2[i*n + j] = p2[i*n + j]
            else:
                child1[i*n + j] = p2[i*n + j]
                child2[i*n + j] = p1[i*n + j]

    return child1, child2


# ==========================================================
# MUTATION (Job Reassignment — Keeps Feasibility of Eq Constraint)
# ==========================================================
def mutate(chromosome, m, n):
    if random.random() < MUTATION_RATE:

        # Select random job
        j = random.randint(0, n - 1)

        # Find current agent
        current_agent = None
        for i in range(m):
            if chromosome[i*n + j] == 1:
                current_agent = i
                break

        # Assign to new agent
        new_agent = random.choice([i for i in range(m) if i != current_agent])

        chromosome[current_agent*n + j] = 0
        chromosome[new_agent*n + j] = 1

    return chromosome


# ==========================================================
# REPAIR INFEASIBLE SOLUTIONS
# ==========================================================
def repair_feasibility(chromosome, C, R, B):
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
    Generate initial feasible population for Generalized Assignment Problem

    C : cost matrix (m x n)
    R : resource matrix (m x n)
    B : capacity vector (m)
    pop_size : number of chromosomes
    """

    m = len(C)       # number of agents
    n = len(C[0])    # number of jobs

    population = []

    for _ in range(pop_size):

        # Initialize chromosome and remaining capacity
        chromosome = [0] * (m * n)
        remaining_capacity = B[:]   # copy

        # Assign jobs one by one
        for j in range(n):

            # Find feasible agents for job j
            feasible_agents = [
                i for i in range(m)
                if R[i][j] <= remaining_capacity[i]
            ]

            # If no feasible agent exists, choose any agent (will be repaired later)
            if not feasible_agents:
                chosen_agent = random.randint(0, m - 1)
            else:
                # Random feasible selection (for diversity)
                chosen_agent = random.choice(feasible_agents)

            # Assign job j to chosen agent
            chromosome[chosen_agent * n + j] = 1
            remaining_capacity[chosen_agent] -= R[chosen_agent][j]

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
    m, n = len(B), len(R[0])
    remaining = B[:]

    # Fix assignment constraint
    for j in range(n):
        assigned = [i for i in range(m) if chromosome[i*n + j] == 1]

        if len(assigned) == 0:
            chromosome[random.randint(0, m-1)*n + j] = 1

        elif len(assigned) > 1:
            keep = random.choice(assigned)
            for i in assigned:
                chromosome[i*n + j] = 0
            chromosome[keep*n + j] = 1

    # Fix capacity constraint
    for i in range(m):
        for j in range(n):
            if chromosome[i*n + j] == 1:
                remaining[i] -= R[i][j]

        while remaining[i] < 0:
            jobs = [j for j in range(n) if chromosome[i*n + j] == 1]
            job = random.choice(jobs)
            chromosome[i*n + job] = 0
            remaining[i] += R[i][job]

            for new_agent in range(m):
                if R[new_agent][job] <= remaining[new_agent]:
                    chromosome[new_agent*n + job] = 1
                    remaining[new_agent] -= R[new_agent][job]
                    break

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
            c1 = repair(mutate(c1), R, B)
            c2 = repair(mutate(c2), R, B)

            new_population.extend([c1, c2])

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




def solve_multiple_files(file_list, base_dir="gap_dataset"):
    all_results = {}
    # Directory where BCGA.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for file in file_list:
        file_path = os.path.join(script_dir, base_dir, file)

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