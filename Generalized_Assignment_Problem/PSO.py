import random
import os

# ALL NECESSARY CONSTANT PARAMETERS
POP_SIZE = 100          # POPULATION SIZE
ITERATIONS = 100        # NUMBER OF ITERATIONS 
LOWER_BOUND = 0         # LOWER BOUND VALUE OF DECISION VARIABLES
UPPER_BOUND = 10        # UPPER BOUND VALUE OF DECISION VARIABLES
INTERTUIA = 0.7         # INTERTIA OF THE A PARTICALE
C1 = 1.5                # ACCERALATION COEFFICIENT
C2 = 1.5                # ACCERALATION COEFFICIENT


def generate_initial_population(C, R, B, POP_SIZE):
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

        # best_assignment, best_cost = genetic_algorithm(C, R, B)

        # print(f"  Genetic Algorithm: Best Cost = {best_cost}")

        results.append({
            # "Genetic Algorithm": (best_assignment, best_cost)
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