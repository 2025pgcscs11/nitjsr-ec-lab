
# ==========================================================
# IMPORTS MODULE HERE
# ==========================================================
import os
import numpy as np

# ==========================================================
# CONSTANT PARAMETERS
# ==========================================================
POP_SIZE = 500
ITERATIONS = 200
SCALING_FACTOR = 0.85
CROSSOVER_RATE = 0.8


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population(pop_size, m, n):
    # Each chromosome: length n (jobs), values in [0, m-1]
    return np.random.randint(0, m-1, size=(pop_size, n))


# ==========================================================
# REPAIR TRIAL USING RANDOM APPROACH
# ==========================================================
def repair_trial_random(trial,C,R,B):
    n = len(C[0])   # number of Jobs
    m = len(C)      # number of Agents

    agents = trial.astype(int)

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
# FITNESS FUNCTION (Maximization with Penalty)
# ==========================================================
def fitness(vector, C, R, B):
    cost = 0

    m = len(B)
    resource_used = [0] * m
    
    agents = vector.astype(int)

    for j, agent in enumerate(agents):
        cost += C[agent][j]
        resource_used[agent] += R[agent][j]


    return cost

# ==========================================================
# CHECK FEASIBILITY OF EACH trial
# ==========================================================
def is_feasible(trial, R, B):
    m = len(R)        # number of agents
    n = len(R[0])     # number of jobs

    agents = trial.astype(int)

    resource_used = np.zeros(m)

    # Compute resource usage
    for j, agent in enumerate(agents):
        resource_used[agent] += R[agent][j]

        # Early stopping (optimization)
        if resource_used[agent] > B[agent]:
            return False

    return True


# ==========================================================
# DIFFERENTIAL EVOLUTION BASED OPTIMIZATION
# ==========================================================
def differential_evolution_based_optimization(C, R, B):

    m = len(C)      # number of Agents
    n = len(C[0])   # number of Jobs

    # Initialize random target vector
    target_vector = generate_initial_population(POP_SIZE, m, n)
    donar_vector = np.zeros((POP_SIZE, n))
    trial_vector =  np.zeros((POP_SIZE, n))

    # Evaluate fitness of the target vector
    fitness_values = np.array([
        fitness(target_vector[i], C, R, B)
        for i in range(POP_SIZE)
    ])

    for t in range(ITERATIONS):

        for i in range(POP_SIZE):
            # Generate random number array
            r1, r2, r3 = np.random.choice(POP_SIZE, 3, replace=False)

            # Generate Donar Vector (mutation)
            donar_vector[i] = target_vector[r1] + SCALING_FACTOR * (target_vector[r2] - target_vector[r3])

            # Generate Trial Vector (crossover)
            del_ = np.random.randint(n)
            r = np.random.rand()

            if r <= CROSSOVER_RATE or i == del_:
                trial_vector[i] = donar_vector[i]
            elif r > CROSSOVER_RATE and i != del_:
                trial_vector[i] = target_vector[i]
            
        
        for i in range(POP_SIZE):
            # Bound
            trial_vector[i] = np.clip(trial_vector[i],0 ,m - 1)

            # Discretize
            trial_vector[i] = np.round(trial_vector[i])

            # Repair if not feasible
            if not is_feasible(trial_vector[i],R,B):
                trial_vector[i] = repair_trial_random(trial_vector[i],C,R,B)
            
            temp = fitness(trial_vector[i],C ,R ,B)
            if temp > fitness_values[i]:
                target_vector[i] = trial_vector[i]
                fitness_values[i] = temp 
        

    best_index = np.argmax(fitness_values)
    return target_vector[best_index], fitness_values[best_index]


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

    print(f"\n===== Solving file: {filename} =====\n")

    for idx, (C, R, B) in enumerate(instances, start=1):

        print(f"Instance {idx}:")

        x_best, f_x_best = differential_evolution_based_optimization(C, R, B)

        print(f"  DE Best Fitness = {f_x_best}")


# ==========================================================
# ITERATE OVER ALL FILES
# ==========================================================
def solve_multiple_files(file_list, base_dir="gap_dataset"):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, base_dir)

    for file in file_list:

        file_path = os.path.join(dataset_dir, file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GAP file not found: {file_path}")

        solve_gap_file(file_path)


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
