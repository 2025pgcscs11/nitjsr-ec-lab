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
INTERTIA = 0.7
C1 = 1.5
C2 = 1.5


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population(pop_size, m, n):
    # Each particle: length n (jobs), values in [0, m-1]
    return np.random.randint(0, m, size=(pop_size, n))


# ==========================================================
# INITIAL VELOCITY
# ==========================================================
def generate_initial_velocity(pop_size, n):
    return np.random.uniform(1, 10, size=(pop_size, n))


# ==========================================================
# REPAIR PARTICLE USING RANDOM APPROACH
# ==========================================================
def repair_particle_random(particle,C,R,B):
    n = len(C[0])   # number of Jobs
    m = len(C)      # number of Agents

    agents = particle.astype(int)

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

    return agents.astype(float)


# ==========================================================
# FITNESS FUNCTION (Maximization with Penalty)
# ==========================================================
def fitness(particle, C, R):
    cost = 0

    m = len(C)      # number of Agents

    resource_used = [0] * m

    particle = particle.astype(int)

    for j, agent in enumerate(particle):
        cost += C[agent][j]
        resource_used[agent] += R[agent][j]

    return cost


# ==========================================================
# CHECK FEASIBILITY OF EACH particle
# ==========================================================
def is_feasible(particle, R, B):
    m = len(R)        # number of agents
    n = len(R[0])     # number of jobs

    agents = particle.astype(int)

    resource_used = np.zeros(m)

    # Compute resource usage
    for j, agent in enumerate(agents):
        resource_used[agent] += R[agent][j]

        # Early stopping (optimization)
        if resource_used[agent] > B[agent]:
            return False

    return True


# ==========================================================
# PARTICLE SWARM OPTIMIZATION
# ==========================================================
def particle_swarm_optimization(C, R, B):

    m = len(C)          
    n = len(C[0])

    # Initialize population and velocity
    population = generate_initial_population(POP_SIZE, m, n).astype(float)
    velocity = generate_initial_velocity(POP_SIZE, n)

    fitness_values = np.array([
        fitness(population[i], C, R)
        for i in range(POP_SIZE)
    ])

    p_best = population.copy()
    f_p_best = fitness_values.copy()

    g_best_index = np.argmax(f_p_best)
    g_best = p_best[g_best_index].copy()
    f_g_best = f_p_best[g_best_index]

    for _ in range(ITERATIONS):

        for i in range(POP_SIZE):

            r1 = np.random.rand()
            r2 = np.random.rand()

            # Velocity update
            velocity[i] = (
                INTERTIA * velocity[i]
                + C1 * r1 * (p_best[i] - population[i])
                + C2 * r2 * (g_best - population[i])
            )

            # Position update
            population[i] += velocity[i]

            # Bound
            population[i] = np.clip(population[i], 0, m - 1)

            # Fitness
            fitness_values[i] = fitness(population[i], C, R)

            # Discretize for evaluation
            discrete_particle = np.round(population[i])

            # Repair a infeasible solution
            if is_feasible(population[i],R,B):
                population[i] = discrete_particle
            else:
                population[i] = repair_particle_random(discrete_particle,C,R,B)

            # Fitness
            fitness_values[i] = fitness(population[i], C, R)

            # Personal best
            if fitness_values[i] > f_p_best[i]:
                p_best[i] = population[i].copy()
                f_p_best[i] = fitness_values[i]

        # Global best update
        best_index = np.argmax(f_p_best)
        if f_p_best[best_index] > f_g_best:
            g_best = p_best[best_index].copy()
            f_g_best = f_p_best[best_index]

    return g_best, f_g_best


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

        g_best, f_g_best = particle_swarm_optimization(C, R, B)

        print(f"  PSO Best Fitness = {f_g_best}")


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
