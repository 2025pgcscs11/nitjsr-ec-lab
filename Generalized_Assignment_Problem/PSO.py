import random
import os
import numpy as np

# ==========================================================
# CONSTANT PARAMETERS
# ==========================================================
POP_SIZE = 200
ITERATIONS = 1000
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
# FITNESS FUNCTION (Maximization with Penalty)
# ==========================================================
def fitness(particle, C, R, B, penalty_weight=1000):

    cost = 0
    penalty = 0

    m = len(B)
    resource_used = [0] * m

    for j in range(len(particle)):
        agent = int(particle[j])
        cost += C[agent][j]
        resource_used[agent] += R[agent][j]

    for a in range(m):
        if resource_used[a] > B[a]:
            penalty += (resource_used[a] - B[a])

    return cost - penalty_weight * penalty


# ==========================================================
# INITIAL VELOCITY
# ==========================================================
def generate_initial_velocity(pop_size, n):
    return np.zeros((pop_size, n), dtype=int)


# ==========================================================
# PARTICLE SWARM OPTIMIZATION
# ==========================================================
def particle_swarm_optimization(C, R, B):

    m = len(C)
    n = len(C[0])

    # Initialize
    population = generate_initial_population(POP_SIZE, m, n)
    velocity = generate_initial_velocity(POP_SIZE, n)

    fitness_values = np.array([
        fitness(population[i], C, R, B)
        for i in range(POP_SIZE)
    ])

    p_best = population.copy()
    f_p_best = fitness_values.copy()

    g_best_index = np.argmax(f_p_best)
    g_best = p_best[g_best_index].copy()
    f_g_best = f_p_best[g_best_index]

    for t in range(ITERATIONS):

        for i in range(POP_SIZE):

            r1 = np.random.rand(n)
            r2 = np.random.rand(n)

            # Velocity update
            velocity[i] = (
                INTERTIA * velocity[i]
                + C1 * r1 * (p_best[i] - population[i])
                + C2 * r2 * (g_best - population[i])
            )

            # Position update
            population[i] += velocity[i]

            # Discretize & bound
            population[i] = np.clip(
                np.round(population[i]),
                0,
                m - 1
            )

            # Fitness
            fitness_values[i] = fitness(population[i], C, R, B)

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


# ==========================================================
# READ GAP FILE
# ==========================================================
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


# ==========================================================
# SOLVE FILE
# ==========================================================
def solve_gap_file(filename):

    instances = read_gap_file(filename)

    print(f"\n===== Solving file: {filename} =====\n")

    for idx, (C, R, B) in enumerate(instances, start=1):

        print(f"Instance {idx}:")

        g_best, f_g_best = particle_swarm_optimization(C, R, B)

        print(f"  PSO Best Fitness = {f_g_best}")


# ==========================================================
# MULTIPLE FILES
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
# MAIN
# ==========================================================
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
    "gap12.txt",
]

if __name__ == "__main__":
    solve_multiple_files(files)
