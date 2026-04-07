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
TEACHING_FACTOR = 2


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population(pop_size, m, n):
    # Each student: length n (jobs), values in [0, m-1]
    return np.random.randint(0, m, size=(pop_size, n))


# ==========================================================
# REPAIR STUDENT USING RANDOM APPROACH
# ==========================================================
def repair_student_random(student,C,R,B):
    n = len(C[0])   # number of Jobs
    m = len(C)      # number of Agents

    agents = student.astype(int)

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
def fitness(student, C, R, B):
    cost = 0

    m = len(B)
    resource_used = [0] * m
    
    agents = student.astype(int)

    for j, agent in enumerate(agents):
        cost += C[agent][j]
        resource_used[agent] += R[agent][j]


    return cost


# ==========================================================
# CHECK FEASIBILITY OF EACH student
# ==========================================================
def is_feasible(student, R, B):
    m = len(R)        # number of agents
    n = len(R[0])     # number of jobs

    agents = student.astype(int)

    resource_used = np.zeros(m)

    # Compute resource usage
    for j, agent in enumerate(agents):
        resource_used[agent] += R[agent][j]

        # Early stopping (optimization)
        if resource_used[agent] > B[agent]:
            return False

    return True


# ==========================================================
# TEACHING LEARNING BASED OPTIMIZATION
# ==========================================================
def teaching_learning_based_optimization(C, R, B):
    m = len(C)      # number of Agents
    n = len(C[0])   # number of Jobs

    # Initialize random population
    population = generate_initial_population(POP_SIZE, m, n)

    # Evaluate fitness of the population
    fitness_values = np.array([
        fitness(population[i], C, R, B)
        for i in range(POP_SIZE)
    ])


    for _ in range(ITERATIONS):

        for i in range(POP_SIZE):
            ##################################
            #        TEACHING PHASE          #
            ##################################

            # Generate random number array
            r1 = np.random.rand()
            r2 = np.random.rand()

            # Find X_best
            x_best_index = np.argmax(fitness_values)
            x_best = population[x_best_index].copy()

            # Determine X_mean
            x_mean = np.mean(population, axis=0)

            # Calculate x_new
            x_new = population[i] + r1 * (x_best - TEACHING_FACTOR * x_mean)

            # Bound x_new
            x_new = np.clip(x_new,0,m - 1)

            # Discretize
            x_new = np.round(x_new)
            
            # Repair if not feasible
            if not is_feasible(x_new,R,B):
                x_new = repair_student_random(x_new,C,R,B)

            # Calculate fitness of x_new
            f_x_new = fitness(x_new,C,R,B)

            # Compare with the past fitness
            if f_x_new > fitness_values[i]:
                population[i] = x_new.copy()
                fitness_values[i] = f_x_new
            

            ##################################
            #        LEARNER PHASE           #
            ##################################

            # Select a random partner solution other than current solution and its fitness value
            x_p_index = np.random.choice(
                np.delete(np.arange(population.shape[0]), i)
            )
            x_p = population[x_p_index]
            f_x_p = fitness_values[x_p_index]

            # Calculate x_new
            if f_x_p > fitness_values[i]:
                x_new = population[i] + r2 * (population[i] - x_p)
            else:
                x_new = population[i] - r2 * (population[i] - x_p)

            # Bound x_new
            x_new = np.clip(x_new,0,m - 1)

            # Discretize
            x_new = np.round(x_new)

            # Repair if not feasible
            if not is_feasible(x_new,R,B):
                x_new = repair_student_random(x_new,C,R,B)

            # Calculate fitness of x_new
            f_x_new = fitness(x_new,C,R,B)

            # Compare with the past fitness
            if f_x_new > fitness_values[i]:
                population[i] = x_new.copy()
                fitness_values[i] = f_x_new

    best_index = np.argmax(fitness_values)
    
    return population[best_index], fitness_values[best_index]


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

        x_best, f_x_best = teaching_learning_based_optimization(C, R, B)

        print(f"  TLBO Best Fitness = {f_x_best}")


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
