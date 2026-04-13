
# ==========================================================
# IMPORTS MODULE HERE
# ==========================================================
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# ==========================================================
# CONSTANT PARAMETERS
# ==========================================================
POP_SIZE = 100
ITERATIONS = 300
DIMENSION = 10
LOWER_BOUND = 0
UPPER_BOUND = 30
SCALING_FACTOR = 0.85
CROSSOVER_RATE = 0.8

# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population():
    # generate population matrix of pop_size * dimension within the range of 
    return np.random.randint(LOWER_BOUND,UPPER_BOUND, size=(POP_SIZE, DIMENSION))


# ==========================================================
# FITNESS FUNCTION 
# ==========================================================
def fitness(vector):
    return np.sum(vector ** 2)


# ==========================================================
# DIFFERENTIAL EVOLUTION BASED OPTIMIZATION
# ==========================================================
def differential_evolution_based_optimization():
    # Initialize random target vector
    target_vector = generate_initial_population()
    
    # Evaluate fitness of the target vector
    fitness_values = np.array([
        fitness(target_vector[i])
        for i in range(POP_SIZE)
    ])

     # ==========================
    # Global best initialization
    # ==========================
    best_index = np.argmin(fitness_values)
    best_solution = target_vector[best_index].copy()
    best_fitness = fitness_values[best_index]
    
    # Best Fitness per generation (iteration best)
    best_fitness_per_gen = []

    donar_vector = np.zeros((POP_SIZE, DIMENSION))
    trial_vector = np.zeros((POP_SIZE, DIMENSION))

    for t in range(ITERATIONS):

        for i in range(POP_SIZE):
            # Generate random number array
            r1, r2, r3 = np.random.choice(POP_SIZE, 3, replace=False)

            # Generate Donar Vector (mutation)
            donar_vector[i] = target_vector[r1] + SCALING_FACTOR * (target_vector[r2] - target_vector[r3])

            # Generate Trial Vector 
            del_ = np.random.randint(DIMENSION)

            for j in range(DIMENSION):
                if np.random.rand() <= CROSSOVER_RATE or j == del_:
                    trial_vector[i][j] = donar_vector[i][j]
                else:
                    trial_vector[i][j] = target_vector[i][j]

            # Bound
            trial_vector[i] = np.clip(trial_vector[i], LOWER_BOUND, UPPER_BOUND)

            # Selection
            temp = fitness(trial_vector[i])
            if temp < fitness_values[i]:
                target_vector[i] = trial_vector[i].copy()
                fitness_values[i] = temp


        # Iteration best (current population)
        gen_best_index = np.argmin(fitness_values)
        gen_best_solution = target_vector[gen_best_index]
        gen_best_fitness = fitness_values[gen_best_index]

        best_fitness_per_gen.append(gen_best_fitness)

        # Global best update

        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = gen_best_solution.copy()

        # print(f"Iteration {t+1}: Best Fitness = {gen_best_fitness}")

    return best_solution, best_fitness, best_fitness_per_gen


# ==================================================================
# ITERATE oVER 20 RUNS
# ==================================================================
def solve_square_function():
        num_runs = 20
        
        all_histories = []
        all_best_sol = []
        all_best_costs = []
        all_times = []

        # Run DE multiple times
        for run in range(num_runs):
            start_time = time.perf_counter()

            best_assignment, best_cost, fitness_per_gen = differential_evolution_based_optimization()

            end_time = time.perf_counter()

            run_time = end_time - start_time

            all_histories.append(fitness_per_gen)
            all_best_sol.append(best_assignment)
            all_best_costs.append(best_cost)
            all_times.append(run_time)

            print(f"  Run {run+1}: Best Cost = {best_cost}, Time = {run_time:.4f} sec")

        # Convert to numpy array for easier computation
        all_histories = np.array(all_histories)

        # Compute average convergence
        avg_fitness = np.mean(all_histories, axis=0)

        # Plot for THIS instance
        plt.figure()

        # Plot all runs (light)
        for i, history in enumerate(all_histories):
            plt.plot(history, alpha=0.6, label=f"Run {i+1}")

        # Plot average (bold)
        plt.plot(avg_fitness, linewidth=2, label="Average")

        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("DE Convergence (Sphere Function)")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/DE_convergence.png", dpi=300)
        plt.show()



# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__": 
    solve_square_function()
