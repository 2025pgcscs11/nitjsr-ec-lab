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
LOWER_BOUND = -22
UPPER_BOUND = 20
TEACHING_FACTOR = 2


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population():
    # generate population matrix of pop_size * dimension within the range of 
    return np.random.randint(LOWER_BOUND,UPPER_BOUND, size=(POP_SIZE,DIMENSION))


# ==========================================================
# FITNESS FUNCTION (MINIMIZATION)
# ==========================================================
def fitness(student):
    return np.sum(student ** 2)


# ==========================================================
# TEACHING LEARNING BASED OPTIMIZATION
# ==========================================================
def teaching_learning_based_optimization():
    # Initialize random population
    population = generate_initial_population()

    # Evaluate fitness of the population
    fitness_values = np.array([
        fitness(population[i])
        for i in range(POP_SIZE)
    ])

    # ==========================
    # Global best initialization
    # ==========================
    best_index = np.argmin(fitness_values)
    best_solution = population[best_index].copy()
    best_fitness = fitness_values[best_index]

    # Best Fitness per generation (iteration best)
    best_fitness_per_gen = []

    for t in range(ITERATIONS):

        for i in range(POP_SIZE):

            ##################################
            #        TEACHING PHASE          #
            ##################################

            r1 = np.random.rand()
            r2 = np.random.rand()

            # Teacher (best solution in current population)
            x_best_index = np.argmin(fitness_values)
            x_best = population[x_best_index].copy()

            # Mean of population
            x_mean = np.mean(population, axis=0)

            # Generate new solution
            x_new = population[i] + r1 * (x_best - TEACHING_FACTOR * x_mean)
            x_new = np.clip(np.round(x_new), LOWER_BOUND, UPPER_BOUND)

            f_x_new = fitness(x_new)

            # Greedy selection
            if f_x_new < fitness_values[i]:
                population[i] = x_new.copy()
                fitness_values[i] = f_x_new

            ##################################
            #        LEARNER PHASE           #
            ##################################

            # Select random partner
            x_p_index = np.random.choice(
                np.delete(np.arange(population.shape[0]), i)
            )
            x_p = population[x_p_index]
            f_x_p = fitness_values[x_p_index]

            # Learning interaction
            if f_x_p < fitness_values[i]:
                x_new = population[i] + r2 * (population[i] - x_p)
            else:
                x_new = population[i] - r2 * (population[i] - x_p)

            x_new = np.clip(np.round(x_new), LOWER_BOUND, UPPER_BOUND)
            f_x_new = fitness(x_new)

            # Greedy selection
            if f_x_new < fitness_values[i]:
                population[i] = x_new.copy()
                fitness_values[i] = f_x_new

        # Iteration best
        gen_best_index = np.argmin(fitness_values)
        gen_best_solution = population[gen_best_index]
        gen_best_fitness = fitness_values[gen_best_index]

        best_fitness_per_gen.append(gen_best_fitness)

        # Global best update
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = gen_best_solution.copy()

        # print(f"Iteration {t+1}: Iteration Best = {gen_best_fitness}, Global Best = {best_fitness}")

    return best_solution, best_fitness, best_fitness_per_gen

# ==================================================================
# ITERATE OVER 20 RUNS
# ==================================================================
def solve_square_function():

    num_runs = 20

    all_histories = []
    all_best_sol = []
    all_best_costs = []
    all_times = []

    for run in range(num_runs):

        start_time = time.perf_counter()

        best_sol, best_cost, history = teaching_learning_based_optimization()

        end_time = time.perf_counter()

        run_time = end_time - start_time

        all_histories.append(history)
        all_best_sol.append(best_sol)
        all_best_costs.append(best_cost)
        all_times.append(run_time)

        print(f"Run {run+1}: Best Cost = {best_cost:.6f}, Time = {run_time:.4f} sec")

    all_histories = np.array(all_histories)
    avg_curve = np.mean(all_histories, axis=0)

    # ==================================================
    # PLOT
    # ==================================================
    plt.figure()

    for i, history in enumerate(all_histories):
        plt.plot(history, alpha=0.6, label=f"Run {i+1}")


    plt.plot(avg_curve, linewidth=2, label="Average")

    plt.xlabel("Generation")
    plt.ylabel("Best Cost (Min)")
    plt.title("TLBO Convergence (Sphere Function)")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/TLBO_convergence.png", dpi=300)
    plt.show()


# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__": 
    solve_square_function()