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
ITERATIONS = 50
DIMENSION = 10
LOWER_BOUND = 0
UPPER_BOUND = 30
LIMIT = 10 


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population():
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(POP_SIZE, DIMENSION))


# ==========================================================
# OBJECTIVE FUNCTION (MINIMIZATION)
# ==========================================================
def objective_function(food_source):
    return np.sum(food_source ** 2)


# ==========================================================
# FITNESS FUNCTION (for maximization)
# ==========================================================
def fitness(f):
    if f >= 0:
        return 1 / (1 + f)
    else:
        return 1 + abs(f)


# ==========================================================
# ARTIFICIAL BEE COLONY OPTIMIZATION (ABC)
# ==========================================================
def artificial_bee_colony_opimization():
    # Initialize random population
    population = generate_initial_population()

    # Evaluate objective function values of the population
    objective_function_values = np.array([
        objective_function(population[i]) for i in range(POP_SIZE)
    ])

    # Evaluate fitness of the population
    fitness_values = np.array([
        fitness(objective_function_values[i]) for i in range(POP_SIZE)
    ])

    # Initial trial vector of the popultion
    trial_vector = np.zeros(POP_SIZE)

    # Global best initialization
    best_index = np.argmin(fitness_values)
    best_solution = population[best_index].copy()
    best_objective_function_value = objective_function_values[best_index]
    best_fitness = fitness_values[best_index]

    # Best Objective Function Value per iteration
    best_objective_function_value_per_gen = []

    for gen in range(ITERATIONS):

        ##################################
        #     Employed Bee Phase         #
        ##################################
        for i in range(POP_SIZE):
            # Select a random variable from 0 to DIMENSION -1
            j = np.random.randint(0, DIMENSION)

            # Select a random partner solution other than current solution
            k = i
            while k == i:
                k = np.random.randint(0, POP_SIZE)

            # Select a random number between (-1,1)
            phi = np.random.uniform(-1, 1)

            # Modify jth variable
            x_new = population[i].copy()
            x_new[j] = x_new[j] + phi * (x_new[j] - population[k][j])
            x_new = np.clip(x_new, LOWER_BOUND, UPPER_BOUND)

            # Evaluate the objective function and fitness of newly generated solution
            f_new = objective_function(x_new)
            fit_new = fitness(f_new)

            # Greedy selection and trial vector updation
            if fit_new > fitness_values[i]:
                population[i] = x_new
                objective_function_values[i] = f_new
                fitness_values[i] = fit_new
                trial_vector[i] = 0
            else:
                trial_vector[i] += 1


        # ==================================================
        # PROBABILITY CALCULATION
        # ==================================================
        prob = 0.9 * (fitness_values / np.max(fitness_values)) + 0.1


        # ==================================================
        # ONLOOKER BEE PHASE
        # ==================================================
        i = 0
        t = 0

        while t < POP_SIZE:
            # Generate random number (0,1)
            r = np.random.rand()

            if r < prob[i]:
                # Select a random varible from 0 to n
                j = np.random.randint(0, DIMENSION)

                # Select a random partner solution other than current solution
                k = i
                while k == i:
                    k = np.random.randint(0, POP_SIZE)

                # Select a random number between (-1,1)
                phi = np.random.uniform(-1, 1)

                x_new = population[i].copy()
                x_new[j] = x_new[j] + phi * (x_new[j] - population[k][j])
                x_new = np.clip(x_new, LOWER_BOUND, UPPER_BOUND)

                # Evaluate the objective function and fitness of newly generated solution
                f_new = objective_function(x_new)
                fit_new = fitness(f_new)

                # Greedy selection and trial vector updation
                if fit_new > fitness_values[i]:
                    population[i] = x_new
                    objective_function_values[i] = f_new
                    fitness_values[i] = fit_new
                    trial_vector[i] = 0
                else:
                    trial_vector[i] += 1

                # Increment t
                t += 1

            i = (i + 1) % POP_SIZE


        # ==================================================
        # SCOUT BEE PHASE
        # ==================================================
        for i in range(POP_SIZE):
            if trial_vector[i] > LIMIT:
                # Generate a random solution
                x_new = np.random.uniform(LOWER_BOUND, UPPER_BOUND + 1, DIMENSION)

                population[i] = x_new
                objective_function_values[i] = objective_function(x_new)
                fitness_values[i] = fitness(objective_function_values[i])
                trial_vector[i] = 0


        # ==================================================
        # GLOBAL BEST UPDATE
        # ==================================================
        gen_best_index = np.argmax(fitness_values)

        if fitness_values[gen_best_index] > best_fitness:
            best_fitness = fitness_values[gen_best_index]
            best_solution = population[gen_best_index].copy()
            best_objective_function_value = objective_function_values[gen_best_index]

        # Store history
        best_objective_function_value_per_gen.append(best_objective_function_value)


    return best_solution, best_objective_function_value, best_objective_function_value_per_gen


# ==========================================================
# MULTIPLE RUNS
# ==========================================================
def solve_square_function():
    results = []
    num_runs = 20

    all_histories = []
    all_best_values = []
    all_best_sol = []
    all_best_Values = []
    all_times = []

    for run in range(num_runs):

        start_time = time.perf_counter()

        best_sol, best_value, history = artificial_bee_colony_opimization()

        end_time = time.perf_counter()

        run_time = end_time - start_time

        all_histories.append(history)
        all_best_values.append(best_value)
        all_best_sol.append(best_sol)
        all_best_Values.append(best_value)
        all_times.append(run_time)

        print(f"Run {run+1}: Best Value = {best_value:}, Time = {run_time:.4f} sec")

    all_histories = np.array(all_histories)
    avg_curve = np.mean(all_histories, axis=0)

    # ==================================================
    # PLOT WITH CONVERGENCE THRESHOLD
    # ==================================================
    plt.figure()

    # Define threshold (epsilon) – adjust based on problem
    EPS = 1e-6  # if best fitness difference < EPS, consider converged

    for i, history in enumerate(all_histories):
        final_best = history[-1]
        # Find first generation where fitness <= final_best + EPS
        converged_idx = len(history)  # default to full length
        for gen, val in enumerate(history):
            if val <= final_best + EPS:
                converged_idx = gen
                break
        # Truncate history up to converged_idx (inclusive)
        truncated = history[:converged_idx+1]
        plt.plot(truncated, alpha=0.6, label=f"Run {i+1}")

    # For the average curve, keep full history (or truncate using same logic)
    avg_curve = np.mean(all_histories, axis=0)
    plt.plot(avg_curve, linewidth=2, label="Average")

    plt.xlabel("GENERATION")
    plt.ylabel("BEST VALUE (MIN)")
    plt.title(f"CONVERGENCE GRAPH || ARTIFICIAL BEE COLONY OPTIMIZATION ALGORITHM || SPHERE FUNCTION")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # os.makedirs("plots", exist_ok=True)
    # plt.savefig("plots/BCGA_convergence.png", dpi=300)
    plt.show()

    # Store results
    results.append({
        "histories": all_histories,
        "best_value_per_run": all_best_values,
        "best_solution_per_run": all_best_sol,
        "time_per_run": all_times,
    })

    return results


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    all_results = solve_square_function()

    for result in all_results:

        best_values = np.array(result["best_value_per_run"])
        times = np.array(result["time_per_run"])

        print("\n===== FINAL SUMMARY (SPHERE FUNCTION) =====")

        print(f"Number of runs        : {len(best_values)}")
        print(f"Best fitness (min)    : {np.min(best_values):}")
        print(f"Worst fitness         : {np.max(best_values):}")
        print(f"Average fitness       : {np.mean(best_values):}")
        print(f"Std deviation         : {np.std(best_values):}")

        print(f"\nAverage time/run      : {np.mean(times):.4f} sec")
        print(f"Total execution time  : {np.sum(times):.4f} sec")