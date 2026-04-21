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
ITERATIONS = 200
DIMENSION = 10
LOWER_BOUND = 0
UPPER_BOUND = 30
INERTIA = 0.7
C1 = 1.5
C2 = 1.5


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population():
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(POP_SIZE,DIMENSION))


# ==========================================================
# INITIAL VELOCITY
# ==========================================================
def generate_initial_velocity():
    return np.random.uniform(-1, 1, size=(POP_SIZE, DIMENSION))


# ==========================================================
# FITNESS FUNCTION (MINIMIZATION)
# ==========================================================
def fitness(particle):
    return np.sum(particle ** 2)


# ==========================================================
# PARTICLE SWARM OPTIMIZATION
# ==========================================================
def particle_swarm_optimization():
    # Initialize population and velocity
    population = generate_initial_population()
    velocity = generate_initial_velocity()

    fitness_values = np.array([
        fitness(population[i])
        for i in range(POP_SIZE)
    ])

    p_best = population.copy()
    f_p_best = fitness_values.copy()

    # Initialize global best (still needed for velocity update)
    g_best_index = np.argmin(f_p_best)
    g_best = p_best[g_best_index].copy()
    f_g_best = f_p_best[g_best_index]

    # Best Fitness per iteration (ONLY iteration best)
    best_fitness_per_gen = []

    for gen in range(ITERATIONS):

        for i in range(POP_SIZE):

            r1 = np.random.rand()
            r2 = np.random.rand()

            # velocity update
            velocity[i] = (
                INERTIA * velocity[i]
                + C1 * r1 * (p_best[i] - population[i])
                + C2 * r2 * (g_best - population[i])
            )

            # Position update
            population[i] += velocity[i]

            # Bound
            population[i] = np.clip(population[i], LOWER_BOUND, UPPER_BOUND)

            # Fitness
            fitness_values[i] = fitness(population[i])

            # Personal best update
            if fitness_values[i] < f_p_best[i]:
                p_best[i] =population[i].copy()
                f_p_best[i] = fitness_values[i]


        # Global best update 
        best_index = np.argmin(f_p_best)
        if f_p_best[best_index] < f_g_best:
            g_best = p_best[best_index].copy()
            f_g_best = f_p_best[best_index]
        

        best_fitness_per_gen.append(f_g_best)
        # print(f"Generation {gen+1}: Iteration Best = {f_g_best}")

    return g_best, f_g_best, best_fitness_per_gen


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

        best_sol, best_value, history = particle_swarm_optimization()

        end_time = time.perf_counter()

        run_time = end_time - start_time

        all_histories.append(history)
        all_best_values.append(best_value)
        all_best_sol.append(best_sol)
        all_best_Values.append(best_value)
        all_times.append(run_time)

        print(f"Run {run+1}: Best Value = {best_value:.6f}, Time = {run_time:.4f} sec")

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
    plt.ylabel("Best Value (Min)")
    plt.title("PSO Convergence (Sphere Function)")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/PSO_convergence.png", dpi=300)
    plt.show()

    # Store results
    results.append({
        "histories": all_histories,
        "best_Value_per_run": all_best_values,
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

        best_values = np.array(result["best_Value_per_run"])
        times = np.array(result["time_per_run"])

        print("\n===== FINAL SUMMARY (SPHERE FUNCTION) =====")

        print(f"Number of runs        : {len(best_values)}")
        print(f"Best fitness (min)    : {np.min(best_values):.6f}")
        print(f"Worst fitness         : {np.max(best_values):.6f}")
        print(f"Average fitness       : {np.mean(best_values):.6f}")
        print(f"Std deviation         : {np.std(best_values):.6f}")

        print(f"\nAverage time/run      : {np.mean(times):.4f} sec")
        print(f"Total execution time  : {np.sum(times):.4f} sec")