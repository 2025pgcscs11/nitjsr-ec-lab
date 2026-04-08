
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
SCALING_FACTOR = 0.5
CROSSOVER_RATE = 0.9

# ==========================================================
# INITIAL POPULATION
# ==========================================================
def generate_initial_population(pop_size,dimension):
    # generate population matrix of pop_size * dimension within the range of 
    return np.random.randint(LOWER_BOUND,UPPER_BOUND, size=(pop_size, dimension))


# ==========================================================
# FITNESS FUNCTION 
# ==========================================================
def fitness(solution):
    cost = 0.0
    
    for dec_var in solution:
        cost += dec_var * dec_var 
    
    return cost


# ==========================================================
# DIFFERENTIAL EVOLUTION BASED OPTIMIZATION
# ==========================================================
def differential_evolution_based_optimization():
    # Initialize random target vector
    target_vector = generate_initial_population(POP_SIZE,DIMENSION)

    # Store best solution
    best_solution = None
    # Store best solution's fitness value
    best_fitness = float('inf')
    # Best Fitness per generation
    best_fitness_per_gen = []

    donar_vector = np.zeros((POP_SIZE, DIMENSION))
    trial_vector =  np.zeros((POP_SIZE, DIMENSION))

    # Evaluate fitness of the target vector
    fitness_values = np.array([
        fitness(target_vector[i])
        for i in range(POP_SIZE)
    ])

    for t in range(ITERATIONS):

        for i in range(POP_SIZE):
            # Generate random number array
            r1, r2, r3 = np.random.choice(POP_SIZE, 3, replace=False)

            # Generate Donar Vector (mutation)
            donar_vector[i] = target_vector[r1] + SCALING_FACTOR * (target_vector[r2] - target_vector[r3])

            # Generate Trial Vector (crossover)
            del_ = np.random.randint(DIMENSION)
            r = np.random.rand()

            if r <= CROSSOVER_RATE or i == del_:
                trial_vector[i] = donar_vector[i]
            elif r > CROSSOVER_RATE and i != del_:
                trial_vector[i] = target_vector[i]

            # Store best solution in generation    
            gen_best_solution = None
            # Store best fitness in generation    
            gen_best_fitness = float('inf')
        
            for j in range(POP_SIZE):
                # Bound
                trial_vector[j] = np.clip(trial_vector[j],LOWER_BOUND, UPPER_BOUND)

                temp = fitness(trial_vector[j])
                if temp < fitness_values[j]:
                    target_vector[j] = trial_vector[j]
                    fitness_values[j] = temp

        gen_best_fitness_index = np.argmin(fitness_values)
        gen_best_solution = target_vector[gen_best_fitness_index]
        gen_best_fitness = fitness_values[gen_best_fitness_index]
        best_fitness_per_gen.append(gen_best_fitness)
                    
            # print(f"Iteration {t+1}: Best Fitness = {gen_best_fitness}")
 
    best_index = np.argmax(fitness_values)
    best_fitness = fitness_values[best_index]
    best_solution = target_vector[best_index]

    return best_solution, best_fitness ,best_fitness_per_gen


# ==================================================================
# ITERATE oVER 20 RUNS
# ==================================================================
def solve_square_function():
        num_runs = 20
        results = []
        
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

        # ==========================
        # Plot for THIS instance
        # ==========================
        plt.figure()

        # Plot all runs (light)
        for i, history in enumerate(all_histories):
            plt.plot(history, alpha=0.4, label=f"Run {i+1}")

        # Plot average (bold)
        plt.plot(avg_fitness, linewidth=2, label="Average")

        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title(f"CONVERGENCE PLOT || DE(penalty based)")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        # plt.savefig(f"plots/_instance_{idx}_BCGA_penalty_convergence.png", dpi=300)
        plt.show()

        # Store results
        results.append({
            "histories": all_histories,
            "best_costs": all_best_costs,
            "avg_fitness": avg_fitness
        })


# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__": 
    solve_square_function()
