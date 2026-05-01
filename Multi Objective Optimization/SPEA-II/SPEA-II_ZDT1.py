import numpy as np
import time
import matplotlib.pyplot as plt

# ==========================================================
# CONSTANT PARAMETERS
# ==========================================================
POP_SIZE = 100
ARCHIVE_SIZE = 100
GENERATIONS  = 200
DIMENSION    = 30
LOWER_BOUND  = 0
UPPER_BOUND  = 1
CROSSOVER_RATE = 0.9
MUTATION_RATE  = 0.1
SBX_DISTRIBUTION_INDEX = 15
POLYNOMIAL_MUTATION_DISTRIBUTION_INDEX = 20


# SPEA2 paper: k = sqrt(|P_t| + |A_t|)
K = int(np.sqrt(POP_SIZE + ARCHIVE_SIZE))


# ==========================================================
# PROBLEM FUNCTIONS (ZDT1)
# ==========================================================
def evaluate_objectives(x):
    """Return (f1, f2) for ZDT1."""
    x = np.asarray(x)
    f1 = x[0]
    g  = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    h  = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return f1, f2


# ==========================================================
# WHETHER a DOMINATES b 
# ==========================================================
def dominates(a, b):
    """Minimisation: a dominates b.  a, b are 1-D numpy arrays."""
    return bool(np.all(a <= b) and np.any(a < b))


# ==========================================================
# FINE‑GRAINED FITNESS ASSIGNMENT (SPEA2)
# ==========================================================
def compute_fitness(pop, k):
    """
    pop : numpy array (N, DIM)
    k   : neighbourhood size for density (same k everywhere — Fix #2)

    Returns obj_vals (N,2), S, R, D, F
    """
    N = len(pop)
    obj_vals = np.array([evaluate_objectives(ind) for ind in pop])

    # ---- Strength S(i) ----
    S = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(N):
            if i != j and dominates(obj_vals[i], obj_vals[j]):
                S[i] += 1

    # ---- Raw fitness R(i) ----
    R = np.zeros(N, dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j and dominates(obj_vals[j], obj_vals[i]):
                R[i] += S[j]

    # ---- Density via k-th nearest neighbour ----
    f_min = obj_vals.min(axis=0)
    f_max = obj_vals.max(axis=0)
    denom = np.where(f_max - f_min == 0, 1e-9, f_max - f_min)
    norm_obj = (obj_vals - f_min) / denom

    D = np.zeros(N)
    for i in range(N):
        # Vectorised distance from individual i to all others
        dists = np.linalg.norm(norm_obj - norm_obj[i], axis=1)  
        dists[i] = np.inf                                         
        dists_sorted = np.sort(dists)
        sigma_k = dists_sorted[min(k - 1, N - 2)]
        D[i] = 1.0 / (sigma_k + 2.0)

    F = R + D
    return obj_vals, S, R, D, F


# ==========================================================
# ARCHIVE UPDATE — returns INDICES
# ==========================================================
def build_archive(pop, obj_vals, fitness, N_bar, k):
    """
    Returns the indices (into `pop`) of the N_bar selected individuals.
    Returning indices rather than copied rows lets the caller retrieve
    fitness values without a second compute_fitness call (Fix #6).
    """
    mask_nd   = fitness < 1.0
    nd_indices = np.where(mask_nd)[0]
    N_nd = len(nd_indices)

    if N_nd == N_bar:
        return nd_indices.copy()

    elif N_nd < N_bar:
        selected = list(nd_indices)
        dominated_idx = np.where(~mask_nd)[0]
        sorted_dom = dominated_idx[np.argsort(fitness[dominated_idx])]
        need = N_bar - len(selected)
        selected.extend(sorted_dom[:need].tolist())
        return np.array(selected)

    else:   # N_nd > N_bar — truncation with k-th nearest neighbour
        nd_obj_list   = list(obj_vals[nd_indices])
        nd_idx_list   = list(nd_indices)

        while len(nd_idx_list) > N_bar:
            n = len(nd_idx_list)
            arr_obj = np.array(nd_obj_list)
            f_min   = arr_obj.min(axis=0)
            f_max   = arr_obj.max(axis=0)
            denom   = np.where(f_max - f_min == 0, 1e-9, f_max - f_min)
            norm_obj = (arr_obj - f_min) / denom

            sigma = np.zeros(n)
            for i in range(n):
                dists = np.linalg.norm(norm_obj - norm_obj[i], axis=1)
                dists[i] = np.inf
                dists_sorted = np.sort(dists)
                sigma[i] = dists_sorted[min(k - 1, n - 2)]

            remove_idx = int(np.argmin(sigma))
            nd_idx_list.pop(remove_idx)
            nd_obj_list.pop(remove_idx)

        return np.array(nd_idx_list)


# ==========================================================
# BINARY TOURNAMENT SELECTION
# ==========================================================
def tournament_selection(archive, archive_fitness):
    """
    FIX #3 — np.random.choice(..., replace=False) guarantees i ≠ j,
    eliminating degenerate self-comparisons.
    """
    M = len(archive)
    selected = []
    for _ in range(POP_SIZE):
        i, j = np.random.choice(M, 2, replace=False)     
        if archive_fitness[i] < archive_fitness[j]:
            winner = archive[i]
        elif archive_fitness[j] < archive_fitness[i]:
            winner = archive[j]
        else:
            winner = archive[np.random.choice([i, j])]
        selected.append(winner)
    return np.array(selected)


# ==========================================================
# SIMULATED BINARY CROSSOVER (SBX)
# ==========================================================
def sbx_crossover(p1, p2):
    if np.random.rand() >= CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    c1, c2 = [], []
    for x1, x2 in zip(p1, p2):
        u = np.random.rand()
        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (SBX_DISTRIBUTION_INDEX + 1))
        else:
            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (SBX_DISTRIBUTION_INDEX + 1))
        c1.append(0.5 * ((1 + beta) * x1 + (1 - beta) * x2))
        c2.append(0.5 * ((1 - beta) * x1 + (1 + beta) * x2))
    return np.clip(c1, LOWER_BOUND, UPPER_BOUND), np.clip(c2, LOWER_BOUND, UPPER_BOUND)


# ==========================================================
# POLYNOMIAL MUTATION
# ==========================================================
def polynomial_mutation(ind):
    if np.random.rand() >= MUTATION_RATE:
        return ind.copy()
    child = ind.copy()
    for i in range(len(child)):
        r = np.random.rand()
        if r < 0.5:
            delta = (2 * r) ** (1.0 / (POLYNOMIAL_MUTATION_DISTRIBUTION_INDEX + 1)) - 1
        else:
            delta = 1 - (2 * (1 - r)) ** (1.0 / (POLYNOMIAL_MUTATION_DISTRIBUTION_INDEX + 1))
        child[i] += delta * (UPPER_BOUND - LOWER_BOUND)
    return np.clip(child, LOWER_BOUND, UPPER_BOUND)


# ==========================================================
# VARIATION (CROSSOVER + MUTATION)
# ==========================================================
def variation(mating_pool):
    """
    FIX #5 — iterates over pairs and then slices to exactly POP_SIZE,
    so an odd-sized mating pool never silently produces POP_SIZE + 1 offspring.
    """
    offspring = []
    for i in range(0, len(mating_pool) - 1, 2):
        p1, p2 = mating_pool[i], mating_pool[i + 1]
        c1, c2 = sbx_crossover(p1, p2)
        offspring.append(polynomial_mutation(c1))
        offspring.append(polynomial_mutation(c2))
    return np.array(offspring[:POP_SIZE])


# ==========================================================
# MAIN SPEA2 LOOP
# ==========================================================
def strength_pareto_evolutionary_algorithm():
    # FIX #4 — A(0) = ∅  (empty archive, as specified in the SPEA2 paper).
    # The first archive A(1) is built from P(0) ∪ A(0) = P(0) below.
    population = np.random.uniform(LOWER_BOUND, UPPER_BOUND, (POP_SIZE, DIMENSION))
    init_obj   = np.array([evaluate_objectives(ind) for ind in population])

    # Pre-loop: fitness of P(0), build first archive A(1)
    obj_vals, _, _, _, F = compute_fitness(population, K)
    arch_idx     = build_archive(population, obj_vals, F, ARCHIVE_SIZE, K)
    archive         = population[arch_idx]
    archive_fitness = F[arch_idx]

    for gen in range(GENERATIONS):

        mating_pool = tournament_selection(archive, archive_fitness)
        offspring   = variation(mating_pool)

        # Combine offspring P(t+1) with current archive A(t)
        combined_pop = np.vstack([offspring, archive])
        combined_obj, _, _, _, combined_fitness = compute_fitness(combined_pop, K)

        # Build A(t+1) and extract its fitness in one step
        new_idx         = build_archive(combined_pop, combined_obj, combined_fitness, ARCHIVE_SIZE, K)
        archive         = combined_pop[new_idx]
        archive_fitness = combined_fitness[new_idx]

    # ---- Final non-dominated solutions ----
    final_obj, _, _, _, final_fit = compute_fitness(archive, K)
    nd_mask    = final_fit < 1.0
    pareto_obj = final_obj[nd_mask]

    return init_obj, pareto_obj


# ==========================================================
# PLOTTING
# ==========================================================
def plot_results(initial_obj, pareto_obj):
    true_f1 = np.linspace(0, 1, 200)
    true_f2 = 1 - np.sqrt(true_f1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(initial_obj[:, 0], initial_obj[:, 1], s=15, alpha=0.6,
                label='Initial population')
    plt.plot(true_f1, true_f2, 'r-', linewidth=2, label='True Pareto front')
    plt.xlabel('f1'); plt.ylabel('f2')
    plt.title('Initial Population')
    plt.grid(alpha=0.3); plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(pareto_obj[:, 0], pareto_obj[:, 1], s=20, c='green', alpha=0.7,
                label='SPEA2 Archive (non‑dominated)')
    plt.plot(true_f1, true_f2, 'r-', linewidth=2, label='True Pareto front')
    plt.xlabel('f1'); plt.ylabel('f2')
    plt.title('Final Non‑dominated Solutions')
    plt.grid(alpha=0.3); plt.legend()

    plt.suptitle('SPEA2 Performance on ZDT1', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ==========================================================
# RUN EXPERIMENT
# ==========================================================
if __name__ == "__main__":
    num_runs = 1
    for run in range(num_runs):
        start = time.perf_counter()
        init_obj, final_pareto = strength_pareto_evolutionary_algorithm()
        elapsed = time.perf_counter() - start
        plot_results(init_obj, final_pareto)
        print(f"Run {run + 1}: Time = {elapsed:.4f} sec")