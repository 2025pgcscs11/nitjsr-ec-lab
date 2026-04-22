import os

def read_gap_file(filename):
    """
    Reads a GAP data file and returns a list of problem instances.
    Each instance is a tuple: (C, R, B)
    """
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

        # Cost matrix
        C = []
        for _ in range(m):
            C.append(data[idx:idx+n])
            idx += n

        # Resource matrix
        R = []
        for _ in range(m):
            R.append(data[idx:idx+n])
            idx += n

        # Capacities
        B = data[idx:idx+m]
        idx += m

        instances.append((C, R, B))

    return instances


def greedy_gap(C, R, B):
    """
    Best-effort greedy GAP solver (Cost Maximization).
    - assignment[j] = agent index OR -1 if unassigned
    - total_cost = sum of costs of assigned jobs only
    """
    m = len(C)
    n = len(C[0])

    remaining_capacity = B.copy()
    assignment = [-1] * n
    total_cost = 0

    for j in range(n):  # job-oriented greedy
        best_agent = -1
        best_cost = float('-inf')

        for i in range(m):
            if remaining_capacity[i] >= R[i][j]:
                if C[i][j] > best_cost:
                    best_cost = C[i][j]
                    best_agent = i

        if best_agent != -1:
            assignment[j] = best_agent
            remaining_capacity[best_agent] -= R[best_agent][j]
            total_cost += C[best_agent][j]
        # else: job remains unassigned

    return assignment, total_cost



def greedy_gap_hard_job_first(C, R, B):
    m = len(C)
    n = len(C[0])

    remaining_capacity = B.copy()
    assignment = [-1] * n
    total_cost = 0

    # Hard jobs first (largest minimum resource)
    job_order = sorted(
        range(n),
        key=lambda j: min(R[i][j] for i in range(m)),
        reverse=True
    )

    for j in job_order:
        best_agent = -1
        best_cost = float('-inf')

        for i in range(m):
            if remaining_capacity[i] >= R[i][j]:
                if C[i][j] > best_cost:
                    best_cost = C[i][j]
                    best_agent = i

        if best_agent != -1:
            assignment[j] = best_agent
            remaining_capacity[best_agent] -= R[best_agent][j]
            total_cost += C[best_agent][j]

    return assignment, total_cost



def greedy_gap_fewest_feasible_agents(C, R, B):
    m = len(C)
    n = len(C[0])

    remaining_capacity = B.copy()
    assignment = [-1] * n
    total_cost = 0

    unassigned_jobs = set(range(n))

    while unassigned_jobs:
        min_agents = float('inf')
        selected_job = None
        feasible_agents = []

        for j in unassigned_jobs:
            feasible = [
                i for i in range(m)
                if remaining_capacity[i] >= R[i][j]
            ]
            if feasible and len(feasible) < min_agents:
                min_agents = len(feasible)
                selected_job = j
                feasible_agents = feasible

        if selected_job is None:
            break

        best_agent = max(
            feasible_agents,
            key=lambda i: C[i][selected_job]
        )

        assignment[selected_job] = best_agent
        remaining_capacity[best_agent] -= R[best_agent][selected_job]
        total_cost += C[best_agent][selected_job]
        unassigned_jobs.remove(selected_job)

    return assignment, total_cost




def greedy_gap_cost_resource_ratio(C, R, B):
    m = len(C)
    n = len(C[0])

    remaining_capacity = B.copy()
    assignment = [-1] * n
    total_cost = 0

    for j in range(n):
        best_agent = -1
        best_ratio = float('-inf')

        for i in range(m):
            if remaining_capacity[i] >= R[i][j] and R[i][j] > 0:
                ratio = C[i][j] / R[i][j]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_agent = i

        if best_agent != -1:
            assignment[j] = best_agent
            remaining_capacity[best_agent] -= R[best_agent][j]
            total_cost += C[best_agent][j]

    return assignment, total_cost




def greedy_gap_agent_oriented(C, R, B):
    m = len(C)
    n = len(C[0])

    remaining_capacity = B.copy()
    assignment = [-1] * n
    total_cost = 0
    unassigned_jobs = set(range(n))

    agents = sorted(range(m), key=lambda i: B[i], reverse=True)

    for i in agents:
        while True:
            feasible_jobs = [
                j for j in unassigned_jobs
                if remaining_capacity[i] >= R[i][j]
            ]

            if not feasible_jobs:
                break

            # pick MOST expensive feasible job
            j = max(feasible_jobs, key=lambda j: C[i][j])

            assignment[j] = i
            remaining_capacity[i] -= R[i][j]
            total_cost += C[i][j]
            unassigned_jobs.remove(j)

    return assignment, total_cost


def assignment_stats(assignment):
    """
    Returns statistics of assignment.
    """
    assigned = sum(1 for a in assignment if a != -1)
    unassigned = len(assignment) - assigned
    return assigned, unassigned


def solve_instance_with_strategy(C, R, B, strategy_name, strategy_func):
    print(f"  Strategy: {strategy_name}")

    assignment, cost = strategy_func(C, R, B)

    print("    Total cost:", cost)
    assigned, unassigned = assignment_stats(assignment)
    print("    Assigned jobs:", assigned)
    print("    Unassigned jobs:", unassigned)
    print()

    return assignment, cost


def solve_gap_file(filename, strategies):
    """
    strategies: list of (strategy_name, strategy_function)
    """
    instances = read_gap_file(filename)
    results = []

    print(f"\n===== Solving file: {filename} =====\n")

    for idx, (C, R, B) in enumerate(instances, start=1):
        print(f"Instance {idx}:")

        instance_results = {}

        for name, func in strategies:
            assignment, cost = solve_instance_with_strategy(
                C, R, B, name, func
            )
            instance_results[name] = (assignment, cost)

        results.append(instance_results)

    return results



def solve_multiple_files(file_list, strategies, base_dir="./gap_dataset"):
    all_results = {}

    for file in file_list:
        file_path = os.path.join(base_dir, file)
        all_results[file] = solve_gap_file(file_path, strategies)

    return all_results


strategies = [
    ("Simple Job-Oriented Greedy", greedy_gap),
    ("Hard-Job-First Greedy", greedy_gap_hard_job_first),
    ("Fewest-Feasible-Agents-First Greedy", greedy_gap_fewest_feasible_agents),
    ("Cost-Per-Resource Ratio Greedy", greedy_gap_cost_resource_ratio),
    ("Agent-Oriented Greedy", greedy_gap_agent_oriented),
]


files = [
    "gap1.txt", "gap2.txt", "gap3.txt","gap4.txt",
    "gap5.txt","gap6.txt","gap7.txt","gap8.txt",
    "gap9.txt","gap10.txt","gap11.txt","gap12.txt"
]


solve_multiple_files(files,strategies)
