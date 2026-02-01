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



def compute_remaining_capacity(R, B, assignment):
    m = len(B)
    remaining = B.copy()

    for j, i in enumerate(assignment):
        if i != -1:
            remaining[i] -= R[i][j]

    return remaining


def total_cost_from_assignment(C, assignment):
    cost = 0
    for j, i in enumerate(assignment):
        if i != -1:
            cost += C[i][j]
    return cost

def coverage_repair_cost_safe(C, R, B, assignment):
    m = len(B)
    n = len(assignment)

    assignment = assignment.copy()
    base_cost = total_cost_from_assignment(C, assignment)

    improved = True

    while improved:
        improved = False
        remaining = compute_remaining_capacity(R, B, assignment)

        unassigned_jobs = [j for j in range(n) if assignment[j] == -1]

        for j_u in unassigned_jobs:
            # ---------- Case 1: Direct assignment ----------
            for i in range(m):
                if remaining[i] >= R[i][j_u]:
                    assignment[j_u] = i
                    new_cost = total_cost_from_assignment(C, assignment)

                    if new_cost >= base_cost:
                        remaining[i] -= R[i][j_u]
                        base_cost = new_cost
                        improved = True
                        break
                    else:
                        assignment[j_u] = -1

            if improved:
                break

            # ---------- Case 2: Relocate one job ----------
            for j_old in range(n):
                i_old = assignment[j_old]
                if i_old == -1:
                    continue

                # temporarily remove j_old
                assignment[j_old] = -1
                remaining[i_old] += R[i_old][j_old]

                for i_new in range(m):
                    if remaining[i_new] >= R[i_new][j_u]:
                        assignment[j_u] = i_new

                        new_cost = total_cost_from_assignment(C, assignment)
                        if new_cost >= base_cost:
                            remaining[i_new] -= R[i_new][j_u]
                            base_cost = new_cost
                            improved = True
                            break
                        else:
                            assignment[j_u] = -1

                if improved:
                    break

                # rollback
                assignment[j_old] = i_old
                remaining[i_old] -= R[i_old][j_old]

            if improved:
                break

    return assignment, base_cost



def assignment_stats(assignment):
    """
    Returns statistics of assignment.
    """
    assigned = sum(1 for a in assignment if a != -1)
    unassigned = len(assignment) - assigned
    return assigned, unassigned

def solve_instance_with_strategy(C, R, B,strategy_name, strategy_func):
    print("After Using Greedy Strategy: ",strategy_name)
    assignment, cost = strategy_func(C, R, B)

    print("    Total cost:", cost)
    assigned, unassigned = assignment_stats(assignment)
    print("    Assigned jobs:", assigned)
    print("    Unassigned jobs:", unassigned)
    print()

    return assignment, cost

def solve_instance_with_repairs(C, R, B,repair_func_name,repair_func,greedy_assignment):
    assignment, cost = repair_func(C, R, B, greedy_assignment)
    assigned, unassigned = assignment_stats(assignment)

    print("    After Using Repair Strategy",repair_func_name)
    print("        Total cost:", cost)
    print("        Assigned jobs:", assigned)
    print("        Unassigned jobs:", unassigned)
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

        for strategy_name, strategy_func in strategies:
            greedy_assignment, greedy_cost = solve_instance_with_strategy(C,R,B,strategy_name,strategy_func)
            instance_results[strategy_name] = {
                "greedy": (greedy_assignment, greedy_cost),
                "repairs": {}
            }
            for repair_name,repair_func in repairs:
                assignment, cost = solve_instance_with_repairs(C, R, B,repair_name,repair_func,greedy_assignment)
                instance_results[strategy_name]["repairs"][repair_name] = (assignment, cost)

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
]

repairs = [
    ("Coverage-First, Cost-Safe Repair", coverage_repair_cost_safe)
]


files = [
    "gap1.txt", "gap2.txt", "gap3.txt","gap4.txt",
    "gap5.txt","gap6.txt","gap7.txt","gap8.txt",
    "gap9.txt","gap10.txt","gap11.txt","gap12.txt"
]

solve_multiple_files(files,strategies)
