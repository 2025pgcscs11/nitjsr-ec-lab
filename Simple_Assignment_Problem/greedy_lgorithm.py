# It gives suboptimal assignments

def greedy_assignment(cost):
    n = len(cost)

    row_used = [False] * n
    col_used = [False] * n

    # Hungarian-style output
    assignment = [-1] * n

    for _ in range(n):
        min_cost = float('inf')
        best_i = best_j = -1

        for i in range(n):
            if row_used[i]:
                continue
            for j in range(n):
                if not col_used[j] and cost[i][j] < min_cost:
                    min_cost = cost[i][j]
                    best_i, best_j = i, j

        assignment[best_i] = best_j
        row_used[best_i] = True
        col_used[best_j] = True

    return assignment



cost_matrix = [
    [41, 72, 39, 52, 25],
    [22, 29, 49, 65, 81],
    [27, 39, 60, 51, 40],
    [45, 50, 48, 52, 37],
    [29, 40, 45, 26, 30]
]


result = greedy_assignment(cost_matrix)

total_cost = 0
print("Greedy Assignment:")
for i, j in enumerate(result):
    print(f"Agent {i+1} -> Job {j+1} (Cost = {cost_matrix[i][j]})")
    total_cost += cost_matrix[i][j]

print("Minimum Total Cost =", total_cost)

