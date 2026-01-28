with open("data.txt", "r") as f:
    data = list(map(int, f.read().split()))

# Pointer index
idx = 0

# Number of agents and jobs
n_agents = data[idx]
n_jobs = data[idx + 1]
idx += 2

# Cost matrix (n_agents × n_jobs)
cost = []
for _ in range(n_agents):
    cost.append(data[idx:idx + n_jobs])
    idx += n_jobs

# Resource required matrix (n_agents × n_jobs)
resource_required = []
for _ in range(n_agents):
    resource_required.append(data[idx:idx + n_jobs])
    idx += n_jobs

# Resource available per agent (n_agents)
resource_available = data[idx:idx + n_agents]

# -------- Verify --------
# print("Agents:", n_agents)
# print("Jobs:", n_jobs)

# print("\nCost matrix:")
# for row in cost:
#     print(row)

# # print("\nResource required matrix:")
# for row in resource_required:
#     print(row)

# print("\nResource available:")
# print(resource_available)


assignment = []
remaining_resource = resource_available.copy()
total_cost = 0

# print("Greedy Assignment (ignoring infeasible agents):")

for j in range(n_jobs):
    best_agent = None
    best_cost = float('inf')

    for i in range(n_agents):
        # Ignore agent i if it cannot do job j due to resource constraint
        if remaining_resource[i] < resource_required[i][j]:
            continue

        # Only feasible agents reach here
        if cost[i][j] < best_cost:
            best_cost = cost[i][j]
            best_agent = i

    # If no feasible agent exists, skip this job (or report)
    if best_agent is None:
        # print(f"Job {j} skipped (no feasible agent)")
        continue

    # Assign job j
    assignment.append((best_agent, j, best_cost))
    remaining_resource[best_agent] -= resource_required[best_agent][j]
    total_cost += best_cost

    # print(f"Agent {best_agent} -> Job {j} | Cost = {best_cost}")

print("\nTotal Cost:", total_cost)
# print("Remaining Resources:", remaining_resource)
