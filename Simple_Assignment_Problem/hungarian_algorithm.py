# It gives optimal assignments

import sys

INF = sys.maxsize

def balance_cost_matrix(cost, dummy_cost=0):
    rows = len(cost)
    cols = len(cost[0])
    size = max(rows, cols)

    balanced = []
    for i in range(rows):
        balanced.append(cost[i] + [dummy_cost] * (size - cols))

    for _ in range(size - rows):
        balanced.append([dummy_cost] * size)

    return balanced, rows, cols


def hungarian(cost):
    cost, orig_rows, orig_cols = balance_cost_matrix(cost)

    N = len(cost)

    # Copy matrix
    mat = [row[:] for row in cost]

    star = [[False]*N for _ in range(N)]
    prime = [[False]*N for _ in range(N)]
    rowCover = [False]*N
    colCover = [False]*N

    # Step 1: Row reduction
    for i in range(N):
        row_min = min(mat[i])
        for j in range(N):
            mat[i][j] -= row_min

    # Step 2: Column reduction
    for j in range(N):
        col_min = min(mat[i][j] for i in range(N))
        for i in range(N):
            mat[i][j] -= col_min

    # Step 3: Star zeros
    for i in range(N):
        for j in range(N):
            if mat[i][j] == 0 and not rowCover[i] and not colCover[j]:
                star[i][j] = True
                rowCover[i] = True
                colCover[j] = True

    rowCover[:] = [False]*N
    colCover[:] = [False]*N

    def cover_columns():
        count = 0
        for j in range(N):
            for i in range(N):
                if star[i][j]:
                    colCover[j] = True
                    count += 1
                    break
        return count

    def find_uncovered_zero():
        for i in range(N):
            if rowCover[i]:
                continue
            for j in range(N):
                if not colCover[j] and mat[i][j] == 0:
                    return i, j
        return None

    def find_star_in_row(r):
        for j in range(N):
            if star[r][j]:
                return j
        return -1

    def find_star_in_col(c):
        for i in range(N):
            if star[i][c]:
                return i
        return -1

    def find_prime_in_row(r):
        for j in range(N):
            if prime[r][j]:
                return j
        return -1

    def clear_primes():
        for i in range(N):
            for j in range(N):
                prime[i][j] = False

    # Main loop
    while cover_columns() < N:
        while True:
            pos = find_uncovered_zero()
            if pos is None:
                # Step 6: Adjust matrix
                min_val = INF
                for i in range(N):
                    if not rowCover[i]:
                        for j in range(N):
                            if not colCover[j]:
                                min_val = min(min_val, mat[i][j])

                for i in range(N):
                    for j in range(N):
                        if rowCover[i]:
                            mat[i][j] += min_val
                        if not colCover[j]:
                            mat[i][j] -= min_val
            else:
                r, c = pos
                prime[r][c] = True
                sc = find_star_in_row(r)

                if sc == -1:
                    # Step 5: Augment path
                    path = [(r, c)]
                    while True:
                        sr = find_star_in_col(path[-1][1])
                        if sr == -1:
                            break
                        path.append((sr, path[-1][1]))
                        pc = find_prime_in_row(path[-1][0])
                        path.append((path[-1][0], pc))

                    for (x, y) in path:
                        star[x][y] = not star[x][y]

                    rowCover[:] = [False]*N
                    colCover[:] = [False]*N
                    clear_primes()
                    break
                else:
                    rowCover[r] = True
                    colCover[sc] = False

    # Extract assignment
    assignment = [-1]*N
    for i in range(N):
        for j in range(N):
            if star[i][j]:
                assignment[i] = j

    return assignment


# -------------------- Example --------------------

cost_matrix = [
    [41, 72, 39, 52, 25],
    [22, 29, 49, 65, 81],
    [27, 39, 60, 51, 40],
    [45, 50, 48, 52, 37],
    [29, 40, 45, 26, 30]
]


result = hungarian(cost_matrix)

total_cost = 0
print("Optimal Assignment:")
for i, j in enumerate(result):
    print(f"Agent {i+1} -> Job {j+1} (Cost = {cost_matrix[i][j]})")
    total_cost += cost_matrix[i][j]

print("Minimum Total Cost =", total_cost)
