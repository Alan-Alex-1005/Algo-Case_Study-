import itertools
import math
import time
import matplotlib.pyplot as plt

# -------------------------
# Helper functions
# -------------------------

def distance(a, b):
    """Euclidean distance between points a and b"""
    return math.dist(a, b)


def total_route_cost(route, nodes):
    """Compute total cost of a route"""
    cost = 0
    for i in range(len(route) - 1):
        cost += distance(nodes[route[i]], nodes[route[i+1]])
    return cost


# -------------------------
# 1. Greedy Algorithm (Nearest Neighbor)
# -------------------------

def greedy_tsp(nodes):
    start = 0
    unvisited = set(range(1, len(nodes)))
    route = [start]

    while unvisited:
        last = route[-1]
        next_node = min(unvisited, key=lambda x: distance(nodes[last], nodes[x]))
        route.append(next_node)
        unvisited.remove(next_node)

    route.append(start)  # return to depot
    return route


# -------------------------
# 2. Held-Karp DP (Exact TSP)
# -------------------------

def held_karp(nodes):
    n = len(nodes)
    dp = {}

    # base: start from 0 → i
    for i in range(1, n):
        dp[(1 << i, i)] = (distance(nodes[0], nodes[i]), [0, i])

    # subsets of increasing size
    for mask in range(1 << n):
        for last in range(n):
            if mask & (1 << last) == 0:
                continue
            if (mask, last) not in dp:
                continue

            cost, path = dp[(mask, last)]

            for nxt in range(1, n):
                if mask & (1 << nxt):
                    continue

                new_mask = mask | (1 << nxt)
                new_cost = cost + distance(nodes[last], nodes[nxt])
                new_path = path + [nxt]

                if (new_mask, nxt) not in dp or dp[(new_mask, nxt)][0] > new_cost:
                    dp[(new_mask, nxt)] = (new_cost, new_path)

    # complete the path by returning to 0
    full_mask = (1 << n) - 1
    best_cost = float("inf")
    best_path = None

    for last in range(1, n):
        if (full_mask, last) in dp:
            cost, path = dp[(full_mask, last)]
            cost += distance(nodes[last], nodes[0])
            if cost < best_cost:
                best_cost = cost
                best_path = path + [0]

    return best_path


# -------------------------
# Warehouse coordinates
# -------------------------

nodes = [
    (0, 0),    # depot
    (5, 2),    # P1
    (6, 5),    # P2
    (12, 10),  # P3
    (14, 3),   # P4
    (9, 1),    # P5
    (4, 8),    # P6
]


# -------------------------
# Run both algorithms
# -------------------------

start = time.time()
greedy_route = greedy_tsp(nodes)
greedy_cost = total_route_cost(greedy_route, nodes)
greedy_time = time.time() - start

start = time.time()
dp_route = held_karp(nodes)
dp_cost = total_route_cost(dp_route, nodes)
dp_time = time.time() - start


# -------------------------
# Print results
# -------------------------

print("===== RESULTS =====")
print("Greedy Route:", greedy_route)
print("Greedy Cost:", round(greedy_cost, 2))
print("Greedy Time:", round(greedy_time, 5), "sec\n")

print("DP Route:", dp_route)
print("DP Cost:", round(dp_cost, 2))
print("DP Time:", round(dp_time, 5), "sec")


# -------------------------
# Plot
# -------------------------

plt.figure(figsize=(10, 7))
x = [p[0] for p in nodes]
y = [p[1] for p in nodes]

plt.scatter(x, y, c='red', s=90)

for i, (xx, yy) in enumerate(nodes):
    plt.text(xx+0.2, yy+0.2, f"{i}", fontsize=12)

# Plot Greedy
gx = [nodes[i][0] for i in greedy_route]
gy = [nodes[i][1] for i in greedy_route]
plt.plot(gx, gy, label="Greedy Route")

# Plot DP
dx = [nodes[i][0] for i in dp_route]
dy = [nodes[i][1] for i in dp_route]
plt.plot(dx, dy, label="Dynamic Programming Route", linestyle='--')

plt.title("Warehouse Route Optimization (Greedy vs DP)")
plt.legend()
plt.grid(True)
plt.show()
