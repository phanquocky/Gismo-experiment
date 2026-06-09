"""
ILP-based minimum sensor set finder for (d,l)-disjunct matrices.

Implements the formulation from ilp.md:

    min   sum(z_i)
    s.t.  sum_{i in L} z_i  +  sum_{i in W'(D,L)} z_i  >=  1
                for all D,L with |D|=d, |L|=l, D∩L=empty, D∪L ⊆ {0..n-1}
          z_i in {0, 1}

W'(D, L) = nodes i (excluding nodes in L) whose sensor columns can witness (D, L):
  - M_orig[k][c] = 0  for all k in D   (sensor c is silent for every failing node in D)
  - M_orig[k][c] = 1  for some k in L  (sensor c fires for at least one failing node in L)

The sum_{i in L} z_i term is the x-block improvement from improvement.md:
  if any node i in L is itself a sensor, then its x_i column fires exactly in
  scenario i (which is in L) and is silent in all scenarios in D — perfect witness.
  So the constraint is automatically satisfied whenever any z_i for i in L equals 1.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import pulp

sys.path.insert(0, str(Path(__file__).parent.parent / "baseline" / "tools"))
from network_to_matrix import network_to_matrix


# ── Witness set ───────────────────────────────────────────────────────────────

def _build_dl_witness_set(
    M: List[List[int]],
    D: Tuple[int, ...],
    L: Tuple[int, ...],
    n: int,
) -> set:
    """
    W'(D, L): node indices (excluding nodes already in L) whose sensor columns
    can witness the (D, L) pair.

    A column c qualifies when:
      - M[k][c] = 0  for all k in D   (silent when any D-scenario fails)
      - M[k][c] = 1  for some k in L  (fires when at least one L-scenario fails)
      - node_of(c) not in L           (L nodes are captured by the z_i sum in constraint)
    """
    L_set = set(L)
    witnesses = set()
    for c in range(2 * n):
        node = c if c < n else c - n
        if node in L_set:
            continue
        if all(M[k][c] == 0 for k in D) and any(M[k][c] == 1 for k in L):
            witnesses.add(node)
    return witnesses


# ── ILP solver ────────────────────────────────────────────────────────────────

def ilp_dl_disjunct(
    network_file: str, d: int, l: int
) -> Tuple[List[int], float, float]:
    """
    Find the minimum (d,l)-disjunct sensor set via ILP.

    Uses the conditional constraint form (improvement.md, Alternative 1):
        sum_{i in L} z_i  +  sum_{i in W'(D,L)} z_i  >=  1

    Returns
    -------
    sensor_nodes : list of node IDs in the minimum sensor set
    obj_value    : optimal objective value (= sensor set size)
    elapsed      : wall-clock seconds
    """
    import time
    t0 = time.time()

    nodes, matrix = network_to_matrix(network_file)
    n = len(nodes)
    M = matrix[1:]  # drop the all-zero empty row → shape (n, 2n)

    prob = pulp.LpProblem("min_dl_disjunct_sensor_set", pulp.LpMinimize)
    z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(z)

    n_constraints = 0
    all_scenarios = list(range(n))

    for S in combinations(all_scenarios, d + l):
        for L_tuple in combinations(S, l):
            D_tuple = tuple(k for k in S if k not in set(L_tuple))
            W_prime = _build_dl_witness_set(M, D_tuple, L_tuple, n)

            # sum_{i in L} z_i + sum_{i in W'(D,L)} z_i >= 1
            prob += (
                pulp.lpSum(z[i] for i in L_tuple)
                + pulp.lpSum(z[i] for i in W_prime)
                >= 1
            )
            n_constraints += 1

    print(f"  Nodes: {n}  |  d: {d}  |  l: {l}  |  Constraints added: {n_constraints}")

    import os
    _cplex_path = os.environ.get("CPLEX_PATH")
    if _cplex_path:
        prob.solve(pulp.CPLEX_CMD(msg=0, path=_cplex_path))
    else:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

    elapsed = time.time() - t0
    obj = pulp.value(prob.objective)
    sensor_nodes = [nodes[i] for i in range(n) if pulp.value(z[i]) > 0.5]
    return sensor_nodes, obj, elapsed


# ── Exact verification ────────────────────────────────────────────────────────

def _verify_dl_disjunct(M_prime: List[List[int]], d: int, l: int) -> bool:
    """
    Exact (exhaustive) (d,l)-disjunct check on the transposed submatrix M'.
    Rows = active sensor columns, columns = failure scenarios.

    For each (d+l)-subset S of columns and every partition into D (size d)
    and L (size l), there must exist a row r where:
      - M'[r][k] = 0  for all k in D
      - M'[r][k] = 1  for at least one k in L
    """
    if not M_prime or not M_prime[0]:
        return True
    n_cols = len(M_prime[0])

    for S in combinations(range(n_cols), d + l):
        for L_tuple in combinations(S, l):
            D_tuple = tuple(k for k in S if k not in set(L_tuple))
            witnessed = any(
                all(row[k] == 0 for k in D_tuple)
                and any(row[k] == 1 for k in L_tuple)
                for row in M_prime
            )
            if not witnessed:
                return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    graph_file = Path(__file__).parent / "graph.txt"
    d, l = 2, 2

    print("=" * 60)
    print(f"Graph  : {graph_file}")
    print(f"d      : {d}  |  l : {l}")
    print("=" * 60)

    nodes, matrix = network_to_matrix(str(graph_file))
    n = len(nodes)
    M = matrix[1:]

    print(f"\nNetwork: {n} nodes")
    print("Failure-sensor matrix M_orig (rows=scenarios, cols=x_1..x_n|y_1..y_n):")
    header = "  ".join(f"x{v}" for v in nodes) + "  |  " + "  ".join(f"y{v}" for v in nodes)
    print("  " + header)
    for i, row in enumerate(M):
        print(f"  node {nodes[i]} fails: " + "  ".join(map(str, row)))

    print("\nSolving ILP ...")
    sensor_nodes, obj, elapsed = ilp_dl_disjunct(str(graph_file), d, l)

    print(f"\nOptimal sensor set size : {int(obj)}")
    print(f"Sensor nodes            : {sensor_nodes}")
    print(f"Solve time              : {elapsed:.3f}s")

    # Build M' for verification
    idx = {v: i for i, v in enumerate(nodes)}
    active_indices = [idx[v] for v in sensor_nodes]
    active_cols = sorted([i for i in active_indices] + [n + i for i in active_indices])
    M_prime = [[M[j][c] for c in active_cols] for j in range(n)]
    M_prime_T = [list(row) for row in zip(*M_prime)]  # transpose

    ok = _verify_dl_disjunct(M_prime_T, d, l)
    status = f"PASS — matrix is ({d},{l})-disjunct" if ok else "FAIL"
    print(f"Verification (exact)    : {status}")

    print("\nActive sensor columns in M' (rows=sensors, cols=failure scenarios):")
    col_labels = [f"node{nodes[j]}fails" for j in range(n)]
    row_labels = (
        [f"x_{nodes[i]}" for i in active_indices]
        + [f"y_{nodes[i]}" for i in active_indices]
    )
    row_labels_sorted = [row_labels[active_cols.index(c)] for c in sorted(active_cols)]
    print("  " + "  ".join(f"{lb:>12}" for lb in col_labels))
    for label, row in zip(row_labels_sorted, M_prime_T):
        print(f"  {label:<6}  " + "  ".join(f"{v:>12}" for v in row))


if __name__ == "__main__":
    main()
