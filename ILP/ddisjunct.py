"""
ILP-based minimum sensor set finder for d-disjunct matrices.

Implements the formulation from ilp.md (with the x-block conditional constraint
improvement from improvement.md):

    min   sum(z_i)
    s.t.  z_j + sum_{i in W'(j,S)} z_i >= 1    for all j, all d-subsets S of V/{j}
          z_i in {0, 1}

W'(j, S) = columns c where M_orig[j][c]=1  AND  M_orig[k][c]=0 for all k in S,
           mapped to their node index, excluding j itself (since z_j already
           covers the x-block contribution of node j).
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import pulp

sys.path.insert(0, str(Path(__file__).parent.parent / "baseline" / "tools"))
from network_to_matrix import network_to_matrix


def _build_witness_set(M: List[List[int]], j: int, S: Tuple[int, ...], n: int) -> set:
    """
    W'(j, S): node indices (excluding j) whose sensor columns can witness
    scenario j against every scenario in S.

    A column c qualifies when:
      - M[j][c] = 1  (sensor c fires when node j fails)
      - M[k][c] = 0  for all k in S  (sensor c is silent for every scenario in S)
      - node_of(c) != j  (j itself is already captured by z_j in the constraint)
    """
    witnesses = set()
    for c in range(2 * n):
        node = c if c < n else c - n
        if node == j:
            continue
        if M[j][c] == 1 and all(M[k][c] == 0 for k in S):
            witnesses.add(node)
    return witnesses


def ilp_d_disjunct(network_file: str, d: int) -> Tuple[List[int], float]:
    """
    Find the minimum d-disjunct sensor set via ILP.

    Uses the conditional constraint form (improvement.md, Alternative 1):
        z_j + sum_{i in W'(j,S)} z_i >= 1

    Returns
    -------
    sensor_nodes : list of node IDs in the minimum sensor set
    obj_value    : optimal objective value (= sensor set size)
    """
    import time
    t0 = time.time()

    nodes, matrix = network_to_matrix(network_file)
    n = len(nodes)
    M = matrix[1:]  # drop the all-zero empty row → shape (n, 2n)

    prob = pulp.LpProblem("min_d_disjunct_sensor_set", pulp.LpMinimize)
    z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(z)

    n_constraints = 0
    for j in range(n):
        other = [k for k in range(n) if k != j]
        for S in combinations(other, d):
            W_prime = _build_witness_set(M, j, S, n)
            # z_j + sum_{i in W'(j,S)} z_i >= 1
            prob += z[j] + pulp.lpSum(z[i] for i in W_prime) >= 1
            n_constraints += 1

    print(f"  Nodes: {n}  |  d: {d}  |  Constraints added: {n_constraints}")

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    elapsed = time.time() - t0
    obj = pulp.value(prob.objective)
    sensor_nodes = [nodes[i] for i in range(n) if pulp.value(z[i]) > 0.5]
    return sensor_nodes, obj, elapsed


def _verify_d_disjunct(M_prime: List[List[int]], d: int) -> bool:
    """
    Exact (exhaustive) d-disjunct check on the transposed submatrix M'.
    Rows = active sensor columns, columns = failure scenarios.
    """
    n_cols = len(M_prime[0]) if M_prime else 0
    for j in range(n_cols):
        other = [k for k in range(n_cols) if k != j]
        for S in combinations(other, d):
            # Need a row r: M'[r][j]=1 and M'[r][s]=0 for all s in S
            witnessed = any(
                row[j] == 1 and all(row[s] == 0 for s in S)
                for row in M_prime
            )
            if not witnessed:
                return False
    return True


def main():
    graph_file = Path(__file__).parent / "graph.txt"
    d = 1

    print("=" * 60)
    print(f"Graph  : {graph_file}")
    print(f"d      : {d}")
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
    sensor_nodes, obj, elapsed = ilp_d_disjunct(str(graph_file), d)

    print(f"\nOptimal sensor set size : {int(obj)}")
    print(f"Sensor nodes            : {sensor_nodes}")
    print(f"Solve time              : {elapsed:.3f}s")

    # Build M' for verification
    idx = {v: i for i, v in enumerate(nodes)}
    active_indices = [idx[v] for v in sensor_nodes]
    active_cols = [i for i in active_indices] + [n + i for i in active_indices]
    active_cols.sort()
    M_prime = [[M[j][c] for c in active_cols] for j in range(n)]
    M_prime_T = [list(row) for row in zip(*M_prime)]  # transpose

    ok = _verify_d_disjunct(M_prime_T, d)
    print(f"Verification (exact)    : {'PASS — matrix is {}-disjunct'.format(d) if ok else 'FAIL'}")

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
