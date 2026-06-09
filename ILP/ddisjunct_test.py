"""
ILP-based sensor set finder — NON-TRANSPOSED d-disjunct variant.

Contrast with ddisjunct.py:
  ddisjunct.py     : checks columns of M_sub.T  (rows=sensors,   cols=scenarios)
  ddisjunct_test.py: checks columns of M_sub    (rows=scenarios,  cols=sensors)

In the non-transposed case the roles of rows and columns swap:
  - Items to distinguish : sensor columns  (x_i, y_i)
  - Witnesses            : failure-scenario rows  (ALL n rows are always present — FIXED)

Because witnesses are fixed (not selected by z), whether column c1 has a witness
against c2 is a PRECOMPUTED boolean from M_orig. This changes the ILP structure:

  Transposed    → "at least one witness sensor must be ACTIVE"   (>= 1 constraint)
  Non-transposed → "conflicting column pairs cannot BOTH be active" (<= 1 constraint)

Same-node pairs (x_i, y_i) are always in conflict (x_i fires only in scenario i,
but y_i also fires in scenario i since i ∈ N[i]).  They are treated as a UNIT:
only CROSS-NODE column pairs are checked.

A coverage (domination) constraint is added so the trivial z=0 solution is excluded:
every failure scenario r must be detected by at least one active sensor.

ILP:
    min   sum(z_i)
    s.t.  z_i + z_j  <= 1              for each conflicting cross-node pair (i,j)
          sum_{i in N[r]} z_i  >= 1    for each failure scenario r  [domination]
          z_i in {0, 1}
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import List, Set, Tuple

import pulp

sys.path.insert(0, str(Path(__file__).parent.parent / "baseline" / "tools"))
from network_to_matrix import network_to_matrix


# ── Core helpers ──────────────────────────────────────────────────────────────

def _node_of(c: int, n: int) -> int:
    return c if c < n else c - n


def _has_witness_row(M: List[List[int]], c1: int, c2: int, n: int) -> bool:
    """
    Return True if there exists a row r where M[r][c1]=1 and M[r][c2]=0.
    (Column c1 is NOT covered by column c2.)
    """
    return any(M[r][c1] == 1 and M[r][c2] == 0 for r in range(n))


def _cross_node_conflicts(M: List[List[int]], n: int, d: int) -> Set[Tuple[int, int]]:
    """
    For the non-transposed d-disjunct check on column pairs, compute which pairs
    of NODES (i, j) cannot both be sensors because some column from node i is
    covered by some column from node j (or vice versa).

    A node pair (i, j) conflicts if ANY (c1 from i, d-subset S from j and others)
    has no witness row. For d=1 this reduces to: any single (c1, c2) pair with no
    witness row (in either direction) causes a conflict between their nodes.

    Returns a set of (i, j) pairs with i < j.
    """
    conflicts: Set[Tuple[int, int]] = set()

    all_cols = list(range(2 * n))
    for c1 in all_cols:
        n1 = _node_of(c1, n)
        for c2 in all_cols:
            n2 = _node_of(c2, n)
            if n1 == n2:
                continue  # same-node pair treated as a unit — skip
            # c1 covered by c2 means no witness row → conflict between nodes n1 and n2
            if not _has_witness_row(M, c1, c2, n):
                key = (min(n1, n2), max(n1, n2))
                conflicts.add(key)

    return conflicts


def _domination_sets(M: List[List[int]], n: int) -> List[Set[int]]:
    """
    For each failure scenario r (row of M), return the set of sensor nodes
    whose columns fire in scenario r.  At least one of these must be active
    (domination constraint).

    Sensor node i "covers" scenario r if M[r][x_i]=1 OR M[r][y_i]=1,
    which simplifies to: i ∈ N[r] (the closed neighbourhood of r).
    """
    dom = []
    for r in range(n):
        covering = {_node_of(c, n) for c in range(2 * n) if M[r][c] == 1}
        dom.append(covering)
    return dom


# ── ILP solver ────────────────────────────────────────────────────────────────

def ilp_d_disjunct_nontransposed(
    network_file: str, d: int
) -> Tuple[List[int], float, float]:
    """
    Find the minimum sensor set such that the NON-TRANSPOSED submatrix
    (rows=failure scenarios, cols=active sensor columns) is d-disjunct.

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

    conflicts = _cross_node_conflicts(M, n, d)
    dom_sets  = _domination_sets(M, n)

    prob = pulp.LpProblem("min_nontransposed_d_disjunct", pulp.LpMinimize)
    z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(z)

    # Conflict constraints: conflicting node pairs cannot both be sensors
    n_conflict = 0
    for (i, j) in conflicts:
        prob += z[i] + z[j] <= 1
        n_conflict += 1

    # Domination constraints: every failure scenario must be detectable
    n_dom = 0
    for r, covering in enumerate(dom_sets):
        prob += pulp.lpSum(z[i] for i in covering) >= 1
        n_dom += 1

    print(f"  Nodes: {n}  |  d: {d}  |  Conflict constraints: {n_conflict}"
          f"  |  Domination constraints: {n_dom}")

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    elapsed = time.time() - t0
    obj = pulp.value(prob.objective)
    if obj is None:
        return [], None, elapsed

    sensor_nodes = [nodes[i] for i in range(n) if pulp.value(z[i]) > 0.5]
    return sensor_nodes, obj, elapsed


# ── Exact verification ────────────────────────────────────────────────────────

def _verify_d_disjunct_nontransposed(M_sub: List[List[int]], d: int) -> bool:
    """
    Exact d-disjunct check on the NON-TRANSPOSED submatrix M_sub.
    Rows = failure scenarios, columns = active sensor columns.

    For each column c and each d-subset S of other columns, there must
    exist a row r where M_sub[r][c]=1 and M_sub[r][ck]=0 for all ck in S.
    """
    if not M_sub or not M_sub[0]:
        return True
    n_rows = len(M_sub)
    n_cols = len(M_sub[0])
    for c in range(n_cols):
        other = [k for k in range(n_cols) if k != c]
        for S in combinations(other, d):
            witnessed = any(
                M_sub[r][c] == 1 and all(M_sub[r][s] == 0 for s in S)
                for r in range(n_rows)
            )
            if not witnessed:
                return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    graph_file = Path(__file__).parent / "graph.txt"
    d = 1

    print("=" * 60)
    print(f"Graph  : {graph_file}")
    print(f"d      : {d}  (non-transposed submatrix)")
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

    # Show precomputed conflicts
    conflicts = _cross_node_conflicts(M, n, d)
    print(f"\nCross-node column conflicts (pairs that cannot both be sensors): {len(conflicts)}")
    for (i, j) in sorted(conflicts):
        print(f"  node {nodes[i]} ↔ node {nodes[j]}")

    print("\nSolving ILP (non-transposed) ...")
    sensor_nodes, obj, elapsed = ilp_d_disjunct_nontransposed(str(graph_file), d)

    if obj is None:
        print("\nINFEASIBLE — no sensor set satisfies the non-transposed d-disjunct condition")
        return

    print(f"\nOptimal sensor set size : {int(obj)}")
    print(f"Sensor nodes            : {sensor_nodes}")
    print(f"Solve time              : {elapsed:.3f}s")

    # Build non-transposed M_sub for verification
    idx = {v: i for i, v in enumerate(nodes)}
    active_indices = [idx[v] for v in sensor_nodes]
    active_cols = sorted([i for i in active_indices] + [n + i for i in active_indices])
    M_sub = [[M[r][c] for c in active_cols] for r in range(n)]  # NOT transposed

    ok = _verify_d_disjunct_nontransposed(M_sub, d)
    print(f"Verification (exact)    : {'PASS — non-transposed matrix is {}-disjunct'.format(d) if ok else 'FAIL'}")

    # Show the non-transposed submatrix
    col_labels = []
    for c in active_cols:
        node_id = nodes[c if c < n else c - n]
        col_labels.append(f"{'x' if c < n else 'y'}_{node_id}")

    print(f"\nM_sub (NON-transposed: rows=scenarios, cols=active sensors):")
    print("  " + "  ".join(f"{lb:>6}" for lb in col_labels))
    for i, row in enumerate(M_sub):
        print(f"  node {nodes[i]} fails:  " + "  ".join(f"{v:>6}" for v in row))

    # Compare with ddisjunct.py result
    print(f"\n{'─'*60}")
    print(f"COMPARISON")
    print(f"  ddisjunct.py     (transposed check) : optimal size = 3, sensors = [1, 3, 5]")
    print(f"  ddisjunct_test.py (non-transposed)  : optimal size = {int(obj)}, sensors = {sensor_nodes}")


if __name__ == "__main__":
    main()
