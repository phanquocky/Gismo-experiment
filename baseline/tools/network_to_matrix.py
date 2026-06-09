"""
Task 2: Convert a network file to the (d,l)-disjunctiveness matrix format.

Matrix layout (n+1 rows x 2n columns):
  Row 0           : empty state — all zeros
  Row i+1 (node v): x_v = 1, y_u = 1 for every u in closed neighbourhood N[v]

Supported file formats:
  .mtx  — MatrixMarket coordinate pattern (lines starting with '%' are comments;
           first non-comment line is "<rows> <cols> <nnz>"; remaining lines are
           "<u> <v>" edge pairs)
  .txt  — plain edge list, one "<u> <v>" per line; lines starting with '#' or '%'
           are ignored
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ── Parsing ──────────────────────────────────────────────────────────────────

def _parse_mtx(path: Path) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    header_consumed = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if not header_consumed:
                header_consumed = True  # skip "<rows> <cols> <nnz>" line
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    edges.append((u, v))
    return edges


def _parse_edgelist(path: Path) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in ("#", "%"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    if u != v:
                        edges.append((u, v))
                except ValueError:
                    continue
    return edges


def _build_adjacency(edges: List[Tuple[int, int]]) -> Tuple[List[int], Dict[int, Set[int]]]:
    adj: Dict[int, Set[int]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    nodes = sorted(adj.keys())
    return nodes, adj


# ── Public API ────────────────────────────────────────────────────────────────

def parse_network(file_path: str) -> Tuple[List[int], Dict[int, Set[int]]]:
    """Return (sorted_node_ids, adjacency_dict) from a network file."""
    path = Path(file_path)
    if path.suffix == ".mtx":
        edges = _parse_mtx(path)
    else:
        edges = _parse_edgelist(path)
    return _build_adjacency(edges)


def network_to_matrix(file_path: str) -> Tuple[List[int], List[List[int]]]:
    """
    Convert a network file to the failure-sensor matrix.

    Returns
    -------
    nodes : List[int]
        Sorted node IDs — column order for both x and y blocks.
    matrix : List[List[int]]
        (n+1) rows × 2n columns.
        Row 0   : empty state (all zeros).
        Row i+1 : node nodes[i] fails.
                  Columns 0..n-1  (x block) — x_j = 1 iff j == nodes[i].
                  Columns n..2n-1 (y block) — y_j = 1 iff nodes[j] ∈ N[nodes[i]].
    """
    nodes, adj = parse_network(file_path)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    matrix: List[List[int]] = [[0] * (2 * n)]  # row 0: empty state

    for v in nodes:
        row = [0] * (2 * n)
        row[idx[v]] = 1                          # x block: failing node
        for u in adj[v] | {v}:                   # y block: closed neighbourhood
            row[n + idx[u]] = 1
        matrix.append(row)

    return nodes, matrix


# ── Pretty-print helper ───────────────────────────────────────────────────────

def print_matrix(nodes: List[int], matrix: List[List[int]], max_nodes: int = 20) -> None:
    n = len(nodes)
    if n > max_nodes:
        print(f"(graph has {n} nodes — showing first {max_nodes} columns of each block)")
        display_nodes = nodes[:max_nodes]
    else:
        display_nodes = nodes

    dn = len(display_nodes)
    header = (
        f"{'Scenario':<20} "
        + "  ".join(f"x{v}" for v in display_nodes)
        + "  |  "
        + "  ".join(f"y{v}" for v in display_nodes)
    )
    print(header)
    print("-" * len(header))

    row_labels = ["empty"] + [f"node {v} fails" for v in nodes]
    for label, row in zip(row_labels, matrix):
        x_part = "  ".join(str(row[idx]) for idx, v in enumerate(nodes) if v in set(display_nodes))
        y_part = "  ".join(str(row[n + idx]) for idx, v in enumerate(nodes) if v in set(display_nodes))
        print(f"{label:<20} {x_part}  |  {y_part}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a network file to failure-sensor matrix.")
    parser.add_argument("--network_file", help="Path to .mtx or .txt network file")
    parser.add_argument("--out", help="Save matrix as CSV to this path (optional)")
    parser.add_argument("--print", dest="do_print", action="store_true",
                        help="Print the matrix to stdout")
    parser.add_argument("--max-display", type=int, default=20,
                        help="Max nodes to display when printing (default: 20)")
    args = parser.parse_args()

    nodes, matrix = network_to_matrix(args.network_file)
    n = len(nodes)
    print(f"Nodes: {n}  |  Matrix shape: {len(matrix)} rows × {2*n} columns")

    # print out the matrix
    for row in matrix:
        print(" ".join(map(str, row)))
        


    if args.do_print:
        print_matrix(nodes, matrix, max_nodes=args.max_display)

    if args.out:
        out_path = Path(args.out)
        header = (
            ",".join(f"x_{v}" for v in nodes)
            + ","
            + ",".join(f"y_{v}" for v in nodes)
        )
        with open(out_path, "w") as f:
            f.write(header + "\n")
            for row in matrix:
                f.write(",".join(map(str, row)) + "\n")
        print(f"Matrix saved to: {out_path}")
