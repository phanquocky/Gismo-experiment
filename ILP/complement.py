"""
Run (d,l)-disjunct ILP experiments using y_i and its complement y_i' (x_i columns dropped).

Same experiment setup as run_experiment.py (datasets, (d,l) pairs, timeout, RAM
limit, constraint limit, subprocess isolation) — the only change is the ILP
formulation itself: the x_i failing-node columns are removed from the witness
matrix. In their place, each sensor i now contributes two witness columns,
mirroring the original x_i/y_i pair structure but built entirely from y:

  - y_i  : M[k][i]  = 1 iff node i is in the closed neighbourhood of k
  - y_i' : M[k][i]' = 1 - M[k][i]   (the complement of y_i)

So the witness matrix has 2n columns total (y block + y' block), and each
column now has its own independent binary variable: z_i controls node i's
y_i column, and z_{n+i} controls node i's y_i' column — the two columns of
a node are selected separately, not merged behind one variable.

    sum_{c in W(D,L)} z_c  >=  1
        for all D,L with |D|=d, |L|=l, D∩L=empty, D∪L ⊆ {0..n-1}

W(D,L) = column indices c (0..2n-1) that can witness (D, L):
  - M[k][c] = 0  for all k in D   (sensor's column is silent for every failing node in D)
  - M[k][c] = 1  for some k in L  (sensor's column fires for at least one failing node in L)

Results saved to Using_YOnlyDisjunct_result.txt.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import shutil
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import pulp

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "baseline" / "tools"))

from network_to_matrix import network_to_matrix, network_to_matrix_y_only  # noqa: E402

DATASETS_DIR = _HERE.parent / "datasets"
RESULT_FILE  = _HERE / "Using_YOnlyDisjunct_result.txt"

TIMEOUT_SEC             = 8 * 3600
RAM_LIMIT_PER_WORKER_GB = 32
CONSTRAINT_LIMIT        = 500_000_000

_cplex_bin = shutil.which("cplex")
if _cplex_bin:
    os.environ["CPLEX_PATH"] = _cplex_bin
    print(f"Solver        : CPLEX  ({_cplex_bin})")
else:
    print("Solver        : CBC  (cplex not found on PATH)")

_D_VALUES = [1, 2, 3, 4, 6, 8, 10, 12, 16]
DL_PAIRS  = [(d, d) for d in _D_VALUES]


def _discover_graphs() -> list[tuple[str, str]]:
    result = []
    for path in sorted(DATASETS_DIR.iterdir()):
        if path.suffix in (".txt", ".mtx", ".edges") and "report" not in path.name:
            result.append((path.stem, path.name))
    return result


# ── ILP solver (y_i and y_i' columns) ──────────────────────────────────────────

def _build_dl_witness_set(M: List[List[int]], D: Tuple[int, ...], L: Tuple[int, ...], n: int) -> set:
    """
    W(D, L): column indices c (0..2n-1) that can witness (D, L).

    M has 2n columns: columns 0..n-1 are y, columns n..2n-1 are y' (complement).
    Column c qualifies when:
      - M[k][c] = 0  for all k in D   (silent when any D-scenario fails)
      - M[k][c] = 1  for some k in L  (fires when at least one L-scenario fails)
    """
    witnesses = set()
    for c in range(2 * n):
        if all(M[k][c] == 0 for k in D) and any(M[k][c] == 1 for k in L):
            witnesses.add(c)
    return witnesses


def ilp_dl_disjunct_y_only(network_file: str, d: int, l: int) -> Tuple[List[int], float, float, str]:
    """
    Find the minimum (d,l)-disjunct sensor set via ILP, using y_i and its
    complement y_i' in place of the x_i/y_i pair.

    Some (d,l) pairs may still have no feasible sensor set (neither a node's
    y_i nor y_i' column can witness every (D,L) split) — in that case the
    solver status will not be "Optimal" and no sensor set is returned.
    """
    t0 = time.time()

    nodes, y_matrix = network_to_matrix_y_only(network_file)
    n = len(nodes)
    Y = y_matrix[1:]  # drop the all-zero empty row -> shape (n, n)
    M = [row + [1 - v for v in row] for row in Y]  # shape (n, 2n): y block + y' block

    prob = pulp.LpProblem("min_dl_disjunct_sensor_set_y_only", pulp.LpMinimize)
    # z_i controls node i's y_i column (i in 0..n-1);
    # z_{n+i} controls node i's y_i' column, selected independently of z_i.
    z = [pulp.LpVariable(f"z_{c}", cat="Binary") for c in range(2 * n)]

    prob += pulp.lpSum(z)

    n_constraints = 0
    all_scenarios = list(range(n))

    for S in combinations(all_scenarios, d + l):
        for L_tuple in combinations(S, l):
            D_tuple = tuple(k for k in S if k not in set(L_tuple))
            W = _build_dl_witness_set(M, D_tuple, L_tuple, n)

            prob += pulp.lpSum(z[i] for i in W) >= 1
            n_constraints += 1

    print(f"  Nodes: {n}  |  d: {d}  |  l: {l}  |  Constraints added: {n_constraints}")

    _cplex_path = os.environ.get("CPLEX_PATH")
    if _cplex_path:
        prob.solve(pulp.CPLEX_CMD(msg=0, path=_cplex_path))
    else:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

    elapsed = time.time() - t0
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return [], None, elapsed, status

    obj = pulp.value(prob.objective)
    sensor_nodes = [
        nodes[c] if c < n else f"{nodes[c - n]}'"
        for c in range(2 * n) if pulp.value(z[c]) > 0.5
    ]
    return sensor_nodes, obj, elapsed, status


# ── Result file helpers ───────────────────────────────────────────────────────

def _load_done() -> set:
    done: set = set()
    if not RESULT_FILE.exists():
        return done
    for line in RESULT_FILE.read_text().splitlines():
        if line.startswith("DATA|"):
            parts = line.split("|")
            if len(parts) >= 4:
                try:
                    done.add((parts[1], int(parts[2]), int(parts[3])))
                except ValueError:
                    pass
    return done


def _append_result(graph_name: str, d: int, l: int, n: int,
                   size, elapsed: float, status: str,
                   sensor_nodes=None) -> None:
    sensors_str = ",".join(map(str, sensor_nodes)) if sensor_nodes else ""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"DATA|{graph_name}|{d}|{l}|{n}|{size}|{elapsed:.3f}|{status}"
        f"|{sensors_str}|{ts}\n"
    )
    with open(RESULT_FILE, "a") as f:
        f.write(line)


# ── Solve subprocess ──────────────────────────────────────────────────────────

def _solve_worker(graph_path: str, d: int, l: int, conn) -> None:
    """
    Runs inside a fresh subprocess for each solve.
    Sets RAM limit here so the main process is never constrained.
    Sends (status, size, elapsed, sensor_nodes) via pipe — no thread spawned,
    so this works even when virtual memory is nearly exhausted.
    """
    import resource
    _limit = int(RAM_LIMIT_PER_WORKER_GB * 1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_limit, _limit))
    except Exception:
        pass

    try:
        sensor_nodes, obj, elapsed, status = ilp_dl_disjunct_y_only(graph_path, d, l)
        if status != "Optimal":
            conn.send(("INFEASIBLE", -1, elapsed, []))
        else:
            conn.send(("OK", int(obj) if obj is not None else -1, elapsed, sensor_nodes))
    except MemoryError:
        conn.send(("RAM_LIMIT", -1, 0.0, []))
    except (Exception, SystemExit) as exc:
        conn.send(("ERROR", -1, 0.0, str(exc)[:60]))
    finally:
        conn.close()


def _run_solve(graph_path: str, d: int, l: int) -> tuple:
    """
    Spawn a subprocess for one ILP solve.
    Returns (status, size, elapsed, sensor_nodes).
    Timeout and OOM are handled without affecting the main process.
    """
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    proc = multiprocessing.Process(target=_solve_worker, args=(graph_path, d, l, child_conn))
    t0 = time.time()
    proc.start()
    child_conn.close()  # only used in child
    proc.join(timeout=TIMEOUT_SEC)
    elapsed = time.time() - t0

    if proc.is_alive():
        proc.kill()
        proc.join()
        parent_conn.close()
        return "TIME_LIMIT", -1, elapsed, []

    try:
        if parent_conn.poll(timeout=10):
            return parent_conn.recv()
    except Exception:
        pass
    finally:
        parent_conn.close()

    # Subprocess exited without sending — killed by OS OOM or unrecoverable crash
    return "RAM_LIMIT", -1, elapsed, []


# ── Per-graph runner ──────────────────────────────────────────────────────────

def _run_graph(graph_name: str, graph_file: str, done: set) -> None:
    graph_path = str(DATASETS_DIR / graph_file)
    nodes, _ = network_to_matrix(graph_path)
    n = len(nodes)

    for d, l in DL_PAIRS:
        key = (graph_name, d, l)

        if key in done:
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> SKIP (already done)")
            continue

        if d + l > n:
            _append_result(graph_name, d, l, n, 0, 0.0, "TRIVIAL")
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> TRIVIAL (d+l={d+l} > n={n})")
            continue

        num_constraints = math.comb(n, d + l) * math.comb(d + l, l)
        if num_constraints > CONSTRAINT_LIMIT:
            _append_result(graph_name, d, l, n, -1, 0.0, "RAM_LIMIT")
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> RAM_LIMIT (exceeded {RAM_LIMIT_PER_WORKER_GB} GB)")
            continue

        status, size, elapsed, sensor_nodes = _run_solve(graph_path, d, l)
        _append_result(graph_name, d, l, n, size, elapsed, status,
                       sensor_nodes if status == "OK" else None)

        if status == "OK":
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> OK  size={size}  time={elapsed:.1f}s")
        elif status == "TIME_LIMIT":
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> TIME_LIMIT after {elapsed:.1f}s (limit={TIMEOUT_SEC // 3600}h)")
        elif status == "RAM_LIMIT":
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> RAM_LIMIT (exceeded {RAM_LIMIT_PER_WORKER_GB} GB)")
        elif status == "INFEASIBLE":
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> INFEASIBLE (no y_i-only sensor set exists)")
        else:
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> {status}  {sensor_nodes[:60]}")


# ── Core experiment loop ──────────────────────────────────────────────────────

def run_experiments() -> None:
    all_graphs = _discover_graphs()
    graphs = all_graphs

    done = _load_done()
    pending = [(gn, gf) for gn, gf in graphs if any(
        (gn, d, l) not in done for d, l in DL_PAIRS
    )]

    print(f"Total graphs  : {len(all_graphs)}")
    print(f"Already done  : {len(done)} run(s)")
    print(f"Graphs to run : {len(pending)}")
    print(f"Timeout/pair  : {TIMEOUT_SEC // 3600}h")
    print(f"RAM limit     : {RAM_LIMIT_PER_WORKER_GB} GB")
    print(f"Const limit   : {CONSTRAINT_LIMIT:,}")
    print()

    for i, (gn, gf) in enumerate(pending, 1):
        _run_graph(gn, gf, done)
        print(f"  >> graph {i}/{len(pending)} finished: {gn}")


# ── Report generator ──────────────────────────────────────────────────────────

def _parse_results() -> list[dict]:
    rows = []
    if not RESULT_FILE.exists():
        return rows
    for line in RESULT_FILE.read_text().splitlines():
        if not line.startswith("DATA|"):
            continue
        parts = line.split("|")
        if len(parts) < 9:
            continue
        rows.append({
            "graph":   parts[1],
            "d":       int(parts[2]),
            "l":       int(parts[3]),
            "n":       int(parts[4]),
            "size":    parts[5],
            "elapsed": parts[6],
            "status":  parts[7],
            "sensors": parts[8],
            "ts":      parts[9] if len(parts) > 9 else "",
        })
    return rows


def write_report() -> None:
    rows = _parse_results()
    if not rows:
        print("No results to report.")
        return

    lines = [
        "=" * 72,
        "EXPERIMENT REPORT — (d,l)-Disjunct ILP Sensor Set Finder (y_i columns only)",
        f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timeout   : {TIMEOUT_SEC // 3600}h per (d,l) pair",
        f"Limit     : {CONSTRAINT_LIMIT:,} constraints",
        "=" * 72,
    ]

    graph_rows: dict[str, list] = {}
    for r in rows:
        graph_rows.setdefault(r["graph"], []).append(r)

    for gname in sorted(graph_rows):
        data = sorted(graph_rows[gname], key=lambda x: (x["d"], x["l"]))
        n = data[0]["n"] if data else "?"
        lines.append("")
        lines.append(f"Graph: {gname}  (n={n} nodes)")
        lines.append(f"  {'(d,l)':<10}  {'Time(s)':>8}  Sensor set / Note")
        lines.append(f"  {'-'*60}")
        for r in data:
            dl = f"({r['d']},{r['l']})"
            status = r["status"]
            if status == "OK":
                sensors = r["sensors"] if r["sensors"] else "∅"
                lines.append(f"  {dl:<10}  {float(r['elapsed']):>8.3f}  {sensors}")
            elif status == "TRIVIAL":
                lines.append(f"  {dl:<10}  {'—':>8}  TRIVIAL (d+l > n)")
            elif status == "TIME_LIMIT":
                lines.append(f"  {dl:<10}  {float(r['elapsed']):>8.1f}  TIME_LIMIT")
            elif status == "RAM_LIMIT":
                lines.append(f"  {dl:<10}  {float(r['elapsed']):>8.3f}  RAM_LIMIT")
            elif status == "INFEASIBLE":
                lines.append(f"  {dl:<10}  {float(r['elapsed']):>8.3f}  INFEASIBLE")
            else:
                lines.append(f"  {dl:<10}  {float(r['elapsed']):>8.3f}  {status[:50]}")

    lines.extend(["", "=" * 72])
    report_text = "\n".join(lines)

    existing = RESULT_FILE.read_text() if RESULT_FILE.exists() else ""
    data_lines = [ln for ln in existing.splitlines() if ln.startswith("DATA|")]
    with open(RESULT_FILE, "w") as f:
        for dl in data_lines:
            f.write(dl + "\n")
        f.write("\n")
        f.write(report_text + "\n")

    print("\n" + report_text)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    run_experiments()
    write_report()


if __name__ == "__main__":
    main()
