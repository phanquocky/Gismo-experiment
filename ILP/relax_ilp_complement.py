"""
Run (d,l)-disjunct LP-relaxation experiments using y_i and its complement
y_i' (x_i columns dropped).

Same experiment setup as complement.py (datasets, (d,l) pairs, timeout, RAM
limit, subprocess isolation) — but the ILP is relaxed to an LP: each column's
selection variable z_c lives in [0,1] instead of {0,1}. We no longer solve to
integer optimality; instead a minimum sensor set is *sampled* from the LP
relaxation's fractional solution.

The x_i failing-node columns are removed from the witness matrix. In their
place, each sensor i contributes two witness columns, mirroring the original
x_i/y_i pair structure but built entirely from y:

  - y_i  : M[k][i]  = 1 iff node i is in the closed neighbourhood of k
  - y_i' : M[k][i]' = 1 - M[k][i]   (the complement of y_i)

So the witness matrix has 2n columns total (y block + y' block), and each
column has its own independent LP variable: z_i controls node i's y_i
column, and z_{n+i} controls node i's y_i' column.

    sum_{c in W(D,L)} z_c  >=  1      (0 <= z_c <= 1)
        for all D,L with |D|=d, |L|=l, D∩L=empty, D∪L ⊆ {0..n-1}

W(D,L) = column indices c (0..2n-1) that can witness (D, L):
  - M[k][c] = 0  for all k in D   (sensor's column is silent for every failing node in D)
  - M[k][c] = 1  for some k in L  (sensor's column fires for at least one failing node in L)

All constraints are built upfront and the LP is solved once. After that,
NUM_SAMPLES independent Bernoulli(z_c*) draws are taken from the same LP
solution (no re-solving) and checked against the already-built witness sets;
the smallest draw that satisfies every constraint is kept. If none of the
draws is fully feasible, the draw with the fewest violated constraints is
reported instead, tagged SAMPLE_INFEASIBLE.

Results saved to Using_YOnlyDisjunct_LPRelax_result.txt.
"""

from __future__ import annotations

import multiprocessing
import os
import random
import shutil
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import pulp

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "tools" ))

from network_to_matrix import network_to_matrix_y_only, parse_network  # noqa: E402

DATASETS_DIR = _HERE.parent / "datasets"
RESULT_FILE  = _HERE / "Using_YOnlyDisjunct_LPRelax_result.txt"

TIMEOUT_SEC             = 8 * 3600
RAM_LIMIT_PER_WORKER_GB = 80
NUM_SAMPLES             = 10000

_cplex_bin = shutil.which("cplex")
if _cplex_bin:
    os.environ["CPLEX_PATH"] = _cplex_bin
    print(f"Solver        : CPLEX  ({_cplex_bin})")
else:
    print("Solver        : CBC  (cplex not found on PATH)")

# _D_VALUES = [1, 2, 3, 4, 6, 8, 10, 12, 16]
_D_VALUES = [1, 2, 3, 4]
DL_PAIRS  = [(d, d) for d in _D_VALUES]

# Small graphs only (10-86 nodes) — the entire small-graph cluster in
# datasets/ (the next smallest graph after this jumps to 332 nodes).
SMALL_GRAPHS = [
    "MadridAdj.txt",
    "inf-USAir97.mtx",
    "power-1138-bus.mtx",
    "web-edu.mtx"
]


def _discover_graphs() -> list[tuple[str, str]]:
    result = []
    for name in SMALL_GRAPHS:
        path = DATASETS_DIR / name
        if path.exists():
            result.append((path.stem, path.name))
        else:
            print(f"WARNING: dataset not found, skipping: {name}")
    return result


# ── LP-relaxation solver (y_i and y_i' columns) ────────────────────────────────

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


def lp_relax_dl_disjunct_y_only(
    network_file: str, d: int, l: int
) -> Tuple[List[int], object, float, str, int, int]:
    """
    Solve the LP relaxation (z_c in [0,1]) of the minimum (d,l)-disjunct
    sensor-set covering problem, using y_i and its complement y_i' in place
    of the x_i/y_i pair, then sample a sensor set from the fractional
    solution.

    Some (d,l) pairs may have no LP-feasible solution at all (some (D,L)
    split has an empty witness set) — in that case the LP status will not be
    "Optimal" and no sensor set is returned.

    Returns (sensor_nodes, size, elapsed, status, violations, success_count):
      status: "OK" (a fully feasible sample was found), "SAMPLE_INFEASIBLE"
        (no sample among NUM_SAMPLES satisfied every constraint — the
        least-violating sample is returned instead), or the raw pulp LP
        status (e.g. "Infeasible") when the LP itself has no solution.
      violations: 0 for "OK", violated-constraint count for
        "SAMPLE_INFEASIBLE", -1 when not applicable.
      success_count: how many of the NUM_SAMPLES Bernoulli draws were fully
        feasible (satisfied every witness constraint) — success_count /
        NUM_SAMPLES is the empirical success probability of the sampling
        strategy. -1 when not applicable (LP itself infeasible).
    """
    t0 = time.time()

    nodes, y_matrix = network_to_matrix_y_only(network_file, dedupe_twins=True)
    n = len(nodes)
    Y = y_matrix[1:]  # drop the all-zero empty row -> shape (n, n)
    M = [row + [1 - v for v in row] for row in Y]  # shape (n, 2n): y block + y' block

    prob = pulp.LpProblem("lp_relax_dl_disjunct_sensor_set_y_only", pulp.LpMinimize)
    # z_i controls node i's y_i column (i in 0..n-1);
    # z_{n+i} controls node i's y_i' column, selected independently of z_i.
    z = [pulp.LpVariable(f"z_{c}", lowBound=0, upBound=1, cat="Continuous") for c in range(2 * n)]

    prob += pulp.lpSum(z)

    witness_sets: List[set] = []
    all_scenarios = list(range(n))

    for S in combinations(all_scenarios, d + l):
        for L_tuple in combinations(S, l):
            D_tuple = tuple(k for k in S if k not in set(L_tuple))
            W = _build_dl_witness_set(M, D_tuple, L_tuple, n)

            witness_sets.append(W)
            prob += pulp.lpSum(z[i] for i in W) >= 1

    print(f"  Nodes: {n}  |  d: {d}  |  l: {l}  |  Constraints added: {len(witness_sets)}")

    _cplex_path = os.environ.get("CPLEX_PATH")
    if _cplex_path:
        prob.solve(pulp.CPLEX_CMD(msg=0, path=_cplex_path))
    else:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        elapsed = time.time() - t0
        return [], None, elapsed, status, -1, -1

    z_star = [pulp.value(zc) or 0.0 for zc in z]

    feasible_best: set | None = None
    infeasible_best: tuple[set, int] | None = None
    success_count = 0

    for _ in range(NUM_SAMPLES):
        selected = {c for c in range(2 * n) if random.random() < z_star[c]}
        violations = sum(1 for W in witness_sets if selected.isdisjoint(W))

        if violations == 0:
            success_count += 1
            if feasible_best is None or len(selected) < len(feasible_best):
                feasible_best = selected
        elif infeasible_best is None or violations < infeasible_best[1]:
            infeasible_best = (selected, violations)

    elapsed = time.time() - t0
    print(f"  Success rate: {success_count}/{NUM_SAMPLES} ({100 * success_count / NUM_SAMPLES:.2f}%)")

    if feasible_best is not None:
        sensor_nodes = [nodes[c] if c < n else f"{nodes[c - n]}'" for c in sorted(feasible_best)]
        return sensor_nodes, len(sensor_nodes), elapsed, "OK", 0, success_count

    selected, violations = infeasible_best
    sensor_nodes = [nodes[c] if c < n else f"{nodes[c - n]}'" for c in sorted(selected)]
    return sensor_nodes, len(sensor_nodes), elapsed, "SAMPLE_INFEASIBLE", violations, success_count


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
                   size, elapsed: float, status: str, violations: int,
                   sensor_nodes=None, success_count: int = -1) -> None:
    sensors_str = ",".join(map(str, sensor_nodes)) if sensor_nodes else ""
    success_str = f"{success_count}/{NUM_SAMPLES}" if success_count >= 0 else ""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"DATA|{graph_name}|{d}|{l}|{n}|{size}|{elapsed:.3f}|{status}|{violations}"
        f"|{sensors_str}|{ts}|{success_str}\n"
    )
    with open(RESULT_FILE, "a") as f:
        f.write(line)


# ── Solve subprocess ──────────────────────────────────────────────────────────

def _solve_worker(graph_path: str, d: int, l: int, conn) -> None:
    """
    Runs inside a fresh subprocess for each solve.
    Sets RAM limit here so the main process is never constrained.
    Sends (status, size, elapsed, sensor_nodes, violations, success_count)
    via pipe — no thread spawned, so this works even when virtual memory is
    nearly exhausted.
    """
    import resource
    _limit = int(RAM_LIMIT_PER_WORKER_GB * 1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_limit, _limit))
    except Exception:
        pass

    try:
        sensor_nodes, size, elapsed, status, violations, success_count = lp_relax_dl_disjunct_y_only(graph_path, d, l)
        if status in ("OK", "SAMPLE_INFEASIBLE"):
            conn.send((status, size, elapsed, sensor_nodes, violations, success_count))
        else:
            conn.send(("INFEASIBLE", -1, elapsed, [], -1, -1))
    except MemoryError:
        conn.send(("RAM_LIMIT", -1, 0.0, [], -1, -1))
    except (Exception, SystemExit) as exc:
        conn.send(("ERROR", -1, 0.0, str(exc)[:60], -1, -1))
    finally:
        conn.close()


def _run_solve(graph_path: str, d: int, l: int) -> tuple:
    """
    Spawn a subprocess for one LP-relaxation solve + sample.
    Returns (status, size, elapsed, sensor_nodes, violations, success_count).
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
        return "TIME_LIMIT", -1, elapsed, [], -1, -1

    try:
        if parent_conn.poll(timeout=10):
            return parent_conn.recv()
    except Exception:
        pass
    finally:
        parent_conn.close()

    # Subprocess exited without sending — killed by OS OOM or unrecoverable crash
    return "RAM_LIMIT", -1, elapsed, [], -1, -1


# ── Per-(graph, d, l) runner ────────────────────────────────────────────────────

def _run_graph_dl(graph_name: str, graph_path: str, n: int, d: int, l: int, done: set) -> None:
    key = (graph_name, d, l)

    if key in done:
        print(f"[{graph_name}] d={d:2d} l={l:2d} -> SKIP (already done)")
        return

    if d + l > n:
        _append_result(graph_name, d, l, n, 0, 0.0, "TRIVIAL", -1)
        print(f"[{graph_name}] d={d:2d} l={l:2d} -> TRIVIAL (d+l={d+l} > n={n})")
        return

    status, size, elapsed, sensor_nodes, violations, success_count = _run_solve(graph_path, d, l)
    _append_result(graph_name, d, l, n, size, elapsed, status, violations,
                   sensor_nodes if status in ("OK", "SAMPLE_INFEASIBLE") else None,
                   success_count)

    if status == "OK":
        rate = 100 * success_count / NUM_SAMPLES
        print(f"[{graph_name}] d={d:2d} l={l:2d} -> OK  size={size}  "
              f"success={success_count}/{NUM_SAMPLES} ({rate:.2f}%)  time={elapsed:.1f}s")
    elif status == "SAMPLE_INFEASIBLE":
        print(f"[{graph_name}] d={d:2d} l={l:2d} -> SAMPLE_INFEASIBLE "
              f"(best of {NUM_SAMPLES} samples still violates {violations} constraint(s); "
              f"success=0/{NUM_SAMPLES})  time={elapsed:.1f}s")
    elif status == "TIME_LIMIT":
        print(f"[{graph_name}] d={d:2d} l={l:2d} -> TIME_LIMIT after {elapsed:.1f}s (limit={TIMEOUT_SEC // 3600}h)")
    elif status == "RAM_LIMIT":
        print(f"[{graph_name}] d={d:2d} l={l:2d} -> RAM_LIMIT (exceeded {RAM_LIMIT_PER_WORKER_GB} GB)")
    elif status == "INFEASIBLE":
        print(f"[{graph_name}] d={d:2d} l={l:2d} -> INFEASIBLE (no y_i-only LP relaxation exists)")
    else:
        print(f"[{graph_name}] d={d:2d} l={l:2d} -> {status}  {sensor_nodes[:60]}")


# ── Core experiment loop ──────────────────────────────────────────────────────

def run_experiments() -> None:
    all_graphs = _discover_graphs()
    done = _load_done()

    # Parse each graph's node count once up front so the (d,l)-outer loop
    # doesn't re-parse the network file for every pair.
    graph_info = []
    for gn, gf in all_graphs:
        graph_path = str(DATASETS_DIR / gf)
        nodes, _ = parse_network(graph_path)
        graph_info.append((gn, graph_path, len(nodes)))

    pending = sum(
        1 for gn, _, _ in graph_info for d, l in DL_PAIRS if (gn, d, l) not in done
    )

    print(f"Total graphs  : {len(all_graphs)}")
    print(f"Already done  : {len(done)} run(s)")
    print(f"Runs to do    : {pending}")
    print(f"Timeout/pair  : {TIMEOUT_SEC // 3600}h")
    print(f"RAM limit     : {RAM_LIMIT_PER_WORKER_GB} GB")
    print(f"Samples/pair  : {NUM_SAMPLES}")
    print()

    for d, l in DL_PAIRS:
        print(f"=== (d,l) = ({d},{l}) ===")
        for gn, graph_path, n in graph_info:
            _run_graph_dl(gn, graph_path, n, d, l, done)
        print(f"  >> (d,l)=({d},{l}) finished")


# ── Report generator ──────────────────────────────────────────────────────────

def _parse_results() -> list[dict]:
    rows = []
    if not RESULT_FILE.exists():
        return rows
    for line in RESULT_FILE.read_text().splitlines():
        if not line.startswith("DATA|"):
            continue
        parts = line.split("|")
        if len(parts) < 10:
            continue
        rows.append({
            "graph":      parts[1],
            "d":          int(parts[2]),
            "l":          int(parts[3]),
            "n":          int(parts[4]),
            "size":       parts[5],
            "elapsed":    parts[6],
            "status":     parts[7],
            "violations": parts[8],
            "sensors":    parts[9],
            "ts":         parts[10] if len(parts) > 10 else "",
            "success":    parts[11] if len(parts) > 11 else "",
        })
    return rows


def write_report() -> None:
    rows = _parse_results()
    if not rows:
        print("No results to report.")
        return

    lines = [
        "=" * 72,
        "EXPERIMENT REPORT — (d,l)-Disjunct LP-Relaxation Sensor Set Finder (y_i columns only)",
        f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timeout   : {TIMEOUT_SEC // 3600}h per (d,l) pair",
        f"Samples   : {NUM_SAMPLES} per (d,l) pair",
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
                rate_note = ""
                if r["success"] and "/" in r["success"]:
                    succ, total = r["success"].split("/")
                    rate_note = f"  [success={r['success']} ({100 * int(succ) / int(total):.2f}%)]"
                lines.append(f"  {dl:<10}  {float(r['elapsed']):>8.3f}  {sensors}{rate_note}")
            elif status == "SAMPLE_INFEASIBLE":
                sensors = r["sensors"] if r["sensors"] else "∅"
                lines.append(f"  {dl:<10}  {float(r['elapsed']):>8.3f}  "
                              f"SAMPLE_INFEASIBLE (violations={r['violations']})  {sensors}")
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
