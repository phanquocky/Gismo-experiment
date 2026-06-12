"""
Run (d,l)-disjunct ILP experiments — first half of graphs (sorted alphabetically).
Results saved to Using_DDDisjunct_result1.txt.
Run alongside run_experiment2.py in a separate terminal for parallel coverage.

Each ILP solve runs in a dedicated subprocess so the main process is never
affected by memory limits or solver crashes.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "baseline" / "tools"))

from network_to_matrix import network_to_matrix  # noqa: E402

DATASETS_DIR = _HERE.parent / "datasets"
RESULT_FILE  = _HERE / "Using_DDDisjunct_result1.txt"

TIMEOUT_SEC             = 2 * 24 * 3600
RAM_LIMIT_PER_WORKER_GB = 30

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

def _solve_worker(graph_path: str, d: int, l: int, out_queue: multiprocessing.Queue) -> None:
    """
    Runs inside a fresh subprocess for each solve.
    Sets RAM limit here so the main process is never constrained.
    Puts (status, size, elapsed, sensor_nodes) into out_queue.
    """
    import resource
    _limit = int(RAM_LIMIT_PER_WORKER_GB * 1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_limit, _limit))
    except Exception:
        pass

    from dldisjunct import ilp_dl_disjunct
    try:
        sensor_nodes, obj, elapsed = ilp_dl_disjunct(graph_path, d, l)
        out_queue.put(("OK", int(obj) if obj is not None else -1, elapsed, sensor_nodes))
    except MemoryError:
        out_queue.put(("TOO_LARGE", -1, 0.0, []))
    except (Exception, SystemExit) as exc:
        out_queue.put(("ERROR", -1, 0.0, str(exc)[:60]))


def _run_solve(graph_path: str, d: int, l: int) -> tuple:
    """
    Spawn a subprocess for one ILP solve.
    Returns (status, size, elapsed, sensor_nodes).
    Timeout and OOM are handled without affecting the main process.
    """
    out_queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_solve_worker, args=(graph_path, d, l, out_queue))
    t0 = time.time()
    proc.start()
    proc.join(timeout=TIMEOUT_SEC)
    elapsed = time.time() - t0

    if proc.is_alive():
        proc.kill()
        proc.join()
        return "TIMEOUT", -1, elapsed, []

    try:
        return out_queue.get(timeout=10)
    except Exception:
        # Process exited without writing a result — killed by OOM or crashed
        return "TOO_LARGE", -1, elapsed, []


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

        status, size, elapsed, sensor_nodes = _run_solve(graph_path, d, l)
        _append_result(graph_name, d, l, n, size, elapsed, status,
                       sensor_nodes if status == "OK" else None)

        if status == "OK":
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> OK  size={size}  time={elapsed:.1f}s")
        elif status == "TIMEOUT":
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> TIMEOUT after {elapsed:.1f}s")
        elif status == "TOO_LARGE":
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> TOO_LARGE (exceeded {RAM_LIMIT_PER_WORKER_GB} GB RAM)")
        else:
            print(f"[{graph_name}] d={d:2d} l={l:2d} -> {status}  {sensor_nodes[:60]}")


# ── Core experiment loop ──────────────────────────────────────────────────────

def run_experiments() -> None:
    all_graphs = _discover_graphs()
    half = len(all_graphs) // 2
    graphs = all_graphs[:half]

    done = _load_done()
    pending = [(gn, gf) for gn, gf in graphs if any(
        (gn, d, l) not in done for d, l in DL_PAIRS
    )]

    print(f"Total graphs  : {len(all_graphs)}  (this file: first {half})")
    print(f"Already done  : {len(done)} run(s)")
    print(f"Graphs to run : {len(pending)}")
    print(f"Timeout/pair  : {TIMEOUT_SEC // 3600}h")
    print(f"RAM limit     : {RAM_LIMIT_PER_WORKER_GB} GB")
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
        "EXPERIMENT REPORT — (d,l)-Disjunct ILP Sensor Set Finder (Part 1)",
        f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timeout   : {TIMEOUT_SEC // 3600}h per (d,l) pair",
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
            elif status == "TIMEOUT":
                lines.append(f"  {dl:<10}  {float(r['elapsed']):>8.1f}  TIMEOUT")
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
