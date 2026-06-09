"""
Task 4: Run (d,l)-disjunct ILP experiments on all datasets.

Strategy per (graph, d, l):
  - d+l > n  → TRIVIAL (0 constraints, any sensor set works)
  - otherwise → solve ILP with a 2-day wall-clock timeout

Results are saved incrementally to Using_DDDisjunct_result.txt after each run.
Before starting, existing results are loaded so completed runs are skipped.
Graphs run in parallel (up to MAX_WORKERS concurrent processes); within each
graph, the 9 (d,l) pairs are solved sequentially.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "baseline" / "tools"))

from network_to_matrix import network_to_matrix  # noqa: E402

DATASETS_DIR = _HERE.parent / "datasets"
RESULT_FILE  = _HERE / "Using_DDDisjunct_result.txt"

TIMEOUT_SEC = 2 * 24 * 3600   # 2 days per (d,l) pair
MAX_WORKERS = 10

_cplex_bin = shutil.which("cplex")
if _cplex_bin:
    os.environ["CPLEX_PATH"] = _cplex_bin

_D_VALUES = [1, 2, 3, 4, 6, 8, 10, 12, 16]
DL_PAIRS  = [(d, d) for d in _D_VALUES]


def _discover_graphs() -> list[tuple[str, str]]:
    """Return (stem_name, filename) for every graph file in DATASETS_DIR."""
    result = []
    for path in sorted(DATASETS_DIR.iterdir()):
        if path.suffix in (".txt", ".mtx", ".edges") and "report" not in path.name:
            result.append((path.stem, path.name))
    return result


# ── Timeout helper ────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


# ── Result file helpers ───────────────────────────────────────────────────────

def _load_done() -> set:
    """Return set of (graph_name, d, l) already recorded in the result file."""
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
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ── Per-graph worker ──────────────────────────────────────────────────────────

def _run_graph(graph_name: str, graph_file: str, done: set) -> list[str]:
    """
    Run all DL_PAIRS for one graph sequentially.
    Skips pairs already in `done` (snapshot from run start).
    Returns a list of result strings for the caller to print.
    """
    from dldisjunct import ilp_dl_disjunct  # imported inside worker for spawn safety

    graph_path = str(DATASETS_DIR / graph_file)
    nodes, _ = network_to_matrix(graph_path)
    n = len(nodes)

    msgs = []
    for d, l in DL_PAIRS:
        key = (graph_name, d, l)

        if key in done:
            msgs.append(f"[{graph_name}] d={d:2d} l={l:2d} -> SKIP (already done)")
            continue

        if d + l > n:
            _append_result(graph_name, d, l, n, 0, 0.0, "TRIVIAL")
            msgs.append(f"[{graph_name}] d={d:2d} l={l:2d} -> TRIVIAL (d+l={d+l} > n={n})")
            continue

        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(TIMEOUT_SEC)
        t0 = time.time()
        try:
            sensor_nodes, obj, elapsed = ilp_dl_disjunct(graph_path, d, l)
            signal.alarm(0)
            size = int(obj) if obj is not None else -1
            _append_result(graph_name, d, l, n, size, elapsed, "OK", sensor_nodes)
            msgs.append(
                f"[{graph_name}] d={d:2d} l={l:2d} -> OK  size={size}  time={elapsed:.1f}s"
            )
        except _Timeout:
            elapsed = time.time() - t0
            _append_result(graph_name, d, l, n, -1, elapsed, "TIMEOUT")
            msgs.append(f"[{graph_name}] d={d:2d} l={l:2d} -> TIMEOUT after {elapsed:.1f}s")
        except (Exception, SystemExit) as exc:
            signal.alarm(0)
            elapsed = time.time() - t0
            msg = str(exc)[:60].replace("|", ";")
            _append_result(graph_name, d, l, n, -1, elapsed, f"ERROR_{msg}")
            msgs.append(f"[{graph_name}] d={d:2d} l={l:2d} -> ERROR: {msg}")

    return msgs


# ── Core experiment loop ──────────────────────────────────────────────────────

def run_experiments() -> None:
    graphs = _discover_graphs()
    done   = _load_done()

    pending = [(gn, gf) for gn, gf in graphs if any(
        (gn, d, l) not in done for d, l in DL_PAIRS
    )]

    print(f"Graphs found  : {len(graphs)}")
    print(f"Already done  : {len(done)} run(s)")
    print(f"Graphs to run : {len(pending)}")
    print(f"Max workers   : {MAX_WORKERS}")
    print(f"Timeout/pair  : {TIMEOUT_SEC // 3600}h")
    print()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_run_graph, gn, gf, done): gn
            for gn, gf in pending
        }
        completed = 0
        for fut in as_completed(futures):
            gn = futures[fut]
            completed += 1
            try:
                for msg in fut.result():
                    print(msg)
            except Exception as exc:
                print(f"[{gn}] WORKER CRASHED: {exc}")
            print(f"  >> graph {completed}/{len(pending)} finished: {gn}")


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
        "EXPERIMENT REPORT — (d,l)-Disjunct ILP Sensor Set Finder",
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
