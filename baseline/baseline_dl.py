"""
Baseline algorithm for sensor placement using (d,l)-disjunctive matrix reduction.

Task 5: Greedy algorithm that removes node pairs (x_i, y_i) from the matrix
        while maintaining (d,l)-disjunctiveness.
Task 6: Batch experiments across all datasets for the (d,l) pairs defined in
        DL_PAIRS below — N_WORKERS parallel subprocesses, each bounded by
        TIME_LIMIT_SEC and RAM_LIMIT_GB.
"""

from __future__ import annotations

import multiprocessing
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "tools"))

from network_to_matrix import network_to_matrix, parse_network  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE        = Path(__file__).parent
DATASETS_DIR = _BASE.parent / "datasets"
RESULTS_FILE = _BASE / "Using_DL_Disjunct_result.txt"
REPORT_FILE  = _BASE / "Using_DL_Disjunct_report.txt"

# ── Experiment config ─────────────────────────────────────────────────────────

TIME_LIMIT_SEC = 8 * 3600   # 8 hours per (graph, d, l) task
RAM_LIMIT_GB   = 32          # GB of virtual memory per worker subprocess
N_WORKERS      = 16           # parallel subprocess slots

# (d, l) pairs from the spec.
DL_PAIRS: List[Tuple[int, int]] = [
    (1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (16, 16),
]


# ── Task 5: greedy sensor placement ──────────────────────────────────────────

def dl_sensor_placement(
    network_file: str,
    d: int,
    l: int,
    m_trials: int = 100_000,
    seed: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    Greedy baseline for finding a (near-)minimal sensor set.

    Speedups vs. the naive version:
      * NumPy arrays — column extraction.
      * Vectorised witness-column check — replaces Python loops.
      * Exhaustive C(d+l, l) enumeration per trial — for each sampled S of size
        d+l, all L-subsets of size l are checked. This is exact per S: if any
        partition has no witness column the node is immediately marked essential.
      * Essential-node caching — once essential, a node is never retested (the
        active column set only shrinks, making the property harder to satisfy).

    Returns:
        sensor_nodes : list of node IDs in the sensor set.
        elapsed      : wall-clock seconds.
    """
    rng = np.random.default_rng(seed)
    if seed is not None:
        random.seed(seed)

    t0 = time.time()
    nodes, matrix = network_to_matrix(network_file)
    n = len(nodes)

    # Convert once; skip row 0 (all-zeros empty state)
    mat = np.array(matrix[1:], dtype=np.int8)  # (n_scenarios, 2n)

    active_cols: Set[int] = set(range(2 * n))
    sensor_indices: Set[int] = set(range(n))
    essential_indices: Set[int] = set()  # confirmed essential — never retest

    dl = d + l

    while True:
        made_progress = False
        candidates = list(sensor_indices - essential_indices)
        random.shuffle(candidates)

        for node_idx in candidates:
            x_col = node_idx
            y_col = n + node_idx
            test_cols = sorted(active_cols - {x_col, y_col})

            if len(test_cols) < dl:
                essential_indices.add(node_idx)
                continue

            # Submatrix: rows = failure scenarios, cols = active col pairs
            sub = mat[:, test_cols]  # (n_scenarios, |test_cols|)
            n_sub_rows = sub.shape[0]
            if n_sub_rows < dl:
                essential_indices.add(node_idx)
                continue

            # Exhaustive (d,l)-disjunct check (NumPy vectorised).
            # Each trial picks a random S of size d+l from failure scenarios,
            # then enumerates ALL C(d+l, l) L-subsets of S. A witness column
            # is one that reads 0 on all D-scenarios and 1 on some L-scenario.
            is_dl = True
            for _ in range(m_trials):
                S = rng.choice(n_sub_rows, dl, replace=False)

                for L_local in combinations(range(dl), l):
                    L_mask = np.zeros(dl, dtype=bool)
                    L_mask[list(L_local)] = True
                    D_mask = ~L_mask

                    D_sub = sub[S[D_mask], :]  # (d, |test_cols|)
                    L_sub = sub[S[L_mask], :]  # (l, |test_cols|)

                    # Witness column: all D-rows are 0 AND at least one L-row is 1
                    d_ok = np.all(D_sub == 0, axis=0)
                    l_ok = np.any(L_sub == 1, axis=0)

                    if not bool(np.any(d_ok & l_ok)):
                        is_dl = False
                        break

                if not is_dl:
                    break

            if is_dl:
                active_cols.discard(x_col)
                active_cols.discard(y_col)
                sensor_indices.discard(node_idx)
                made_progress = True
            else:
                essential_indices.add(node_idx)

        if not made_progress:
            break

    elapsed = time.time() - t0
    sensor_nodes = [nodes[i] for i in sorted(sensor_indices)]
    return sensor_nodes, elapsed


# ── Subprocess worker ─────────────────────────────────────────────────────────

def _worker_subprocess(graph_path: str, d: int, l: int, conn) -> None:
    """
    Runs inside a dedicated subprocess for each (graph, d, l) task.
    Sets RAM limit via resource.setrlimit, then runs dl_sensor_placement.
    Sends (status, size, elapsed, sensors) back through the pipe.
    """
    import resource
    limit = int(RAM_LIMIT_GB * 1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass

    try:
        nodes_list, _ = parse_network(graph_path)
        if d + l > len(nodes_list):
            conn.send(("TRIVIAL", 0, 0.0, []))
            return
        sensors, elapsed = dl_sensor_placement(graph_path, d, l)
        conn.send(("OK", len(sensors), elapsed, sensors))
    except MemoryError:
        conn.send(("RAM_LIMIT", -1, 0.0, []))
    except (Exception, SystemExit) as exc:
        conn.send(("ERROR", -1, 0.0, str(exc)[:120]))
    finally:
        conn.close()


def _run_one_task(graph_path: str, d: int, l: int) -> tuple:
    """
    Thread-level task runner.
    Spawns a subprocess, waits up to TIME_LIMIT_SEC, then kills if still alive.
    Returns (status, size, elapsed, sensors).
      status: "OK" | "TIME_LIMIT" | "RAM_LIMIT" | "ERROR"
    """
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    proc = multiprocessing.Process(
        target=_worker_subprocess,
        args=(graph_path, d, l, child_conn),
    )
    t0 = time.time()
    proc.start()
    child_conn.close()  # only used in child

    proc.join(timeout=TIME_LIMIT_SEC)
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


# ── Result helpers ────────────────────────────────────────────────────────────

def _load_done() -> set:
    """Return set of (dataset_name, d, l) already recorded in RESULTS_FILE."""
    done: set = set()
    if not RESULTS_FILE.exists():
        return done
    for line in RESULTS_FILE.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        try:
            name = parts[0]
            d = int(parts[1].split("=")[1])
            l = int(parts[2].split("=")[1])
            done.add((name, d, l))
        except (ValueError, IndexError):
            pass
    return done


def _append_raw_result(
    dataset_name: str, d: int, l: int, sensor_size: int, elapsed: float, status: str,
) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{dataset_name}\td={d}\tl={l}\tsensor_size={sensor_size}"
            f"\telapsed={elapsed:.2f}s\tstatus={status}\tts={ts}\n"
        )


# ── Batch experiment runner ───────────────────────────────────────────────────

def run_experiments() -> None:
    """
    Task 6: Run dl_sensor_placement on every (dataset, d, l) combination.

    Layout:
      - N_WORKERS threads each manage one subprocess slot.
      - Each subprocess is bounded by TIME_LIMIT_SEC and RAM_LIMIT_GB.
      - Every completed task (OK / TIME_LIMIT / RAM_LIMIT / ERROR) is written
        to RESULTS_FILE before the next task starts — no result is ever lost.
      - Already-recorded tasks are skipped on resume.
    """
    done = _load_done()

    dataset_files = sorted(
        f for f in DATASETS_DIR.iterdir() if f.suffix in (".txt", ".mtx", ".edges")
    )

    tasks = [
        (ds_path, d, l)
        for d, l in DL_PAIRS
        for ds_path in dataset_files
        if (ds_path.name, d, l) not in done
    ]

    total = len(tasks)
    print("=" * 60)
    print(f"Datasets       : {len(dataset_files)}")
    print(f"(d,l) pairs    : {len(DL_PAIRS)}  {DL_PAIRS}")
    print(f"Total tasks    : {total}  ({len(done)} already done, skipped)")
    print(f"Workers        : {N_WORKERS} parallel subprocesses")
    print(f"Time limit     : {TIME_LIMIT_SEC // 3600}h per task")
    print(f"RAM limit      : {RAM_LIMIT_GB} GB per worker")
    print("=" * 60)
    print()

    if not tasks:
        print("Nothing to do — all tasks already recorded.")
        _write_report([f.name for f in dataset_files])
        return

    write_lock = threading.Lock()
    completed = 0

    def handle_task(ds_path: Path, d: int, l: int) -> None:
        nonlocal completed
        name = ds_path.name
        ts_start = time.strftime("%H:%M:%S")
        print(f"[{ts_start}] START  {name}  d={d} l={l}")

        status, size, elapsed, _ = _run_one_task(str(ds_path), d, l)

        with write_lock:
            _append_raw_result(name, d, l, size, elapsed, status)
            completed += 1
            ts_end = time.strftime("%H:%M:%S")
            print(
                f"[{ts_end}] DONE   {name}  d={d} l={l}"
                f"  -> {status}  size={size}  time={elapsed:.1f}s"
                f"  [{completed}/{total}]"
            )

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(handle_task, ds_path, d, l): (ds_path.name, d, l)
            for ds_path, d, l in tasks
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                name, d, l = futures[future]
                print(f"[UNEXPECTED ERROR] {name} d={d} l={l}: {exc}")

    print()
    _write_report([f.name for f in dataset_files])


# ── Report generator ──────────────────────────────────────────────────────────

def _write_report(dataset_names: List[str]) -> None:
    records: dict = {}
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            try:
                name = parts[0]
                d = int(parts[1].split("=")[1])
                l = int(parts[2].split("=")[1])
                size_str = parts[3].split("=")[1]
                elapsed_str = parts[4].split("=")[1].rstrip("s")
                status = parts[5].split("=")[1]
                records[f"{name}|d={d}|l={l}"] = {
                    "sensor_size": int(size_str),
                    "elapsed": float(elapsed_str),
                    "status": status,
                }
            except (ValueError, IndexError):
                pass

    col_w = 14
    dl_headers = [f"d={d},l={l}" for d, l in DL_PAIRS]

    def _row(label: str, values: list) -> str:
        return f"{label:<45}" + "".join(f"{str(v):>{col_w}}" for v in values)

    header = _row("Dataset", dl_headers)
    sep    = "-" * len(header)

    def _size_cell(rec: Optional[dict]) -> str:
        if rec is None:
            return "N/A"
        s = rec.get("status", "OK")
        return str(rec["sensor_size"]) if s == "OK" else s

    def _time_cell(rec: Optional[dict]) -> str:
        if rec is None:
            return "N/A"
        return f"{rec['elapsed']:.2f}"

    def _section(title: str, cell_fn) -> List[str]:
        lines = [title, sep, header, sep]
        for name in dataset_names:
            vals = [cell_fn(records.get(f"{name}|d={d}|l={l}")) for d, l in DL_PAIRS]
            lines.append(_row(name, vals))
        return lines

    report_lines = (
        [
            "Baseline Algorithm – (d,l)-Disjunct Greedy Reduction",
            "=" * 80,
            f"Generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Time limit : {TIME_LIMIT_SEC // 3600}h per task",
            f"RAM limit  : {RAM_LIMIT_GB} GB per worker",
            f"Workers    : {N_WORKERS}",
            "=" * 80,
            "",
        ]
        + _section(
            "Sensor Set Size  (OK → count, otherwise: TIME_LIMIT / RAM_LIMIT / ERROR)",
            _size_cell,
        )
        + [""]
        + _section("Run Time (seconds)", _time_cell)
        + [""]
    )

    REPORT_FILE.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved → {REPORT_FILE}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_experiments()
