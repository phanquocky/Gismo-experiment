"""
Baseline algorithm for sensor placement using k-disjunctive matrix reduction.

Task 3: Greedy algorithm that removes node pairs (x_i, y_i) from the matrix
        while maintaining k-disjunctiveness.
Task 4: Batch experiments across all datasets for k in {1,2,3,4,6,8,10,12,16}.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "tools"))

from network_to_matrix import network_to_matrix  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE = Path(__file__).parent
DATASETS_DIR = _BASE.parent / "datasets"
RESULTS_FILE = _BASE / "Using_DDisjunct_result.txt"
RECORDS_FILE = _BASE / "Using_DDisjunct_records.json"
REPORT_FILE  = _BASE / "Using_DDisjunct_report.txt"

K_VALUES = [1, 2, 3, 4, 6, 8, 10, 12, 16]

# ── Task 3: greedy sensor placement ──────────────────────────────────────────

def baseline_sensor_placement(
    network_file: str,
    k: int,
    m_trials: int = 10_000,
    seed: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    Greedy baseline for finding a (near-)minimal sensor set.

    Speedups vs. the naive version:
      * NumPy arrays — column extraction and transpose are O(1) views, not list copies.
      * Fast identity check — uses row-sum comparison instead of Python set-of-tuples.
      * Essential-node caching — once a node is found essential it is never retested.
        Proof: the active column set only shrinks between passes, so the transposed
        submatrix only loses rows, making k-disjunctness harder to satisfy.
        A node that was essential before is still essential after any further removals.

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
    mat = np.array(matrix[1:], dtype=np.int8)  # (n, 2n)

    active_cols: Set[int] = set(range(2 * n))
    sensor_indices: Set[int] = set(range(n))
    essential_indices: Set[int] = set()  # confirmed essential — never retest

    k1 = k + 1

    while True:
        made_progress = False
        candidates = list(sensor_indices - essential_indices)
        random.shuffle(candidates)

        for node_idx in candidates:
            x_col = node_idx
            y_col = n + node_idx
            test_cols = sorted(active_cols - {x_col, y_col})

            if len(test_cols) < k1:
                essential_indices.add(node_idx)
                continue

            # Transposed submatrix: rows = active col pairs, cols = failure scenarios
            sub = mat[:, test_cols].T  # (|test_cols|, n)
            n_sub_cols = sub.shape[1]
            if n_sub_cols < k1:
                essential_indices.add(node_idx)
                continue

            # Randomised k-disjunct check (NumPy vectorised)
            is_kd = True
            for _ in range(m_trials):
                trial_cols = rng.choice(n_sub_cols, k1, replace=False)
                trial_sub = sub[:, trial_cols]           # (rows, k1)
                row_sums = trial_sub.sum(axis=1, dtype=np.int32)
                unit_mask = row_sums == 1                # rows that are unit vectors
                for j in range(k1):
                    # Identity row j requires: unit vector with the 1 at position j
                    if not bool(np.any(unit_mask & (trial_sub[:, j] == 1))):
                        is_kd = False
                        break
                if not is_kd:
                    break

            if is_kd:
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


# ── Task 4: batch experiments ─────────────────────────────────────────────────

def _load_records() -> Dict[str, dict]:
    if RECORDS_FILE.exists():
        with open(RECORDS_FILE) as f:
            return json.load(f)
    return {}


def _save_record(
    records: Dict[str, dict],
    dataset_name: str,
    k: int,
    sensor_size: int,
    elapsed: float,
) -> None:
    key = f"{dataset_name}|k={k}"
    records[key] = dict(
        dataset=dataset_name, k=k, sensor_size=sensor_size, elapsed=elapsed
    )
    with open(RECORDS_FILE, "w") as f:
        json.dump(records, f, indent=2)


def _append_raw_result(
    dataset_name: str, k: int, sensor_size: int, elapsed: float
) -> None:
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{dataset_name}\tk={k}\tsensor_size={sensor_size}"
            f"\telapsed={elapsed:.2f}s\n"
        )


def _write_report(records: Dict[str, dict], dataset_names: List[str]) -> None:
    col_w = 10
    k_headers = [f"k={k}" for k in K_VALUES]

    def _row(label: str, values: list) -> str:
        return f"{label:<45}" + "".join(f"{v:>{col_w}}" for v in values)

    header = _row("Dataset", k_headers)
    sep = "-" * len(header)

    def _section(title: str, field: str, fmt: str) -> List[str]:
        lines = [title, sep, header, sep]
        for name in dataset_names:
            vals = []
            for k in K_VALUES:
                rec = records.get(f"{name}|k={k}")
                vals.append(f"{rec[field]:{fmt}}" if rec else "N/A")
            lines.append(_row(name, vals))
        return lines

    report_lines = (
        ["Baseline Algorithm – k-Disjunct Greedy Reduction", "=" * 80, ""]
        + _section("Sensor Set Size", "sensor_size", "d")
        + [""]
        + _section("Run Time (seconds)", "elapsed", ".2f")
        + [""]
    )

    REPORT_FILE.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved → {REPORT_FILE}")


def run_experiments() -> None:
    """Task 4: Run the baseline algorithm on every dataset for every k value."""
    records = _load_records()

    dataset_files = sorted(
        f
        for f in DATASETS_DIR.iterdir()
        if f.suffix in (".txt", ".mtx")
    )
    print(f"Datasets found : {len(dataset_files)}")
    print(f"k values       : {K_VALUES}\n")

    for ds_path in dataset_files:
        name = ds_path.name
        for k in K_VALUES:
            key = f"{name}|k={k}"
            if key in records:
                rec = records[key]
                print(
                    f"  SKIP  {name}  k={k}"
                    f"  (size={rec['sensor_size']}, {rec['elapsed']:.1f}s)"
                )
                continue

            print(f"  RUN   {name}  k={k} ...", end="", flush=True)
            try:
                sensors, elapsed = baseline_sensor_placement(str(ds_path), k)
                size = len(sensors)
                print(f"  size={size}  time={elapsed:.2f}s")
                _save_record(records, name, k, size, elapsed)
                _append_raw_result(name, k, size, elapsed)
            except Exception as exc:
                print(f"  ERROR: {exc}")

    _write_report(records, [f.name for f in dataset_files])


if __name__ == "__main__":
    run_experiments()
