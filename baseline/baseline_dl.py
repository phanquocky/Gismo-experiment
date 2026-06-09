"""
Baseline algorithm for sensor placement using (d,l)-disjunctive matrix reduction.

Task 5: Greedy algorithm that removes node pairs (x_i, y_i) from the matrix
        while maintaining (d,l)-disjunctiveness.
Task 6: Batch experiments across all datasets for the (d,l) pairs defined in
        DL_PAIRS below.
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
RESULTS_FILE = _BASE / "Using_DL_Disjunct_result.txt"
RECORDS_FILE = _BASE / "Using_DL_Disjunct_records.json"
REPORT_FILE  = _BASE / "Using_DL_Disjunct_report.txt"

# (d, l) pairs from the spec.
# Note: (10, 17) in the original spec has l > d (violates l <= d), treated as (10, 7).
DL_PAIRS: List[Tuple[int, int]] = [
    (1, 1), (2, 2), (3, 3), (4, 4), (6, 5), (8, 6), (10, 7), (12, 8), (16, 9),
]

# ── Task 5: greedy sensor placement ──────────────────────────────────────────

def dl_sensor_placement(
    network_file: str,
    d: int,
    l: int,
    m_trials: int = 10_000,
    seed: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    Greedy baseline for finding a (near-)minimal sensor set.

    Speedups vs. the naive version:
      * NumPy arrays — column extraction and transpose are O(1) views.
      * Vectorised witness-row check — replaces Python loops in has_witness_row.
      * Random (S, L) sampling per trial — replaces the O(C(d+l, l)) exhaustive
        enumeration of L-subsets per trial. Equivalent probabilistic coverage
        with O(1) work per trial instead of exponential.
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
    mat = np.array(matrix[1:], dtype=np.int8)  # (n, 2n)

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

            # Transposed submatrix: rows = active col pairs, cols = failure scenarios
            sub = mat[:, test_cols].T  # (|test_cols|, n)
            n_sub_cols = sub.shape[1]
            if n_sub_cols < dl:
                essential_indices.add(node_idx)
                continue

            # Randomised (d,l)-disjunct check (NumPy vectorised).
            # Each trial picks a random S of size d+l AND a random L of size l
            # (instead of enumerating all C(d+l,l) partitions per trial).
            # This reduces per-trial cost from O(C(d+l,l) × t) to O(t), while
            # still sampling uniformly over all (S, L) counterexample candidates.
            is_dl = True
            for _ in range(m_trials):
                S = rng.choice(n_sub_cols, dl, replace=False)

                # Random L of size l drawn from S's local indices [0..dl)
                L_local = rng.choice(dl, l, replace=False)
                L_mask = np.zeros(dl, dtype=bool)
                L_mask[L_local] = True
                D_mask = ~L_mask

                D_sub = sub[:, S[D_mask]]  # (t, d)
                L_sub = sub[:, S[L_mask]]  # (t, l)

                # Witness row: all D-cols are 0 AND at least one L-col is 1
                d_ok = np.all(D_sub == 0, axis=1)   # (t,)
                l_ok = np.any(L_sub == 1, axis=1)   # (t,)

                if not bool(np.any(d_ok & l_ok)):
                    is_dl = False
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


# ── Task 6: batch experiments ─────────────────────────────────────────────────

def _load_records() -> Dict[str, dict]:
    if RECORDS_FILE.exists():
        with open(RECORDS_FILE) as f:
            return json.load(f)
    return {}


def _save_record(
    records: Dict[str, dict],
    dataset_name: str,
    d: int,
    l: int,
    sensor_size: int,
    elapsed: float,
) -> None:
    key = f"{dataset_name}|d={d}|l={l}"
    records[key] = dict(
        dataset=dataset_name, d=d, l=l, sensor_size=sensor_size, elapsed=elapsed
    )
    with open(RECORDS_FILE, "w") as f:
        json.dump(records, f, indent=2)


def _append_raw_result(
    dataset_name: str, d: int, l: int, sensor_size: int, elapsed: float
) -> None:
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{dataset_name}\td={d}\tl={l}\tsensor_size={sensor_size}"
            f"\telapsed={elapsed:.2f}s\n"
        )


def _write_report(records: Dict[str, dict], dataset_names: List[str]) -> None:
    col_w = 10
    dl_headers = [f"d={d},l={l}" for d, l in DL_PAIRS]

    def _row(label: str, values: list) -> str:
        return f"{label:<45}" + "".join(f"{v:>{col_w}}" for v in values)

    header = _row("Dataset", dl_headers)
    sep = "-" * len(header)

    def _section(title: str, field: str, fmt: str) -> List[str]:
        lines = [title, sep, header, sep]
        for name in dataset_names:
            vals = []
            for d, l in DL_PAIRS:
                rec = records.get(f"{name}|d={d}|l={l}")
                vals.append(f"{rec[field]:{fmt}}" if rec else "N/A")
            lines.append(_row(name, vals))
        return lines

    report_lines = (
        ["Baseline Algorithm – (d,l)-Disjunct Greedy Reduction", "=" * 80, ""]
        + _section("Sensor Set Size", "sensor_size", "d")
        + [""]
        + _section("Run Time (seconds)", "elapsed", ".2f")
        + [""]
    )

    REPORT_FILE.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved → {REPORT_FILE}")


def run_experiments() -> None:
    """Task 6: Run the DL baseline on every dataset for every (d,l) pair."""
    records = _load_records()

    dataset_files = sorted(
        f for f in DATASETS_DIR.iterdir() if f.suffix in (".txt", ".mtx")
    )
    print(f"Datasets found : {len(dataset_files)}")
    print(f"(d,l) pairs    : {DL_PAIRS}\n")

    for ds_path in dataset_files:
        name = ds_path.name
        for d, l in DL_PAIRS:
            key = f"{name}|d={d}|l={l}"
            if key in records:
                rec = records[key]
                print(
                    f"  SKIP  {name}  d={d}  l={l}"
                    f"  (size={rec['sensor_size']}, {rec['elapsed']:.1f}s)"
                )
                continue

            print(f"  RUN   {name}  d={d}  l={l} ...", end="", flush=True)
            try:
                sensors, elapsed = dl_sensor_placement(str(ds_path), d, l)
                size = len(sensors)
                print(f"  size={size}  time={elapsed:.2f}s")
                _save_record(records, name, d, l, size, elapsed)
                _append_raw_result(name, d, l, size, elapsed)
            except Exception as exc:
                print(f"  ERROR: {exc}")

    _write_report(records, [f.name for f in dataset_files])


if __name__ == "__main__":
    run_experiments()
