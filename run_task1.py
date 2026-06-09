#!/usr/bin/env python3
"""Task 1: Run gismo experiment on all datasets."""

import json
import os
import sys
import subprocess
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_gismo_output import parse_sensor_set_from_gismo_output

DOCKER_IMAGE = "ubuntu-flask-app"
DATASETS_DIR = Path(__file__).parent / "datasets"
RESULTS_JSONL = DATASETS_DIR / "gismo_results.jsonl"
REPORT_TXT    = DATASETS_DIR / "gismo_report.txt"


def get_running_container_id(image: str) -> str:
    r = subprocess.run(
        ["docker", "ps", "--filter", f"ancestor={image}", "--filter", "status=running", "-q"],
        capture_output=True, text=True
    )
    ids = r.stdout.strip().splitlines()
    if not ids:
        raise RuntimeError(f"No running container found for image '{image}'. Start it first.")
    if len(ids) > 1:
        print(f"  Warning: multiple running containers for '{image}', using first: {ids[0]}")
    return ids[0]


CONTAINER_ID = get_running_container_id(DOCKER_IMAGE)
GISMO_BIN = "/app/gismo/build/gismo"
ENCODE_SCRIPT = "/app/identifying-codes/scripts/encoding/encode_network.py"
PROJECT_DIR_IN_CONTAINER = "/app/identifying-codes"
MAXC = 5000
K_VALUES = [1, 2, 3, 4, 6, 8, 10, 12, 16]


def docker_cp_to(local_path: str, container_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "cp", local_path, f"{CONTAINER_ID}:{container_path}"],
        capture_output=True, text=True
    )


def docker_cp_from(container_path: str, local_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "cp", f"{CONTAINER_ID}:{container_path}", local_path],
        capture_output=True, text=True
    )


def docker_exec(cmd: list[str], env: dict = None) -> subprocess.CompletedProcess:
    base = ["docker", "exec"]
    if env:
        for k, v in env.items():
            base += ["-e", f"{k}={v}"]
    base += [CONTAINER_ID] + cmd
    return subprocess.run(base, capture_output=True, text=True)


def load_existing_results() -> dict[tuple[str, int], dict]:
    """Load previously completed (name, k) pairs from the jsonl cache."""
    cache = {}
    if not RESULTS_JSONL.exists():
        return cache
    with open(RESULTS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cache[(rec["name"], rec["k"])] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return cache


def append_result(rec: dict) -> None:
    """Append one result record to the jsonl file immediately."""
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")


def write_report(all_results: dict[tuple[str, int], dict], dataset_names: list[str]) -> None:
    col_name = 40
    col_k    = 5
    col_rt   = 15
    col_solv = 12
    col_size = 16

    with open(REPORT_TXT, "w") as f:
        header = (f"{'Dataset':<{col_name}} {'k':>{col_k}} {'Run Time (s)':>{col_rt}} "
                  f"{'Is Solvable':>{col_solv}} {'Sensor Set Size':>{col_size}}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i, name in enumerate(dataset_names):
            if i > 0:
                f.write("\n")
            for k in K_VALUES:
                res = all_results.get((name, k))
                if res is None:
                    continue
                rt = f"{res['run_time']:.4f}" if res["run_time"] is not None else "N/A"
                ss = str(res["sensor_set_size"]) if res["sensor_set_size"] is not None else "N/A"
                line = (f"{name:<{col_name}} {k:>{col_k}} {rt:>{col_rt}} "
                        f"{str(res['is_solvable']):>{col_solv}} {ss:>{col_size}}")
                f.write(line + "\n")
                if res.get("error"):
                    f.write(f"  ^ ERROR: {res['error']}\n")


def main():
    dataset_files = sorted([
        f for f in DATASETS_DIR.iterdir()
        if f.suffix in (".mtx", ".txt") and f.is_file()
    ])
    print(f"Found {len(dataset_files)} datasets.")

    cache = load_existing_results()
    skipped = sum(1 for f in dataset_files for k in K_VALUES if (f.stem, k) in cache)
    if skipped:
        print(f"Loaded {len(cache)} cached results ({skipped} (dataset, k) pairs will be skipped).")

    dataset_names = [f.stem for f in dataset_files]
    all_results = dict(cache)

    for dataset_path in dataset_files:
        name = dataset_path.stem
        ext = dataset_path.suffix

        pending_k = [k for k in K_VALUES if (name, k) not in cache]
        if not pending_k:
            print(f"\n{'=' * 60}")
            print(f"Skipping {name}{ext} — all k values already done.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {name}{ext}  (pending k={pending_k})")

        container_input = f"/tmp/ds_{name}{ext}"
        container_gcnf_dir = f"/tmp/gcnf_{name}"

        # ── Step 1: Copy dataset into container (once per dataset) ───
        print("  [1] Copying dataset to container...")
        r = docker_cp_to(str(dataset_path), container_input)
        if r.returncode != 0:
            msg = r.stderr.strip()
            print(f"  ERROR: {msg}")
            for k in pending_k:
                rec = dict(name=name, k=k, run_time=None, is_solvable=False,
                           sensor_set_size=None, error=f"cp failed: {msg}")
                append_result(rec)
                all_results[(name, k)] = rec
            continue

        for k in pending_k:
            print(f"\n  -- k={k} --")
            container_gcnf = f"{container_gcnf_dir}/k{k}/{name}.gcnf"

            # ── Step 2: Encode to gcnf ───────────────────────────────
            print(f"  [2] Encoding network to gcnf (k={k})...")
            r = docker_exec(
                ["python3", ENCODE_SCRIPT,
                 "-n", container_input,
                 "--out_dir", container_gcnf_dir,
                 "--out_file", f"{name}.gcnf",
                 "--encoding", "gis",
                 "--two_step",
                 "-k", str(k)],
                env={"PROJECT_DIR": PROJECT_DIR_IN_CONTAINER}
            )
            if r.returncode != 0:
                msg = (r.stderr or r.stdout).strip()
                print(f"  ERROR encoding: {msg}")
                rec = dict(name=name, k=k, run_time=None, is_solvable=False,
                           sensor_set_size=None, error=f"encode failed: {msg}")
                append_result(rec)
                all_results[(name, k)] = rec
                continue

            # ── Step 3: Run gismo (timed) ────────────────────────────
            print(f"  [3] Running gismo --maxc {MAXC}...")
            t0 = time.perf_counter()
            r = docker_exec([GISMO_BIN, "--maxc", str(MAXC), container_gcnf])
            run_time = time.perf_counter() - t0
            gismo_output = r.stdout
            print(f"  Done in {run_time:.2f}s (exit={r.returncode})")

            # ── Step 4: Parse result ─────────────────────────────────
            print("  [4] Parsing gismo output...")
            is_solvable = False
            sensor_set_size = None
            error = None

            if "c ind " in gismo_output:
                with tempfile.NamedTemporaryFile(suffix=".gcnf", delete=False) as tmp:
                    local_gcnf = tmp.name
                try:
                    cp_r = docker_cp_from(container_gcnf, local_gcnf)
                    if cp_r.returncode != 0:
                        raise RuntimeError(f"cp gcnf back failed: {cp_r.stderr.strip()}")
                    sensor_set = parse_sensor_set_from_gismo_output(gismo_output, local_gcnf)
                    is_solvable = True
                    sensor_set_size = len(sensor_set)
                    print(f"  Solvable. Sensor set size: {sensor_set_size}")
                except Exception as e:
                    error = str(e)
                    print(f"  Parse error: {e}")
                finally:
                    if os.path.exists(local_gcnf):
                        os.unlink(local_gcnf)
            else:
                print(f"  Not solvable within maxc={MAXC}.")

            rec = dict(name=name, k=k, run_time=run_time, is_solvable=is_solvable,
                       sensor_set_size=sensor_set_size, error=error)
            append_result(rec)
            all_results[(name, k)] = rec

    # ── Write final report ────────────────────────────────────────────
    write_report(all_results, dataset_names)
    print(f"\n{'=' * 60}")
    print(f"All done.")
    print(f"  Results cache : {RESULTS_JSONL}")
    print(f"  Report        : {REPORT_TXT}")


if __name__ == "__main__":
    main()
