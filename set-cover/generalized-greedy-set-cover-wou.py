#!/usr/bin/env python3
"""Chay generalized greedy without-universe voi moi k trong K.

Mac dinh script chay toan bo 50 graph trong ``standardized_dataset/`` theo
so dinh tang dan. Moi graph chay trong mot worker rieng voi timeout 8 gio va
gioi han RAM 64 GiB. Voi moi gia tri trong K, script chay lai toan bo graph.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType


K = [2]
ALGORITHM_VERSION = "or-observation-v2"
EXPECTED_DATASET_COUNT = 50
DEFAULT_TIMEOUT_SECONDS = 8 * 60 * 60
DEFAULT_RAM_LIMIT_GB = 64.0
SUPPORTED_SUFFIXES = {".txt", ".mtx", ".edges"}

RESULT_FIELDS = [
    "algorithm_version",
    "dataset",
    "status",
    "vertices",
    "edges",
    "expanded_vertices",
    "expanded_edges",
    "total_constraints",
    "code_size",
    "elapsed_seconds",
    "peak_ram_bytes",
    "solution_file",
    "detail",
]


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Khong the tao module spec tu {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_algorithm() -> ModuleType:
    """Tai phan thuat toan generalized tu file sibling ``-wou-x.py``."""
    path = Path(__file__).resolve().with_name(
        "generalized-greedy-set-cover-wou-x.py"
    )
    return _load_module(path, "generalized_greedy_set_cover_wou_x")


def load_batch_support() -> ModuleType:
    """Tai parser graph va cac helper resource tu greedy-set-cover.py."""
    path = Path(__file__).resolve().with_name("greedy-set-cover.py")
    return _load_module(path, "greedy_set_cover_batch_support")


def result_template(graph_path: Path) -> dict[str, object]:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "dataset": graph_path.name,
        "status": "INVALID",
        "vertices": "",
        "edges": "",
        "expanded_vertices": "",
        "expanded_edges": "",
        "total_constraints": "",
        "code_size": "",
        "elapsed_seconds": "",
        "peak_ram_bytes": "",
        "solution_file": "",
        "detail": "",
    }


def temp_directory_for_k(project_directory: Path, k: int) -> Path:
    """Tao scratch directory ``temp/k<k>/`` dung cho parent va worker."""
    directory = project_directory / "temp" / f"k{k}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_worker(
    graph_path: Path,
    result_path: Path,
    solution_directory: Path,
    k: int,
    algorithm: ModuleType,
    batch_support: ModuleType,
) -> int:
    """Chay thuat toan va evaluate mot graph trong worker rieng."""
    start = time.perf_counter()
    result = result_template(graph_path)
    try:
        graph = batch_support.read_graph(graph_path)
        result["vertices"] = len(graph)
        result["edges"] = sum(map(len, graph.values())) // 2
        (
            result["expanded_vertices"],
            result["expanded_edges"],
            result["total_constraints"],
        ) = algorithm.expanded_graph_statistics(graph, k)

        algorithm_start = time.perf_counter()
        code = algorithm.generalized_id_greedy_without_universe_with_t0(
            graph, k
        )
        algorithm_elapsed = time.perf_counter() - algorithm_start
        valid, message = algorithm.evaluate_output(graph, code, k)

        solution_directory.mkdir(parents=True, exist_ok=True)
        solution_path = solution_directory / f"{graph_path.name}.code.txt"
        temporary_solution = solution_path.with_name(solution_path.name + ".tmp")
        temporary_solution.write_text(
            "\n".join(map(str, sorted(code))) + "\n", encoding="utf-8"
        )
        temporary_solution.replace(solution_path)

        result.update(
            status="VALID" if valid else "INVALID",
            code_size=len(code),
            elapsed_seconds=f"{algorithm_elapsed:.9f}",
            solution_file=str(solution_path),
            detail=message,
        )
    except MemoryError as exc:
        result.update(
            status="RAM_LIMITED",
            elapsed_seconds=f"{time.perf_counter() - start:.9f}",
            detail=str(exc) or "Vuot gioi han RAM",
        )
    except Exception as exc:
        result.update(
            status="INVALID",
            elapsed_seconds=f"{time.perf_counter() - start:.9f}",
            detail=f"{type(exc).__name__}: {exc}",
        )
    finally:
        result["peak_ram_bytes"] = batch_support._peak_ram_bytes()
        batch_support._write_json_atomic(result_path, result)
    return 0


def run_dataset_process(
    script_path: Path,
    graph_path: Path,
    solution_directory: Path,
    timeout_seconds: float,
    ram_limit_bytes: int,
    k: int,
) -> dict[str, object]:
    """Chay mot graph cach ly va theo doi timeout/RAM cua ca process tree."""
    project_directory = script_path.parent.parent
    temp_directory = temp_directory_for_k(project_directory, k)
    handle = tempfile.NamedTemporaryFile(
        prefix="generalized-greedy-result-",
        suffix=".json",
        dir=temp_directory,
        delete=False,
    )
    result_path = Path(handle.name)
    handle.close()
    result_path.unlink(missing_ok=True)

    command = [
        sys.executable,
        str(script_path),
        "--worker",
        str(graph_path),
        "--result-file",
        str(result_path),
        "--solution-dir",
        str(solution_directory),
        "--worker-ram-bytes",
        str(ram_limit_bytes),
        "--worker-k",
        str(k),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    start = time.perf_counter()

    try:
        import psutil
    except ImportError as exc:
        process.kill()
        process.communicate()
        result_path.unlink(missing_ok=True)
        raise RuntimeError("Can cai psutil de gioi han RAM cua worker") from exc

    worker = psutil.Process(process.pid)
    observed_peak_ram = 0
    limit_status = ""
    while process.poll() is None:
        if time.perf_counter() - start > timeout_seconds:
            limit_status = "TIMEOUT"
            break
        try:
            try:
                processes = [worker, *worker.children(recursive=True)]
            except (psutil.Error, OSError):
                processes = [worker]
            current_ram = 0
            for monitored_process in processes:
                try:
                    current_ram += monitored_process.memory_info().rss
                except (psutil.Error, OSError):
                    pass
            observed_peak_ram = max(observed_peak_ram, current_ram)
        except (psutil.Error, OSError):
            pass
        if observed_peak_ram > ram_limit_bytes:
            limit_status = "RAM_LIMITED"
            break
        time.sleep(0.2)

    if limit_status:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        result_path.unlink(missing_ok=True)
        result = result_template(graph_path)
        result.update(
            status=limit_status,
            elapsed_seconds=f"{time.perf_counter() - start:.9f}",
            peak_ram_bytes=observed_peak_ram,
            detail=(
                f"Vuot timeout {timeout_seconds:g} giay"
                if limit_status == "TIMEOUT"
                else f"Vuot gioi han RAM {ram_limit_bytes} bytes"
            ),
        )
        return result

    _, stderr = process.communicate()
    try:
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["peak_ram_bytes"] = max(
                int(result.get("peak_ram_bytes") or 0), observed_peak_ram
            )
            if process.returncode:
                worker_error = stderr.strip()[-1000:]
                result["elapsed_seconds"] = result.get("elapsed_seconds") or (
                    f"{time.perf_counter() - start:.9f}"
                )
                if process.returncode == -signal.SIGTERM:
                    result["status"] = "TIMEOUT"
                elif process.returncode in {
                    -signal.SIGKILL,
                    -signal.SIGABRT,
                    -signal.SIGSEGV,
                }:
                    result["status"] = "RAM_LIMITED"
                if worker_error:
                    old_detail = str(result.get("detail") or "")
                    result["detail"] = f"{old_detail} | {worker_error}".strip(" |")
            return result

        result = result_template(graph_path)
        killed_for_memory = process.returncode in {
            -signal.SIGKILL,
            -signal.SIGABRT,
            -signal.SIGSEGV,
        }
        result.update(
            status="RAM_LIMITED" if killed_for_memory else "INVALID",
            elapsed_seconds=f"{time.perf_counter() - start:.9f}",
            peak_ram_bytes=observed_peak_ram,
            detail=stderr.strip()[-1000:] or f"worker exit code {process.returncode}",
        )
        return result
    finally:
        result_path.unlink(missing_ok=True)


def completed_datasets(csv_path: Path, retry_invalid: bool = False) -> set[str]:
    """Lay cac dataset da xong trong CSV cua mot gia tri k."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"dataset", "status"} <= set(
            reader.fieldnames
        ):
            raise ValueError(f"CSV {csv_path} thieu cot dataset/status")

        completed: set[str] = set()
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            status = (row.get("status") or "").strip()
            version = (row.get("algorithm_version") or "").strip()
            if (
                version == ALGORITHM_VERSION
                and dataset
                and status
                and not (retry_invalid and status == "INVALID")
            ):
                completed.add(dataset)
        return completed


def append_result(csv_path: Path, result: dict[str, object]) -> None:
    ensure_result_schema(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in RESULT_FIELDS})
        output.flush()


def ensure_result_schema(csv_path: Path) -> None:
    """Nang CSV cu len schema co thong ke graph mo rong truoc khi append."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames == RESULT_FIELDS:
            return
        if reader.fieldnames is None or not {"dataset", "status"} <= set(
            reader.fieldnames
        ):
            raise ValueError(f"CSV {csv_path} thieu cot dataset/status")
        rows = list(reader)

    temporary = csv_path.with_name(csv_path.name + ".schema.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
    temporary.replace(csv_path)


def remove_results(csv_path: Path, datasets: set[str]) -> None:
    """Xoa dong INVALID truoc khi retry de tranh trung dataset."""
    if not datasets or not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        rows = [row for row in reader if row.get("dataset") not in datasets]

    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
    temporary.replace(csv_path)


def result_csv_for_k(base_path: Path, k: int) -> Path:
    """Tao CSV rieng theo model version va k de khong tron ket qua cu."""
    suffix = base_path.suffix or ".csv"
    stem = base_path.stem if base_path.suffix else base_path.name
    return base_path.with_name(f"{stem}-{ALGORITHM_VERSION}-k{k}{suffix}")


def main() -> None:
    script_path = Path(__file__).resolve()
    project_directory = script_path.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graphs",
        nargs="*",
        type=Path,
        help="graph tuy chon; mac dinh chay 50 graph standardized_dataset/",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--ram-limit-gb", type=float, default=DEFAULT_RAM_LIMIT_GB)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_path.parent / "generalized-greedy-set-cover-wou-results.csv",
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=script_path.parent / "generalized-greedy-set-cover-wou-solutions",
    )
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="chay lai cac dataset co STATUS=INVALID",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-ram-bytes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-k", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        algorithm = load_algorithm()
        batch_support = load_batch_support()
    except (OSError, ImportError) as exc:
        parser.exit(1, f"Loi tai module: {exc}\n")

    if args.worker is not None:
        if (
            args.result_file is None
            or args.worker_ram_bytes is None
            or args.worker_k is None
        ):
            parser.error(
                "--worker requires --result-file, --worker-ram-bytes, --worker-k"
            )
        if args.worker_k not in K:
            parser.error(f"--worker-k phai thuoc {K}")
        batch_support._set_ram_limit(args.worker_ram_bytes)
        raise SystemExit(
            run_worker(
                args.worker,
                args.result_file,
                args.solution_dir,
                args.worker_k,
                algorithm,
                batch_support,
            )
        )

    if args.timeout <= 0:
        parser.error("--timeout phai lon hon 0")
    if args.ram_limit_gb <= 0:
        parser.error("--ram-limit-gb phai lon hon 0")

    dataset_directory = project_directory / "standardized_dataset"
    graph_paths = args.graphs or [
        path
        for path in dataset_directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not graph_paths:
        parser.error(f"Khong tim thay graph trong {dataset_directory}")
    if not args.graphs and len(graph_paths) != EXPECTED_DATASET_COUNT:
        parser.error(
            f"Can {EXPECTED_DATASET_COUNT} graph trong {dataset_directory}, "
            f"tim thay {len(graph_paths)}"
        )
    try:
        graph_paths = sorted(
            graph_paths,
            key=lambda path: (
                batch_support.declared_vertex_count(path),
                path.name.lower(),
            ),
        )
    except (OSError, ValueError) as exc:
        parser.error(f"Khong the sap xep graph theo so dinh: {exc}")

    ram_limit_bytes = int(args.ram_limit_gb * 1024**3)
    for k_index, k in enumerate(K, start=1):
        output_csv = result_csv_for_k(args.output_csv, k)
        solution_directory = args.solution_dir / ALGORITHM_VERSION / f"k{k}"
        try:
            ensure_result_schema(output_csv)
            done = completed_datasets(
                output_csv, retry_invalid=args.retry_invalid
            )
        except (OSError, ValueError, csv.Error) as exc:
            parser.error(f"Khong the doc CSV ket qua cu: {exc}")

        pending = [path for path in graph_paths if path.name not in done]
        if args.retry_invalid:
            remove_results(output_csv, {path.name for path in pending})

        print(
            f"[K {k_index}/{len(K)}] graph={len(graph_paths)}; "
            f"da co ket qua={len(graph_paths) - len(pending)}; "
            f"con lai={len(pending)}; k={k}; timeout={args.timeout:g}s; "
            f"RAM={args.ram_limit_gb:g} GiB/graph."
        )
        if not pending:
            print(f"Khong con graph voi k={k}. Ket qua: {output_csv}")
            continue

        for index, graph_path in enumerate(pending, start=1):
            result = run_dataset_process(
                script_path,
                graph_path.resolve(),
                solution_directory.resolve(),
                args.timeout,
                ram_limit_bytes,
                k,
            )
            append_result(output_csv, result)
            print(
                f"[k={k} | {index:02d}/{len(pending):02d}] "
                f"{graph_path.name}: STATUS={result['status']} | "
                f"time={result['elapsed_seconds']}s | "
                f"code_size={result['code_size'] or 'N/A'}"
            )
            if result["status"] != "VALID":
                detail = str(result.get("detail") or "Khong co thong tin loi")
                print(f"  detail: {detail}")
        print(f"Da ghi tong ket k={k}: {output_csv}")


if __name__ == "__main__":
    main()
