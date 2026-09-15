#!/usr/bin/env python3
"""Chay Greedy-WOU relaxed voi new_gain tren cac graph da chuan hoa.

Thuc nghiem co dinh ``E = 2`` va chi dung cong thuc ``new_gain`` trong
``greedy_wou_relaxed_E.md``. Moi graph chay trong mot worker rieng voi timeout
8 gio va gioi han RAM 64 GiB. Ket qua duoc ghi vao CSV va co the tiep tuc
(resume) theo cap ``dataset, E``.
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
from collections import Counter
from pathlib import Path


Graph = dict[int, set[int]]
E_VALUES = (3,)
DEFAULT_TIMEOUT_SECONDS = 8 * 60 * 60
DEFAULT_RAM_LIMIT_GB = 64.0
SUPPORTED_SUFFIXES = {".txt", ".mtx", ".edges"}


def _load_batch_support():
    """Tai loader va cac ham gioi han tai nguyen tu thuc nghiem goc."""
    path = Path(__file__).resolve().with_name("greedy-set-cover.py")
    spec = importlib.util.spec_from_file_location(
        "greedy_set_cover_batch_support", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Khong the tai batch harness tu {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def closed_neighborhoods(graph: Graph) -> dict[int, set[int]]:
    """Tra ve N[v], bao gom chinh v, cho moi dinh cua graph."""
    vertices = set(graph)
    neighborhoods: dict[int, set[int]] = {}
    for vertex, neighbors in graph.items():
        unknown_neighbors = neighbors - vertices
        if unknown_neighbors:
            first_unknown = min(unknown_neighbors)
            raise ValueError(
                f"Dinh ke {first_unknown} cua {vertex} khong co trong graph"
            )
        neighborhoods[vertex] = set(neighbors) | {vertex}

    for vertex, neighborhood in neighborhoods.items():
        for neighbor in neighborhood - {vertex}:
            if vertex not in neighborhoods[neighbor]:
                raise ValueError(
                    f"Graph khong vo huong: co {vertex}->{neighbor} nhung thieu "
                    f"{neighbor}->{vertex}"
                )
    return neighborhoods


def combination_table(number_of_vertices: int, E: int) -> list[int]:
    """Tinh chinh xac B[t] = C(t, E + 1) bang Python integer."""
    B = [0] * (number_of_vertices + 1)
    r = E + 1
    if r <= number_of_vertices:
        B[r] = 1
        for size in range(r + 1, number_of_vertices + 1):
            B[size] = B[size - 1] * size // (size - r)
    return B


def candidate_gains(
    groups: list[set[int]],
    undominated: set[int],
    neighborhoods: dict[int, set[int]],
    selected: set[int],
    E: int,
    B: list[int],
) -> dict[int, int]:
    """Tinh new_gain cua moi candidate con lai."""
    gains = {
        candidate: 0 for candidate in sorted(neighborhoods) if candidate not in selected
    }

    for vertex in undominated:
        for candidate in neighborhoods[vertex]:
            if candidate not in selected:
                gains[candidate] += 1

    for group in groups:
        group_size = len(group)
        if group_size <= E:
            continue

        number_of_ones: dict[int, int] = {}
        for vertex in group:
            for candidate in neighborhoods[vertex]:
                if candidate not in selected:
                    number_of_ones[candidate] = number_of_ones.get(candidate, 0) + 1

        for candidate, ones in number_of_ones.items():
            gains[candidate] += B[group_size] - B[ones] - B[group_size - ones]
    return gains


def split_groups(groups: list[set[int]], candidate_column: set[int]) -> list[set[int]]:
    """Tinh partition signature sau khi them mot sensor."""
    refined: list[set[int]] = []
    for group in groups:
        vector_1: set[int] = set()
        vector_0: set[int] = set()
        for vertex in group:
            (vector_1 if vertex in candidate_column else vector_0).add(vertex)
        if vector_1:
            refined.append(vector_1)
        if vector_0:
            refined.append(vector_0)
    return refined


def solve_relaxed_wou(
    adj: Graph,
    E: int = 2,
    trace: bool = False,
) -> dict[str, object]:
    """Giai bai toan mot dinh loi, danh sach nghi van co kich thuoc toi da E."""
    if isinstance(E, bool) or not isinstance(E, int) or E < 1:
        raise ValueError("E phai la so nguyen >= 1")

    vertices = sorted(adj)
    neighborhoods = closed_neighborhoods(adj)

    # Khong xoa closed twins: moi dinh goc van la mot doi tuong nghi van.
    twin_sizes = Counter(frozenset(neighborhoods[vertex]) for vertex in vertices)
    if any(size > E for size in twin_sizes.values()):
        return {
            "status": "infeasible",
            "sensors": None,
            "size": None,
            "max_group_size": None,
            "trace": [],
        }

    B = combination_table(len(vertices), E)
    selected: set[int] = set()
    selected_order: list[int] = []
    undominated = set(vertices)
    groups = [set(vertices)] if vertices else []
    history: list[dict[str, object]] = []

    while undominated or any(len(group) > E for group in groups):
        gains = candidate_gains(
            groups,
            undominated,
            neighborhoods,
            selected,
            E,
            B,
        )
        if not gains:
            raise RuntimeError("Khong con candidate khi bai toan chua hoan tat")

        candidate = min(gains, key=lambda vertex: (-gains[vertex], vertex))
        if gains[candidate] <= 0:
            raise RuntimeError("Gain bang 0 tren mot instance kha thi chua hoan tat")

        if trace:
            history.append({"gains": dict(gains), "selected": candidate})
        selected.add(candidate)
        selected_order.append(candidate)
        candidate_column = neighborhoods[candidate]
        undominated.difference_update(candidate_column)
        # Moi group, ke ca group da co size <= E, deu phai duoc cap nhat.
        groups = split_groups(groups, candidate_column)

    return {
        "status": "ok",
        "sensors": selected_order,
        "size": len(selected_order),
        "max_group_size": max(map(len, groups), default=0),
        "trace": history,
    }


def _validate_relaxed_code_details(
    graph: Graph, sensors: set[int] | list[int], E: int
) -> tuple[bool, str, int]:
    """Kiem tra doc lap domination va kich thuoc moi lop signature."""
    if isinstance(E, bool) or not isinstance(E, int) or E < 1:
        return False, "E phai la so nguyen >= 1", 0

    code = set(sensors)
    if not code <= graph.keys():
        return False, "Code chua dinh khong thuoc graph", 0

    neighborhoods = closed_neighborhoods(graph)
    signature_sizes: Counter[frozenset[int]] = Counter()
    for vertex in sorted(graph):
        signature = frozenset(neighborhoods[vertex] & code)
        if not signature:
            return False, f"Dinh {vertex} khong duoc dominate", 0
        signature_sizes[signature] += 1

    max_group_size = max(signature_sizes.values(), default=0)
    if max_group_size > E:
        return (
            False,
            f"Co danh sach nghi van kich thuoc {max_group_size}, vuot E={E}",
            max_group_size,
        )
    return (
        True,
        f"Thoa man domination va moi danh sach nghi van co kich thuoc <= {E}",
        max_group_size,
    )


def validate_relaxed_code(adj: Graph, sensors: set[int] | list[int], E: int) -> bool:
    """Kiem tra nghiem tu signature thuc, doc lap voi trang thai cua solver."""
    valid, _, _ = _validate_relaxed_code_details(adj, sensors, E)
    return valid


RESULT_FIELDS = [
    "dataset",
    "E",
    "status",
    "vertices",
    "edges",
    "code_size",
    "max_group_size",
    "elapsed_seconds",
    "peak_ram_bytes",
    "solution_file",
    "detail",
]


def run_worker(
    graph_path: Path,
    result_path: Path,
    solution_directory: Path,
    E: int,
    batch_support,
) -> int:
    """Chay va kiem chung mot cau hinh trong worker rieng."""
    start = time.perf_counter()
    result: dict[str, object] = {
        "dataset": graph_path.name,
        "E": E,
        "status": "INVALID",
        "vertices": "",
        "edges": "",
        "code_size": "",
        "max_group_size": "",
        "elapsed_seconds": "",
        "peak_ram_bytes": "",
        "solution_file": "",
        "detail": "",
    }
    try:
        graph = batch_support.read_graph(graph_path)
        result["vertices"] = len(graph)
        result["edges"] = sum(map(len, graph.values())) // 2

        algorithm_start = time.perf_counter()
        solution = solve_relaxed_wou(graph, E=E)
        algorithm_elapsed = time.perf_counter() - algorithm_start

        if solution["status"] == "infeasible":
            result.update(
                status="INFEASIBLE",
                elapsed_seconds=f"{algorithm_elapsed:.9f}",
                detail=f"Ton tai lop closed twins co nhieu hon E={E} dinh",
            )
        else:
            sensors = solution["sensors"]
            if not isinstance(sensors, list):
                raise RuntimeError("Solver khong tra ve danh sach sensor")
            valid, message, max_group_size = _validate_relaxed_code_details(
                graph, sensors, E
            )

            solution_directory.mkdir(parents=True, exist_ok=True)
            solution_path = solution_directory / (
                f"{graph_path.name}.E{E}.new_gain.code.txt"
            )
            temporary_solution = solution_path.with_name(solution_path.name + ".tmp")
            temporary_solution.write_text(
                "\n".join(map(str, sorted(sensors))) + "\n", encoding="utf-8"
            )
            temporary_solution.replace(solution_path)

            result.update(
                status="VALID" if valid else "INVALID",
                code_size=len(sensors),
                max_group_size=max_group_size,
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
    E: int,
) -> dict[str, object]:
    """Chay mot cau hinh trong process rieng va gioi han wall-time/RAM."""
    result_file_handle = tempfile.NamedTemporaryFile(
        prefix="relaxed-greedy-result-", suffix=".json", delete=False
    )
    result_path = Path(result_file_handle.name)
    result_file_handle.close()
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
        "--E",
        str(E),
    ]
    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
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
        elapsed = time.perf_counter() - start
        if elapsed > timeout_seconds:
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
                except psutil.Error:
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
        return {
            "dataset": graph_path.name,
            "E": E,
            "status": limit_status,
            "vertices": "",
            "edges": "",
            "code_size": "",
            "max_group_size": "",
            "elapsed_seconds": f"{time.perf_counter() - start:.9f}",
            "peak_ram_bytes": observed_peak_ram,
            "solution_file": "",
            "detail": (
                f"Vuot timeout {timeout_seconds:g} giay"
                if limit_status == "TIMEOUT"
                else f"Vuot gioi han RAM {ram_limit_bytes} bytes"
            ),
        }

    _, stderr = process.communicate()
    try:
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            reported_peak = result.get("peak_ram_bytes") or 0
            result["peak_ram_bytes"] = max(int(reported_peak), observed_peak_ram)
            if process.returncode:
                worker_error = stderr.strip()[-1000:]
                if process.returncode in {
                    -signal.SIGKILL,
                    -signal.SIGABRT,
                    -signal.SIGSEGV,
                }:
                    result["status"] = "RAM_LIMITED"
                if worker_error:
                    old_detail = str(result.get("detail") or "")
                    result["detail"] = f"{old_detail} | {worker_error}".strip(" |")
            return result

        detail = stderr.strip()[-1000:] or f"worker exit code {process.returncode}"
        killed_for_memory = process.returncode in {
            -signal.SIGKILL,
            -signal.SIGABRT,
            -signal.SIGSEGV,
        }
        return {
            "dataset": graph_path.name,
            "E": E,
            "status": "RAM_LIMITED" if killed_for_memory else "INVALID",
            "vertices": "",
            "edges": "",
            "code_size": "",
            "max_group_size": "",
            "elapsed_seconds": f"{time.perf_counter() - start:.9f}",
            "peak_ram_bytes": observed_peak_ram,
            "solution_file": "",
            "detail": detail,
        }
    finally:
        result_path.unlink(missing_ok=True)


def completed_runs(csv_path: Path) -> set[tuple[str, int]]:
    """Doc cac cau hinh da co ket qua de resume chinh xac."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required_fields = {"dataset", "E", "status"}
        if reader.fieldnames is None or not required_fields <= set(reader.fieldnames):
            raise ValueError(
                f"CSV {csv_path} thieu cac cot bat buoc: "
                + ", ".join(sorted(required_fields))
            )
        completed: set[tuple[str, int]] = set()
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            e_text = (row.get("E") or "").strip()
            status = (row.get("status") or "").strip()
            if dataset and e_text and status:
                completed.add((dataset, int(e_text)))
        return completed


def append_csv_result(csv_path: Path, result: dict[str, object]) -> None:
    """Them mot ket qua va flush ngay de an toan khi batch bi dung."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in RESULT_FIELDS})
        output.flush()


def main() -> None:
    script_path = Path(__file__).resolve()
    project_directory = script_path.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graphs",
        nargs="*",
        type=Path,
        help="graph tuy chon; mac dinh chay tat ca standardized_dataset/",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--ram-limit-gb", type=float, default=DEFAULT_RAM_LIMIT_GB)
    parser.add_argument("--E", type=int, choices=E_VALUES, default=E_VALUES[0])
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_path.parent / "greedy-wou-relaxed-E-results.csv",
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=script_path.parent / "greedy-wou-relaxed-E-solutions",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-ram-bytes", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        batch_support = _load_batch_support()
    except (OSError, ImportError) as exc:
        parser.exit(1, f"Loi tai batch harness: {exc}\n")

    if args.worker is not None:
        if args.result_file is None or args.worker_ram_bytes is None:
            parser.error("--worker requires --result-file and --worker-ram-bytes")
        batch_support._set_ram_limit(args.worker_ram_bytes)
        raise SystemExit(
            run_worker(
                args.worker,
                args.result_file,
                args.solution_dir,
                args.E,
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

    configurations = [(graph_path, args.E) for graph_path in graph_paths]
    try:
        already_completed = completed_runs(args.output_csv)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(f"Khong the doc CSV ket qua cu: {exc}")
    pending = [
        configuration
        for configuration in configurations
        if (configuration[0].name, configuration[1]) not in already_completed
    ]

    print(
        f"Tim thay {len(graph_paths)} graph, {len(configurations)} cau hinh; "
        f"da co ket qua={len(configurations) - len(pending)}, con lai={len(pending)}. "
        f"E={args.E}; mode=new_gain; timeout={args.timeout:g}s; "
        f"RAM limit={args.ram_limit_gb:g} GiB/cau hinh. "
        "Thu tu graph: so dinh tang dan."
    )
    if not pending:
        print(f"Khong co cau hinh nao can chay. Ket qua: {args.output_csv}")
        return

    ram_limit_bytes = int(args.ram_limit_gb * 1024**3)
    for index, (graph_path, E) in enumerate(pending, start=1):
        result = run_dataset_process(
            script_path,
            graph_path.resolve(),
            args.solution_dir.resolve(),
            args.timeout,
            ram_limit_bytes,
            E,
        )
        append_csv_result(args.output_csv, result)
        print(
            f"[{index:03d}/{len(pending):03d}] {graph_path.name} | E={E} | "
            f"mode=new_gain: STATUS={result['status']} | "
            f"time={result['elapsed_seconds']}s | "
            f"code_size={result['code_size'] or 'N/A'}"
        )
    print(f"Da ghi tong ket: {args.output_csv}")


if __name__ == "__main__":
    main()
