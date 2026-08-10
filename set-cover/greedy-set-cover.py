#!/usr/bin/env python3
"""Chay ID-Greedy tren cac graph da chuan hoa va kiem tra Identifying Code.

Mac dinh script duyet toan bo ``standardized_dataset/``. Moi graph chay trong
mot worker rieng voi timeout 8 gio va gioi han RAM 64 GiB. Tong ket duoc ghi
vao CSV; nghiem cua cac lan chay hoan tat duoc ghi thanh file rieng.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Hashable


Constraint = tuple[str, int, int] | tuple[str, int]
DEFAULT_TIMEOUT_SECONDS = 8 * 60 * 60
DEFAULT_RAM_LIMIT_GB = 64.0
SUPPORTED_SUFFIXES = {".txt", ".mtx", ".edges"}


def declared_vertex_count(path: Path) -> int:
    """Read only the header needed to sort standardized graphs by |V|."""
    with path.open("r", encoding="utf-8") as graph_file:
        if path.suffix.lower() == ".mtx":
            for raw_line in graph_file:
                line = raw_line.strip()
                if not line or line.startswith("%"):
                    continue
                fields = line.split()
                if len(fields) >= 2:
                    rows, columns = int(fields[0]), int(fields[1])
                    if rows != columns:
                        raise ValueError(f"Matrix khong vuong: {path}")
                    return rows
        else:
            for raw_line in graph_file:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(("%", "#")):
                    fields = line.lstrip("%#").split()
                    if len(fields) == 3:
                        _, rows, columns = map(int, fields)
                        if rows == columns:
                            return rows
                    continue
                break
    raise ValueError(f"Khong tim thay so dinh khai bao trong {path}")


def _read_edge_list(lines: list[str]) -> dict[int, set[int]]:
    """Doc cac dong edge-list."""
    graph: dict[int, set[int]] = {}
    declared_vertices: int | None = None
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("%"):
            fields = line.lstrip("%").split()
            if len(fields) == 3:
                try:
                    _, rows, columns = map(int, fields)
                except ValueError:
                    continue
                if rows == columns:
                    declared_vertices = rows
            continue

        fields = line.split()
        if len(fields) not in (1, 2):
            raise ValueError(
                f"Dong {line_number}: can 1 dinh hoac 2 dinh, nhan duoc {line!r}"
            )
        try:
            vertices = [int(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"Dong {line_number}: nhan dinh phai la so nguyen") from exc

        u = vertices[0]
        graph.setdefault(u, set())
        if len(vertices) == 2:
            v = vertices[1]
            graph.setdefault(v, set())
            if u != v:
                graph[u].add(v)
                graph[v].add(u)

    if declared_vertices is not None:
        if graph and max(graph) > declared_vertices:
            raise ValueError("Nhan dinh vuot qua kich thuoc graph da khai bao")
        for vertex in range(1, declared_vertices + 1):
            graph.setdefault(vertex, set())

    if not graph:
        raise ValueError("Graph rong")
    return graph


def _read_matrix_market(lines: list[str]) -> dict[int, set[int]]:
    """Doc Matrix Market coordinate symmetric, bo qua gia tri trong cot 3."""
    data_lines = [line.strip() for line in lines[1:] if line.strip() and not line.lstrip().startswith("%")]
    if not data_lines:
        raise ValueError("File Matrix Market thieu dong kich thuoc")

    dimensions = data_lines[0].split()
    if len(dimensions) < 3:
        raise ValueError("Dong kich thuoc Matrix Market khong hop le")
    try:
        number_of_rows, number_of_columns, declared_entries = map(int, dimensions[:3])
    except ValueError as exc:
        raise ValueError("Kich thuoc Matrix Market phai la so nguyen") from exc
    if number_of_rows != number_of_columns:
        raise ValueError("Identifying Code yeu cau ma tran vuong")

    graph = {vertex: set() for vertex in range(1, number_of_rows + 1)}
    actual_entries = 0
    for entry_number, line in enumerate(data_lines[1:], start=1):
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Entry Matrix Market {entry_number} khong hop le")
        try:
            u, v = map(int, fields[:2])
        except ValueError as exc:
            raise ValueError(f"Entry Matrix Market {entry_number} khong hop le") from exc
        if not (1 <= u <= number_of_rows and 1 <= v <= number_of_rows):
            raise ValueError(f"Entry Matrix Market {entry_number} vuot ngoai kich thuoc")
        actual_entries += 1
        if u != v:
            graph[u].add(v)
            graph[v].add(u)

    if actual_entries != declared_entries:
        raise ValueError(
            f"Matrix Market khai bao {declared_entries} entries, doc duoc {actual_entries}"
        )
    return graph


def read_graph(path: str | Path) -> dict[int, set[int]]:
    """Doc edge-list hoac Matrix Market va tra ve danh sach ke."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    first_line = next((line.strip() for line in lines if line.strip()), "")
    if first_line.lstrip("%").lower().startswith("matrixmarket"):
        return _read_matrix_market(lines)
    return _read_edge_list(lines)


def closed_neighborhoods(graph: dict[int, set[int]]) -> dict[int, set[int]]:
    """Buoc 1: tinh I(v) = N[v] cho moi v."""
    return {vertex: neighbors | {vertex} for vertex, neighbors in graph.items()}


def build_set_cover(
    neighborhoods: dict[int, set[int]],
) -> tuple[set[Constraint], dict[int, set[Constraint]]]:
    """Buoc 2: tao universe va distinguishing set delta_c.

    Ngoai constraint ('pair', u, v), universe co ('dom', v) de bao dam moi
    identifying set khac rong. delta_c cover ('dom', v) iff c thuoc N[v].
    """
    vertices = sorted(neighborhoods)
    universe: set[Constraint] = {("dom", vertex) for vertex in vertices}
    deltas: dict[int, set[Constraint]] = {vertex: set() for vertex in vertices}

    for vertex in vertices:
        for codeword in neighborhoods[vertex]:
            deltas[codeword].add(("dom", vertex))

    for index, u in enumerate(vertices):
        for v in vertices[index + 1 :]:
            constraint: Constraint = ("pair", u, v)
            universe.add(constraint)
            for codeword in neighborhoods[u] ^ neighborhoods[v]:
                deltas[codeword].add(constraint)

    return universe, deltas


def greedy_set_cover(
    universe: set[Hashable], subsets: dict[int, set[Hashable]]
) -> set[int]:
    """Chon lap lai tap phu duoc nhieu constraint chua duoc phu nhat."""
    uncovered = set(universe)
    selected: set[int] = set()

    while uncovered:
        candidate, gain = max(
            ((vertex, len(delta & uncovered)) for vertex, delta in subsets.items()
             if vertex not in selected),
            key=lambda item: (item[1], -item[0]),
            default=(-1, 0),
        )
        if gain == 0:
            raise ValueError(
                "Graph khong ton tai identifying code "
                "(co the co hai dinh co cung lan can dong)"
            )
        selected.add(candidate)
        uncovered.difference_update(subsets[candidate])

    return selected


def build_bitset_set_cover(
    neighborhoods: dict[int, set[int]],
) -> tuple[int, list[int], list[int]]:
    """Materialize universe va moi delta_c bang bitset nen.

    Moi bit van tuong ung voi dung mot constraint. Cach bieu dien nay giu
    nguyen greedy set cover explicit nhung tranh overhead rat lon cua set
    chua hang tram trieu tuple Python.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Can NumPy de tao explicit universe dang bitset cho graph lon"
        ) from exc

    vertices = sorted(neighborhoods)
    number_of_vertices = len(vertices)
    vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
    closed_matrix = np.zeros(
        (number_of_vertices, number_of_vertices), dtype=np.bool_
    )
    for vertex, closed_neighborhood in neighborhoods.items():
        row = vertex_index[vertex]
        for codeword in closed_neighborhood:
            closed_matrix[row, vertex_index[codeword]] = True

    pair_left, pair_right = np.triu_indices(number_of_vertices, k=1)
    universe_size = (
        number_of_vertices + number_of_vertices * (number_of_vertices - 1) // 2
    )
    deltas: list[int] = []
    for candidate_index in range(number_of_vertices):
        pair_constraints = np.logical_xor(
            closed_matrix[pair_left, candidate_index],
            closed_matrix[pair_right, candidate_index],
        )
        constraint_bits = np.concatenate(
            (closed_matrix[:, candidate_index], pair_constraints)
        )
        packed = np.packbits(constraint_bits, bitorder="little")
        deltas.append(int.from_bytes(packed, byteorder="little"))

    return universe_size, deltas, vertices


def greedy_set_cover_bitsets(
    universe_size: int, deltas: list[int], vertices: list[int]
) -> set[int]:
    """Greedy set cover tren universe va cac delta duoc ma hoa thanh bitset."""
    uncovered = (1 << universe_size) - 1
    selected_indices: set[int] = set()

    while uncovered:
        candidate_index, gain = max(
            (
                (index, (delta & uncovered).bit_count())
                for index, delta in enumerate(deltas)
                if index not in selected_indices
            ),
            key=lambda item: (item[1], -vertices[item[0]]),
            default=(-1, 0),
        )
        if gain == 0:
            raise ValueError(
                "Graph khong ton tai identifying code "
                "(co the co hai dinh co cung lan can dong)"
            )
        selected_indices.add(candidate_index)
        uncovered &= ~deltas[candidate_index]

    return {vertices[index] for index in selected_indices}


def id_greedy(graph: dict[int, set[int]]) -> set[int]:
    """Thuc hien cac buoc cua ID-Greedy va tra ve C_greedy."""
    signatures: dict[frozenset[int], int] = {}
    for vertex in sorted(graph):
        signature = frozenset(graph[vertex] | {vertex})
        if signature in signatures:
            raise ValueError(
                f"Graph khong ton tai identifying code: hai dinh "
                f"{signatures[signature]} va {vertex} co cung lan can dong"
            )
        signatures[signature] = vertex

    identifying_sets = closed_neighborhoods(graph)
    number_of_vertices = len(graph)
    estimated_delta_entries = sum(
        len(closed_neighborhood)
        * (number_of_vertices - len(closed_neighborhood) + 1)
        for closed_neighborhood in identifying_sets.values()
    )
    if estimated_delta_entries > 10_000_000:
        universe_size, deltas, vertices = build_bitset_set_cover(identifying_sets)
        return greedy_set_cover_bitsets(universe_size, deltas, vertices)

    universe, deltas = build_set_cover(identifying_sets)
    return greedy_set_cover(universe, deltas)


def is_identifying_code(
    graph: dict[int, set[int]], code: set[int]
) -> tuple[bool, str]:
    """Kiem tra dong thoi domination va separation cua ``code``."""
    if not code <= graph.keys():
        return False, "Code chua dinh khong thuoc graph"

    neighborhoods = closed_neighborhoods(graph)
    seen: dict[frozenset[int], int] = {}
    for vertex in sorted(graph):
        identifying_set = frozenset(neighborhoods[vertex] & code)
        if not identifying_set:
            return False, f"Dinh {vertex} khong duoc dominate"
        if identifying_set in seen:
            return (
                False,
                f"Hai dinh {seen[identifying_set]} va {vertex} khong duoc phan biet",
            )
        seen[identifying_set] = vertex
    return True, "Thoa man tinh chat Identifying Code"


def _peak_ram_bytes() -> int:
    """Return peak RSS; macOS reports bytes while Linux reports KiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _set_ram_limit(limit_bytes: int) -> None:
    """Apply RLIMIT_AS where the operating system supports it."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    effective_limit = min(limit_bytes, hard) if hard != resource.RLIM_INFINITY else limit_bytes
    # Keep the inherited hard limit. macOS rejects lowering its special
    # 2**63-1 hard value even though it displays it as unlimited.
    try:
        resource.setrlimit(resource.RLIMIT_AS, (effective_limit, hard))
    except (OSError, ValueError):
        # Darwin commonly rejects RLIMIT_AS changes. The parent independently
        # enforces the RSS limit with psutil on every supported platform.
        pass


def _write_json_atomic(path: Path, result: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_worker(
    graph_path: Path,
    result_path: Path,
    solution_directory: Path,
    ram_limit_bytes: int,
) -> int:
    """Run and verify one dataset. Resource limits are installed by the parent."""
    start = time.perf_counter()
    result: dict[str, object] = {
        "dataset": graph_path.name,
        "status": "INVALID",
        "vertices": "",
        "edges": "",
        "code_size": "",
        "elapsed_seconds": "",
        "peak_ram_bytes": "",
        "solution_file": "",
        "detail": "",
    }
    try:
        graph = read_graph(graph_path)
        result["vertices"] = len(graph)
        result["edges"] = sum(map(len, graph.values())) // 2

        number_of_vertices = len(graph)
        estimated_delta_entries = sum(
            (len(neighbors) + 1) * (number_of_vertices - len(neighbors))
            for neighbors in graph.values()
        )
        if estimated_delta_entries > 10_000_000:
            universe_size = number_of_vertices * (number_of_vertices + 1) // 2
            bitset_bytes = number_of_vertices * ((universe_size + 7) // 8)
            if bitset_bytes > ram_limit_bytes:
                raise MemoryError(
                    "Bitset delta toi thieu can "
                    f"{bitset_bytes / 1024**3:.2f} GiB, "
                    f"vuot gioi han {ram_limit_bytes / 1024**3:.2f} GiB"
                )

        algorithm_start = time.perf_counter()
        code = id_greedy(graph)
        algorithm_elapsed = time.perf_counter() - algorithm_start
        valid, message = is_identifying_code(graph, code)

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
    except (OSError, RuntimeError, ValueError) as exc:
        # A standardized graph should have a solution. Failure to produce one is
        # therefore an invalid algorithm result for this experiment.
        result.update(
            status="INVALID",
            elapsed_seconds=f"{time.perf_counter() - start:.9f}",
            detail=f"{type(exc).__name__}: {exc}",
        )
    finally:
        result["peak_ram_bytes"] = _peak_ram_bytes()
        _write_json_atomic(result_path, result)
    return 0


def run_dataset_process(
    script_path: Path,
    graph_path: Path,
    solution_directory: Path,
    timeout_seconds: float,
    ram_limit_bytes: int,
) -> dict[str, object]:
    """Run one graph in an isolated process and enforce wall-time/RAM limits."""
    result_file_handle = tempfile.NamedTemporaryFile(
        prefix="greedy-result-", suffix=".json", delete=False
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
            observed_peak_ram = max(observed_peak_ram, worker.memory_info().rss)
        except psutil.Error:
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
        _, stderr = process.communicate()
        result_path.unlink(missing_ok=True)
        return {
            "dataset": graph_path.name,
            "status": limit_status,
            "vertices": "",
            "edges": "",
            "code_size": "",
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
            return json.loads(result_path.read_text(encoding="utf-8"))
        detail = stderr.strip()[-1000:] or f"worker exit code {process.returncode}"
        killed_for_memory = process.returncode in {
            -signal.SIGKILL,
            -signal.SIGABRT,
            -signal.SIGSEGV,
        }
        return {
            "dataset": graph_path.name,
            # Native allocators may be killed/abort before Python can catch a
            # MemoryError and write its result JSON.
            "status": "RAM_LIMITED" if killed_for_memory else "INVALID",
            "vertices": "",
            "edges": "",
            "code_size": "",
            "elapsed_seconds": f"{time.perf_counter() - start:.9f}",
            "peak_ram_bytes": observed_peak_ram,
            "solution_file": "",
            "detail": detail,
        }
    finally:
        result_path.unlink(missing_ok=True)


RESULT_FIELDS = [
    "dataset",
    "status",
    "vertices",
    "edges",
    "code_size",
    "elapsed_seconds",
    "peak_ram_bytes",
    "solution_file",
    "detail",
]


def append_csv_result(csv_path: Path, result: dict[str, object]) -> None:
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
        "graphs", nargs="*", type=Path,
        help="graph tuy chon; mac dinh chay tat ca standardized_dataset/",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--ram-limit-gb", type=float, default=DEFAULT_RAM_LIMIT_GB)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_path.parent / "greedy-set-cover-results.csv",
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=script_path.parent / "greedy-set-cover-solutions",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-ram-bytes", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker is not None:
        if args.result_file is None:
            parser.error("--worker requires --result-file")
        if args.worker_ram_bytes is None:
            parser.error("--worker requires --worker-ram-bytes")
        _set_ram_limit(args.worker_ram_bytes)
        raise SystemExit(
            run_worker(
                args.worker,
                args.result_file,
                args.solution_dir,
                args.worker_ram_bytes,
            )
        )
    if args.timeout <= 0:
        parser.error("--timeout phai lon hon 0")
    if args.ram_limit_gb <= 0:
        parser.error("--ram-limit-gb phai lon hon 0")

    dataset_directory = project_directory / "standardized_dataset"
    graph_paths = args.graphs or list(
            path for path in dataset_directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not graph_paths:
        parser.error(f"Khong tim thay graph trong {dataset_directory}")
    try:
        graph_paths = sorted(
            graph_paths,
            key=lambda path: (declared_vertex_count(path), path.name.lower()),
        )
    except (OSError, ValueError) as exc:
        parser.error(f"Khong the sap xep graph theo so dinh: {exc}")

    ram_limit_bytes = int(args.ram_limit_gb * 1024**3)
    # A new run replaces the old summary; each row is flushed immediately.
    args.output_csv.unlink(missing_ok=True)
    print(
        f"Tim thay {len(graph_paths)} graph. Timeout={args.timeout:g}s, "
        f"RAM limit={args.ram_limit_gb:g} GiB/graph. "
        "Thu tu: so dinh tang dan."
    )
    for index, graph_path in enumerate(graph_paths, start=1):
        result = run_dataset_process(
            script_path,
            graph_path.resolve(),
            args.solution_dir.resolve(),
            args.timeout,
            ram_limit_bytes,
        )
        append_csv_result(args.output_csv, result)
        print(
            f"[{index:02d}/{len(graph_paths):02d}] {graph_path.name}: "
            f"STATUS={result['status']} | time={result['elapsed_seconds']}s | "
            f"code_size={result['code_size'] or 'N/A'}"
        )
    print(f"Da ghi tong ket: {args.output_csv}")


if __name__ == "__main__":
    main()
