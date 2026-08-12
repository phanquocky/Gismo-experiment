#!/usr/bin/env python3
"""Chay ID-Greedy without-universe tren cac graph da chuan hoa.

Mac dinh script duyet toan bo ``standardized_dataset/`` theo so dinh tang dan.
Moi graph chay trong mot worker rieng voi timeout 8 gio va gioi han RAM 64 GiB.
Tong ket duoc ghi vao CSV; nghiem hoan tat duoc ghi thanh file rieng.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path


Graph = dict[int, set[int]]
DEFAULT_TIMEOUT_SECONDS = 8 * 60 * 60
DEFAULT_RAM_LIMIT_GB = 64.0
SUPPORTED_SUFFIXES = {".txt", ".mtx", ".edges"}


def declared_vertex_count(path: Path) -> int:
    """Chi doc header de sap xep graph theo so dinh."""
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


def _read_edge_list(lines: list[str]) -> Graph:
    """Doc cac dong edge-list."""
    graph: Graph = {}
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


def _read_matrix_market(lines: list[str]) -> Graph:
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

    graph: Graph = {vertex: set() for vertex in range(1, number_of_rows + 1)}
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


def read_graph(path: str | Path) -> Graph:
    """Doc edge-list hoac Matrix Market va tra ve danh sach ke."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    first_line = next((line.strip() for line in lines if line.strip()), "")
    if first_line.lstrip("%").lower().startswith("matrixmarket"):
        return _read_matrix_market(lines)
    return _read_edge_list(lines)


def closed_neighborhoods(graph: Graph) -> dict[int, set[int]]:
    """Tao cac cot/hang cua ma tran closed-neighborhood."""
    return {vertex: neighbors | {vertex} for vertex, neighbors in graph.items()}


def candidate_gains(
    groups: list[set[int]],
    vector_0: set[int],
    neighborhoods: dict[int, set[int]],
) -> dict[int, int]:
    """Tinh gain cua tat ca candidate trong mot lan quet ma tran.

    ``groups`` luu cac nhom dinh co cung vector nhan dang tren nhung cot da
    chon. Moi candidate tach mot nhom thanh phan 1 va phan 0. Neu hai phan
    co kich thuoc a va b thi candidate phan biet them a*b cap dinh.
    ``vector_0`` la nhom cac dinh chua duoc dominate.
    """
    gains = {candidate: 0 for candidate in neighborhoods}

    for group in groups:
        number_of_ones: dict[int, int] = {}
        for vertex in group:
            for candidate in neighborhoods[vertex]:
                number_of_ones[candidate] = number_of_ones.get(candidate, 0) + 1

        group_size = len(group)
        for candidate, ones in number_of_ones.items():
            gains[candidate] += ones * (group_size - ones)

    for vertex in vector_0:
        for candidate in neighborhoods[vertex]:
            gains[candidate] += 1

    return gains


def split_groups(
    groups: list[set[int]], candidate_column: set[int]
) -> list[set[int]]:
    """Cap nhat cac nhom vector 0/1 sau khi chon them mot cot."""
    new_groups: list[set[int]] = []
    for group in groups:
        vector_1 = group & candidate_column
        vector_0 = group - candidate_column
        if vector_1:
            new_groups.append(vector_1)
        if vector_0:
            new_groups.append(vector_0)
    return new_groups


def id_greedy_without_universe(graph: Graph) -> set[int]:
    """Chay ID-Greedy bang cac vector/nhom, khong materialize universe."""
    vertices = sorted(graph)

    signatures: dict[frozenset[int], int] = {}
    for vertex in vertices:
        signature = frozenset(graph[vertex] | {vertex})
        if signature in signatures:
            raise ValueError(
                f"Graph khong ton tai identifying code: hai dinh "
                f"{signatures[signature]} va {vertex} co cung lan can dong"
            )
        signatures[signature] = vertex

    neighborhoods = closed_neighborhoods(graph)
    # Ban dau moi dinh co cung vector rong va chua co dinh nao duoc dominate.
    groups = [set(vertices)]
    vector_0 = set(vertices)
    selected: set[int] = set()
    total_constraints = len(vertices) + len(vertices) * (len(vertices) - 1) // 2

    while total_constraints > 0:
        gains = candidate_gains(groups, vector_0, neighborhoods)
        candidate, gain = max(
            (
                (vertex, gains[vertex])
                for vertex in vertices
                if vertex not in selected
            ),
            key=lambda item: (item[1], -item[0]),
            default=(-1, 0),
        )

        if gain == 0:
            raise ValueError(
                "Graph khong ton tai identifying code "
                "(co the co hai dinh co cung lan can dong)"
            )

        selected.add(candidate)
        candidate_column = neighborhoods[candidate]
        groups = split_groups(groups, candidate_column)
        vector_0.difference_update(candidate_column)
        total_constraints -= gain

    return selected


# Alias ngan gon de co the dung cung interface voi greedy-set-cover.py.
id_greedy = id_greedy_without_universe


def is_identifying_code(graph: Graph, code: set[int]) -> tuple[bool, str]:
    """Kiem tra domination va separation cua output."""
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


def _load_batch_support():
    """Tai resource-isolated batch harness dung chung voi ban co universe."""
    path = Path(__file__).resolve().with_name("greedy-set-cover.py")
    spec = importlib.util.spec_from_file_location("greedy_set_cover_batch_support", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Khong the tai batch harness tu {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_worker(
    graph_path: Path,
    result_path: Path,
    solution_directory: Path,
    batch_support,
) -> int:
    """Chay va kiem chung mot dataset trong worker without-universe."""
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

        algorithm_start = time.perf_counter()
        code = id_greedy_without_universe(graph)
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
        default=script_path.parent / "greedy-set-cover-wou-results.csv",
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=script_path.parent / "greedy-set-cover-wou-solutions",
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
            run_worker(args.worker, args.result_file, args.solution_dir, batch_support)
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
    try:
        already_completed = batch_support.completed_datasets(args.output_csv)
    except (OSError, ValueError) as exc:
        parser.error(f"Khong the doc CSV ket qua cu: {exc}")
    pending_graphs = [
        path for path in graph_paths if path.name not in already_completed
    ]
    print(
        f"Tim thay {len(graph_paths)} graph; da co ket qua={len(graph_paths) - len(pending_graphs)}, "
        f"con lai={len(pending_graphs)}. Timeout={args.timeout:g}s, "
        f"RAM limit={args.ram_limit_gb:g} GiB/graph. "
        "Thuat toan=WITHOUT_UNIVERSE. Thu tu: so dinh tang dan."
    )
    if not pending_graphs:
        print(f"Khong co graph nao can chay. Ket qua: {args.output_csv}")
        return
    for index, graph_path in enumerate(pending_graphs, start=1):
        result = batch_support.run_dataset_process(
            script_path,
            graph_path.resolve(),
            args.solution_dir.resolve(),
            args.timeout,
            ram_limit_bytes,
        )
        batch_support.append_csv_result(args.output_csv, result)
        print(
            f"[{index:02d}/{len(pending_graphs):02d}] {graph_path.name}: "
            f"STATUS={result['status']} | time={result['elapsed_seconds']}s | "
            f"code_size={result['code_size'] or 'N/A'}"
        )
    print(f"Da ghi tong ket: {args.output_csv}")


if __name__ == "__main__":
    main()
