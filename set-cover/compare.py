#!/usr/bin/env python3
"""So sanh hai ID-Greedy tren tat ca graph trong thu muc datasets."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import signal
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, TextIO


Graph = dict[int, set[int]]
Algorithm = Callable[[Graph], set[int]]
Verifier = Callable[[Graph, set[int]], tuple[bool, str]]
SUPPORTED_SUFFIXES = {".txt", ".mtx", ".edges"}

# Cac gioi han mac dinh giup batch khong OOM/treo boi dataset rat lon.
DEFAULT_MAX_EXPLICIT_BYTES = 1_000_000_000
DEFAULT_MAX_LOAD_VERTICES = 2_000_000
DEFAULT_MAX_ALGORITHM_VERTICES = 5_000
DEFAULT_TIMEOUT_SECONDS = 60.0


class DatasetTooLargeError(ValueError):
    def __init__(self, vertices: int, limit: int) -> None:
        self.vertices = vertices
        self.limit = limit
        super().__init__(f"{vertices:,} dinh vuot gioi han doc {limit:,} dinh")


class AlgorithmTimeoutError(TimeoutError):
    pass


@dataclass
class BenchmarkResult:
    status: str
    code: set[int] | None = None
    elapsed: float | None = None
    valid: bool | None = None
    validation_message: str = ""
    detail: str = ""


@dataclass
class DatasetSummary:
    name: str
    vertices: int | None
    with_universe: BenchmarkResult
    without_universe: BenchmarkResult
    comparison: str


def load_module(path: Path, module_name: str) -> ModuleType:
    """Load module Python co ten file chua dau gach ngang."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Khong the load module tu {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def discover_datasets(dataset_directory: Path) -> list[Path]:
    """Lay tat ca file graph duoc ho tro, sap xep theo ten."""
    return sorted(
        (
            path
            for path in dataset_directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )


def _integer_triple(text: str) -> tuple[int, int, int] | None:
    fields = text.split()
    if len(fields) != 3:
        return None
    try:
        return tuple(map(int, fields))  # type: ignore[return-value]
    except ValueError:
        return None


def _matrix_market_entries(
    graph_file: TextIO,
) -> tuple[tuple[int, int, int], str | None, Iterator[str]]:
    """Doc dimension, ke ca bien the dat dimension trong comment."""
    commented_dimensions: tuple[int, int, int] | None = None

    for raw_line in graph_file:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("%"):
            candidate = _integer_triple(line.lstrip("%").strip())
            if candidate is not None and candidate[0] == candidate[1]:
                commented_dimensions = candidate
            continue

        candidate = _integer_triple(line)
        if candidate is not None and candidate[0] == candidate[1]:
            return candidate, None, iter(graph_file)
        if commented_dimensions is not None:
            return commented_dimensions, line, iter(graph_file)
        raise ValueError("Khong tim thay dong kich thuoc Matrix Market hop le")

    raise ValueError("File Matrix Market thieu du lieu")


def _read_matrix_market(path: Path, max_vertices: int) -> Graph:
    with path.open("r", encoding="utf-8") as graph_file:
        header = graph_file.readline().strip().lstrip("%").lower()
        if not header.startswith("matrixmarket matrix coordinate"):
            raise ValueError("Matrix Market header khong hop le")
        if "symmetric" not in header:
            raise ValueError("Chi ho tro Matrix Market symmetric")

        dimensions, first_entry, remaining_lines = _matrix_market_entries(graph_file)
        number_of_rows, number_of_columns, declared_entries = dimensions
        if number_of_rows != number_of_columns:
            raise ValueError("Identifying Code yeu cau ma tran vuong")
        if number_of_rows > max_vertices:
            raise DatasetTooLargeError(number_of_rows, max_vertices)

        graph: Graph = {
            vertex: set() for vertex in range(1, number_of_rows + 1)
        }
        entry_lines = [first_entry] if first_entry is not None else []
        actual_entries = 0
        for raw_line in chain(entry_lines, remaining_lines):
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Entry Matrix Market {actual_entries + 1} khong hop le")
            try:
                u, v = map(int, fields[:2])
            except ValueError as exc:
                raise ValueError(
                    f"Entry Matrix Market {actual_entries + 1} khong hop le"
                ) from exc
            if not (1 <= u <= number_of_rows and 1 <= v <= number_of_rows):
                raise ValueError(
                    f"Entry Matrix Market {actual_entries + 1} vuot ngoai kich thuoc"
                )
            actual_entries += 1
            if u != v:
                graph[u].add(v)
                graph[v].add(u)

        if actual_entries != declared_entries:
            raise ValueError(
                f"Khai bao {declared_entries} entries, doc duoc {actual_entries}"
            )
        return graph


def _read_edge_list(path: Path, max_vertices: int) -> Graph:
    graph: Graph = {}
    maximum_vertex = 0
    declared_vertices: int | None = None
    declared_edges: int | None = None
    actual_edges = 0

    with path.open("r", encoding="utf-8") as graph_file:
        for line_number, raw_line in enumerate(graph_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("%", "#")):
                metadata = _integer_triple(line.lstrip("%#").strip())
                # Dinh dang .edges: "% number_of_edges n n".
                if metadata is not None and metadata[1] == metadata[2]:
                    declared_edges = metadata[0]
                    declared_vertices = metadata[1]
                    if declared_vertices > max_vertices:
                        raise DatasetTooLargeError(declared_vertices, max_vertices)
                continue

            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Dong {line_number}: can it nhat 2 nhan dinh")
            try:
                u, v = map(int, fields[:2])
            except ValueError as exc:
                raise ValueError(
                    f"Dong {line_number}: nhan dinh phai la so nguyen"
                ) from exc
            if u <= 0 or v <= 0:
                raise ValueError(f"Dong {line_number}: chi ho tro nhan dinh duong")
            maximum_vertex = max(maximum_vertex, u, v)
            if maximum_vertex > max_vertices:
                raise DatasetTooLargeError(maximum_vertex, max_vertices)
            graph.setdefault(u, set())
            graph.setdefault(v, set())
            actual_edges += 1
            if u != v:
                graph[u].add(v)
                graph[v].add(u)

    number_of_vertices = declared_vertices or maximum_vertex
    if number_of_vertices == 0:
        raise ValueError("Graph rong")
    if maximum_vertex > number_of_vertices:
        raise ValueError("Nhan dinh lon hon so dinh duoc khai bao")
    if declared_edges is not None and actual_edges != declared_edges:
        raise ValueError(
            f"Khai bao {declared_edges} canh, doc duoc {actual_edges}"
        )
    for vertex in range(1, number_of_vertices + 1):
        graph.setdefault(vertex, set())
    return graph


def read_dataset(path: Path, max_vertices: int) -> Graph:
    if path.suffix.lower() == ".mtx":
        return _read_matrix_market(path, max_vertices)
    return _read_edge_list(path, max_vertices)


def find_closed_twins(graph: Graph) -> tuple[int, int] | None:
    """Tra ve ngay cap closed twins dau tien, neu co."""
    signatures: dict[frozenset[int], int] = {}
    for vertex in sorted(graph):
        signature = frozenset(graph[vertex] | {vertex})
        if signature in signatures:
            return signatures[signature], vertex
        signatures[signature] = vertex
    return None


def estimate_explicit_instance(graph: Graph) -> tuple[int, int]:
    number_of_vertices = len(graph)
    universe_size = number_of_vertices + number_of_vertices * (number_of_vertices - 1) // 2
    delta_entries = 0
    for neighbors in graph.values():
        column_size = len(neighbors) + 1
        delta_entries += column_size * (number_of_vertices - column_size + 1)
    return universe_size, delta_entries


def _raise_timeout(_signum: int, _frame: FrameType | None) -> None:
    raise AlgorithmTimeoutError


def benchmark_algorithm(
    algorithm: Algorithm,
    verifier: Verifier,
    graph: Graph,
    repeat: int,
    timeout_seconds: float,
) -> BenchmarkResult:
    """Do median thoi gian, co timeout cho tung lan chay."""
    durations: list[int] = []
    expected_code: set[int] | None = None

    for _ in range(repeat):
        previous_handler = None
        if timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
            previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        start = time.perf_counter_ns()
        try:
            code = algorithm(graph)
        except AlgorithmTimeoutError:
            elapsed = (time.perf_counter_ns() - start) / 1_000_000_000
            return BenchmarkResult(
                status="timeout",
                elapsed=elapsed,
                detail=f"vuot {timeout_seconds:g} giay",
            )
        except ValueError as exc:
            elapsed = (time.perf_counter_ns() - start) / 1_000_000_000
            return BenchmarkResult(
                status="no_solution", elapsed=elapsed, detail=str(exc)
            )
        except (MemoryError, RuntimeError) as exc:
            elapsed = (time.perf_counter_ns() - start) / 1_000_000_000
            return BenchmarkResult(
                status="error",
                elapsed=elapsed,
                detail=str(exc) or type(exc).__name__,
            )
        finally:
            if timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)

        durations.append(time.perf_counter_ns() - start)
        if expected_code is None:
            expected_code = code
        elif code != expected_code:
            return BenchmarkResult(
                status="error", detail="Thuat toan tra ve output khong on dinh"
            )

    if expected_code is None:
        return BenchmarkResult(status="error", detail="Khong co lan chay nao")
    valid, validation_message = verifier(graph, expected_code)
    return BenchmarkResult(
        status="ok",
        code=expected_code,
        elapsed=statistics.median(durations) / 1_000_000_000,
        valid=valid,
        validation_message=validation_message,
    )


def format_bytes(number_of_bytes: int) -> str:
    value = float(number_of_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def format_code(code: set[int], full_output: bool, output_limit: int) -> str:
    ordered = sorted(code)
    if full_output or len(ordered) <= output_limit:
        return str(ordered)
    half = output_limit // 2
    return f"{ordered[:half]} ... {ordered[-half:]} (rut gon)"


def print_result(
    title: str,
    result: BenchmarkResult,
    full_output: bool,
    output_limit: int,
) -> None:
    print(title)
    if result.status == "ok":
        assert result.code is not None and result.elapsed is not None
        print(f"  Output: {format_code(result.code, full_output, output_limit)}")
        print(f"  Size: {len(result.code)}")
        print(f"  Thoi gian: {result.elapsed:.9f} giay")
        print(
            f"  Kiem tra: {'DAT' if result.valid else 'KHONG DAT'}"
            f" - {result.validation_message}"
        )
    elif result.status == "no_solution":
        print("  Output: KHONG CO | Size: N/A")
        print(f"  Thoi gian den khi phat hien: {result.elapsed:.9f} giay")
        print(f"  Kiem tra: GRAPH KHONG TON TAI IDENTIFYING CODE - {result.detail}")
    elif result.status == "timeout":
        print("  Output: N/A | Size: N/A")
        print(f"  Thoi gian: {result.elapsed:.9f} giay")
        print(f"  Trang thai: TIMEOUT - {result.detail}")
    elif result.status == "skipped":
        print("  Output: N/A | Size: N/A | Thoi gian: N/A")
        print(f"  Trang thai: BO QUA - {result.detail}")
    else:
        print("  Output: N/A | Size: N/A")
        if result.elapsed is not None:
            print(f"  Thoi gian den khi loi: {result.elapsed:.9f} giay")
        print(f"  Trang thai: LOI - {result.detail}")


def comparison_label(
    with_result: BenchmarkResult, without_result: BenchmarkResult
) -> str:
    if with_result.status == without_result.status == "ok":
        if with_result.code == without_result.code:
            return "SAME_VALID" if with_result.valid and without_result.valid else "SAME_INVALID"
        if with_result.valid and without_result.valid:
            return "DIFF_BOTH_VALID"
        return "DIFF_HAS_INVALID"
    if with_result.status == without_result.status == "no_solution":
        return "NO_IC"
    if "timeout" in (with_result.status, without_result.status):
        return "TIMEOUT"
    if "skipped" in (with_result.status, without_result.status):
        return "SKIPPED"
    return "DIFF_STATUS"


def print_comparison(with_result: BenchmarkResult, without_result: BenchmarkResult) -> str:
    label = comparison_label(with_result, without_result)
    print("So sanh")
    if with_result.status == without_result.status == "ok":
        assert with_result.code is not None and without_result.code is not None
        assert with_result.elapsed is not None and without_result.elapsed is not None
        print(f"  Cung output: {'CO' if with_result.code == without_result.code else 'KHONG'}")
        print(f"  Cung size: {'CO' if len(with_result.code) == len(without_result.code) else 'KHONG'}")
        print(
            "  Tinh hop le: "
            f"co universe={'DAT' if with_result.valid else 'KHONG DAT'}, "
            f"without universe={'DAT' if without_result.valid else 'KHONG DAT'}"
        )
        if with_result.elapsed <= without_result.elapsed:
            print(f"  Nhanh hon: co universe ({without_result.elapsed / with_result.elapsed:.3f} lan)")
        else:
            print(f"  Nhanh hon: without universe ({with_result.elapsed / without_result.elapsed:.3f} lan)")
    elif label == "NO_IC":
        print("  Ca hai cung phat hien graph khong ton tai Identifying Code.")
    else:
        print(f"  Khong the so sanh day du; trang thai tong hop: {label}.")
        for name, result in (("co universe", with_result), ("without universe", without_result)):
            if result.status == "ok":
                print(f"  Output {name}: {'HOP LE' if result.valid else 'KHONG HOP LE'}")
    return label


def result_cell(result: BenchmarkResult) -> str:
    if result.status == "ok":
        assert result.code is not None and result.elapsed is not None
        return f"OK n={len(result.code)} {result.elapsed:.3f}s"
    if result.elapsed is not None:
        return f"{result.status.upper()} {result.elapsed:.3f}s"
    return result.status.upper()


def print_summary(summaries: list[DatasetSummary]) -> None:
    print("\n" + "=" * 120)
    print(f"TONG KET {len(summaries)} DATASETS")
    print(f"{'Dataset':30} {'|V|':>9} {'Co universe':>24} {'Without universe':>24} {'So sanh':>20}")
    print("-" * 120)
    for item in summaries:
        vertices = f"{item.vertices:,}" if item.vertices is not None else "N/A"
        print(
            f"{item.name[:30]:30} {vertices:>9} "
            f"{result_cell(item.with_universe):>24} "
            f"{result_cell(item.without_universe):>24} "
            f"{item.comparison:>20}"
        )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_directory = script_dir.parent / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graphs",
        nargs="*",
        type=Path,
        help="file tuy chon; neu bo trong se tu dong chay tat ca datasets",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="timeout moi thuat toan/dataset; 0 de tat (mac dinh: 60 giay)",
    )
    parser.add_argument("--max-load-vertices", type=int, default=DEFAULT_MAX_LOAD_VERTICES)
    parser.add_argument(
        "--max-algorithm-vertices", type=int, default=DEFAULT_MAX_ALGORITHM_VERTICES
    )
    parser.add_argument("--max-explicit-bytes", type=int, default=DEFAULT_MAX_EXPLICIT_BYTES)
    parser.add_argument("--force-explicit", action="store_true")
    parser.add_argument("--force-large", action="store_true")
    parser.add_argument("--full-output", action="store_true")
    parser.add_argument("--output-limit", type=int, default=40)
    args = parser.parse_args()

    if args.repeat <= 0:
        parser.error("--repeat phai lon hon 0")
    if args.timeout < 0:
        parser.error("--timeout khong duoc am")
    if args.max_load_vertices <= 0 or args.max_algorithm_vertices <= 0:
        parser.error("gioi han so dinh phai lon hon 0")
    if args.max_explicit_bytes <= 0:
        parser.error("--max-explicit-bytes phai lon hon 0")
    if args.output_limit < 2:
        parser.error("--output-limit phai it nhat la 2")

    try:
        with_universe: Any = load_module(
            script_dir / "greedy-set-cover.py", "greedy_set_cover"
        )
        without_universe: Any = load_module(
            script_dir / "greedy-set-cover-wou.py", "greedy_set_cover_wou"
        )
        graph_paths = args.graphs or discover_datasets(dataset_directory)
    except (OSError, ImportError) as exc:
        parser.exit(1, f"Loi khoi tao: {exc}\n")

    print(f"Tim thay {len(graph_paths)} dataset. Timeout: {args.timeout:g} giay/thuat toan.")
    summaries: list[DatasetSummary] = []

    for dataset_number, graph_path in enumerate(graph_paths, start=1):
        print("\n" + "=" * 88)
        print(f"[{dataset_number}/{len(graph_paths)}] Dataset: {graph_path}")
        try:
            graph = read_dataset(graph_path, args.max_load_vertices)
        except DatasetTooLargeError as exc:
            skipped = BenchmarkResult(status="skipped", detail=str(exc))
            print(f"BO QUA: {exc}")
            summaries.append(
                DatasetSummary(graph_path.name, exc.vertices, skipped, skipped, "SKIPPED")
            )
            continue
        except (OSError, ValueError) as exc:
            failed = BenchmarkResult(status="error", detail=str(exc))
            print(f"LOI DOC GRAPH: {exc}")
            summaries.append(
                DatasetSummary(graph_path.name, None, failed, failed, "READ_ERROR")
            )
            continue

        number_of_vertices = len(graph)
        number_of_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
        universe_size, delta_entries = estimate_explicit_instance(graph)
        compact_delta_bytes = number_of_vertices * ((universe_size + 7) // 8)
        twin_pair = find_closed_twins(graph)
        print(f"So dinh: {number_of_vertices:,} | So canh: {number_of_edges:,}")
        print(f"Universe: {universe_size:,} constraints")
        print(f"Delta entries: {delta_entries:,} | Bitset: {format_bytes(compact_delta_bytes)}")
        if twin_pair is not None:
            print(f"Closed twins: co; cap dau tien {list(twin_pair)}")
        else:
            print("Closed twins: khong co")

        large_algorithm = (
            twin_pair is None
            and number_of_vertices > args.max_algorithm_vertices
            and not args.force_large
        )
        if large_algorithm:
            detail = (
                f"{number_of_vertices:,} dinh vuot gioi han chay "
                f"{args.max_algorithm_vertices:,}; dung --force-large de ep chay"
            )
            without_result = BenchmarkResult(status="skipped", detail=detail)
            with_result = BenchmarkResult(status="skipped", detail=detail)
        else:
            without_result = benchmark_algorithm(
                without_universe.id_greedy,
                without_universe.is_identifying_code,
                graph,
                args.repeat,
                args.timeout,
            )
            gc.collect()

            if (
                twin_pair is None
                and compact_delta_bytes > args.max_explicit_bytes
                and not args.force_explicit
            ):
                with_result = BenchmarkResult(
                    status="skipped",
                    detail=(
                        f"bitset can {format_bytes(compact_delta_bytes)}, vuot "
                        f"gioi han {format_bytes(args.max_explicit_bytes)}; "
                        "dung --force-explicit de ep chay"
                    ),
                )
            else:
                with_result = benchmark_algorithm(
                    with_universe.id_greedy,
                    with_universe.is_identifying_code,
                    graph,
                    args.repeat,
                    args.timeout,
                )
                gc.collect()

        print()
        print_result(
            "Greedy Set Cover (co universe)",
            with_result,
            args.full_output,
            args.output_limit,
        )
        print()
        print_result(
            "Greedy Set Cover (without universe)",
            without_result,
            args.full_output,
            args.output_limit,
        )
        print()
        label = print_comparison(with_result, without_result)
        summaries.append(
            DatasetSummary(
                graph_path.name,
                number_of_vertices,
                with_result,
                without_result,
                label,
            )
        )
        del graph
        gc.collect()

    print_summary(summaries)


if __name__ == "__main__":
    main()
