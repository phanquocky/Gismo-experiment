#!/usr/bin/env python3
"""Thuc nghiem Greedy K=1 voi partition dinh goc va hai loai sensor.

Mac dinh chay 50 graph trong standardized_dataset/ theo so dinh tang dan.
Moi graph chay trong worker rieng, timeout 8 gio, RAM 64 GiB va resume tu
CSV. Moi ung vien greedy la mot dinh v: khi chon v, ca N-sensor
(closed-neighborhood) va L-sensor (single-vertex) tai v duoc them cung luc.
Moi goi nhu vay co chi phi 1 vi tri. Chon tat ca cac goi luon cho nghiem vi
chung chua day du L-sensors, neu khong bi timeout/RAM limit.

Voi K=1, moi fire state la mot dinh. Partition gom cac nhom dinh co cung
signature tren cac sensor da chon. Constraint con lai gom cac state co
signature 0 (domination) va moi cap state trong cung mot nhom (separation).

code_size dem so goi/vi tri da chon; total_selected_types = 2 * code_size.
Gain cua goi duoc tinh chung sau khi them ca hai bit, khong cong hai gain
rieng le vi cac constraints ma hai loai sensor giai quyet co the trung nhau.
Lazy heap chi tinh lai cac ung vien co upper bound du lon de canh tranh.
Sau greedy, reverse-delete loai cac goi du thua trong thu tu nguoc.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType


Graph = dict[int, set[int]]
SensorID = tuple[str, int]
Group = tuple[int, set[int]]
K = 1
ALGORITHM_VERSION = "partition-bundled-two-sensor-types-k1-pruned-lazy-v1"
EXPECTED_DATASET_COUNT = 50
DEFAULT_TIMEOUT_SECONDS = 8 * 60 * 60
DEFAULT_RAM_LIMIT_GB = 64.0
SUPPORTED_SUFFIXES = {".txt", ".mtx", ".edges"}
RESULT_FIELDS = [
    "algorithm_version",
    "dataset",
    "K",
    "status",
    "vertices",
    "edges",
    "fire_states",
    "candidate_type_count",
    "candidate_bundle_count",
    "total_selected_types",
    "n_sensor_count",
    "l_sensor_count",
    "dual_type_vertex_count",
    "pre_prune_code_size",
    "code_size",
    "remaining_constraints",
    "max_signature_multiplicity",
    "validation_passed",
    "elapsed_seconds",
    "validation_seconds",
    "peak_ram_bytes",
    "solution_file",
    "vertex_solution_file",
    "detail",
]


def load_batch_support() -> ModuleType:
    """Tai graph parser va process/RAM helpers tu harness set-cover."""
    path = Path(__file__).resolve().parent.parent / "set-cover" / "greedy-set-cover.py"
    spec = importlib.util.spec_from_file_location(
        "greedy_set_cover_batch_support", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Khong the tai batch harness tu {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose2(value: int) -> int:
    return value * (value - 1) // 2


def closed_neighborhoods(graph: Graph) -> Graph:
    """Kiem tra graph vo huong va tao N[v] ma khong xoa twins."""
    vertices = set(graph)
    neighborhoods = {v: set(neighbors) | {v} for v, neighbors in graph.items()}
    for vertex, neighbors in neighborhoods.items():
        if not neighbors <= vertices:
            raise ValueError(f"Dinh {vertex} co hang xom khong thuoc graph")
    for vertex, neighbors in neighborhoods.items():
        if any(vertex not in neighborhoods[u] for u in neighbors):
            raise ValueError("Graph phai vo huong")
    return neighborhoods


def build_candidates(graph: Graph) -> dict[SensorID, set[int]]:
    """Insertion order quy dinh tie-break: tat ca N truoc, sau do tat ca L."""
    neighborhoods = closed_neighborhoods(graph)
    vertices = sorted(graph)
    candidates = {("N", v): neighborhoods[v] for v in vertices}
    candidates.update({("L", v): {v} for v in vertices})
    return candidates


def remaining(groups: Sequence[Group]) -> int:
    """Dem domination va pair-separation constraints K=1 chua duoc thoa."""
    unresolved = 0
    for signature, vertices in groups:
        size = len(vertices)
        if signature == 0:
            unresolved += size
        unresolved += choose2(size)
    return unresolved


def candidate_gain(groups: Sequence[Group], detection: set[int]) -> int:
    """Exact gain khi them mot sensor don co detection cho truoc."""
    gain = 0
    for signature, vertices in groups:
        hit_count = len(vertices & detection)
        if not hit_count:
            continue
        size = len(vertices)
        if signature == 0:
            gain += hit_count
        gain += (
            choose2(size)
            - choose2(hit_count)
            - choose2(size - hit_count)
        )
    return gain


def refine(groups: Sequence[Group], detection: set[int], bit: int) -> list[Group]:
    """Them mot signal bit va tach partition cua cac fire state singleton."""
    refined: list[Group] = []
    for signature, vertices in groups:
        hit = vertices & detection
        miss = vertices - detection
        if miss:
            refined.append((signature, miss))
        if hit:
            refined.append((signature | bit, hit))
    return refined


def prepare_bundled_gain_evaluator(
    groups: Sequence[Group],
    neighborhoods: Mapping[int, set[int]],
) -> Callable[[int], int]:
    """Tra evaluator gain exact khi them dong thoi L(v) va N[v]."""
    group_sizes = [len(vertices) for _, vertices in groups]
    group_of: dict[int, int] = {}
    zero_group: int | None = None
    for group_index, (signature, group_vertices) in enumerate(groups):
        if signature == 0:
            zero_group = group_index
        for vertex in group_vertices:
            if vertex in group_of:
                raise RuntimeError("Partition groups bi trung dinh")
            group_of[vertex] = group_index
    if len(group_of) != sum(group_sizes):
        raise RuntimeError("Partition groups bi trung dinh")

    def evaluate(vertex: int) -> int:
        if vertex not in group_of or vertex not in neighborhoods:
            raise ValueError(f"Dinh {vertex} khong thuoc partition/graph")

        hit_by_group: dict[int, int] = {}
        for detected_vertex in neighborhoods[vertex]:
            group_index = group_of[detected_vertex]
            hit_by_group[group_index] = hit_by_group.get(group_index, 0) + 1

        gain = hit_by_group.get(zero_group, 0) if zero_group is not None else 0
        vertex_group = group_of[vertex]
        for group_index, hit_count in hit_by_group.items():
            size = group_sizes[group_index]
            # N(v) tach hit/miss; L(v) tiep tuc tach rieng v khoi cac N-hit
            # trong nhom dang chua v.
            other_hit_count = hit_count - (group_index == vertex_group)
            gain += (
                choose2(size)
                - choose2(size - hit_count)
                - choose2(other_hit_count)
            )
        return gain

    return evaluate


def all_bundled_gains(
    groups: Sequence[Group],
    neighborhoods: Mapping[int, set[int]],
    available: set[int],
) -> dict[int, int]:
    """Tinh gain moi goi trong mot batch; helper dung cho test/debug."""
    evaluate = prepare_bundled_gain_evaluator(groups, neighborhoods)
    return {vertex: evaluate(vertex) for vertex in sorted(available)}


def greedy_k1_bundled(
    graph: Graph,
    *,
    trace: bool = False,
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    """Greedy K=1 tren n goi, moi goi them ca N(v) va L(v), chi phi 1."""
    neighborhoods = closed_neighborhoods(graph)
    vertices = sorted(graph)
    available = set(vertices)
    groups: list[Group] = [(0, set(vertices))] if vertices else []
    selected: list[int] = []
    history: list[tuple[int, int, int, int]] = []
    gain_heap: list[tuple[int, int, int]] = []
    iteration = 0

    while True:
        if sum(len(group_vertices) for _, group_vertices in groups) != len(vertices):
            raise RuntimeError("Partition khong bao phu dung n fire states K=1")
        before = remaining(groups)
        if before == 0:
            return selected, history

        evaluate = prepare_bundled_gain_evaluator(groups, neighborhoods)
        if iteration == 0:
            for vertex in vertices:
                heapq.heappush(gain_heap, (-evaluate(vertex), vertex, iteration))

        while gain_heap:
            negative_bound, vertex, evaluated_iteration = heapq.heappop(gain_heap)
            if vertex not in available:
                continue
            bound = -negative_bound
            if evaluated_iteration == iteration:
                best_vertex = vertex
                best_gain = bound
                break
            exact_gain = evaluate(vertex)
            if exact_gain > bound:
                raise RuntimeError(
                    "Marginal gain tang; vi pham bat bien lazy Set Cover"
                )
            heapq.heappush(gain_heap, (-exact_gain, vertex, iteration))
        else:
            raise RuntimeError("Het goi ung vien khi van con constraint")

        if best_gain <= 0:
            raise RuntimeError(
                "Khong co positive bundled gain; implementation hoac input sai"
            )
        available.remove(best_vertex)
        selected.append(best_vertex)
        first_bit = 1 << (2 * (len(selected) - 1))
        groups = refine(groups, {best_vertex}, first_bit)
        groups = refine(groups, neighborhoods[best_vertex], first_bit << 1)
        actual_after = remaining(groups)
        if actual_after != before - best_gain:
            raise RuntimeError(
                f"Gain invariant sai: before={before}, gain={best_gain}, "
                f"actual_after={actual_after}"
            )
        if trace:
            history.append((best_vertex, best_gain, before, actual_after))
        iteration += 1


def greedy_k1(
    vertices: Iterable[int],
    candidates: Mapping[SensorID, set[int]],
    *,
    trace: bool = False,
) -> tuple[list[SensorID], list[tuple[SensorID, int, int, int]]]:
    """Greedy K=1 cho cac sensor rieng le; stable tie-break theo mapping."""
    vertex_set = set(vertices)
    available: dict[SensorID, set[int]] = {}
    for sensor, detection in candidates.items():
        if not detection <= vertex_set:
            raise ValueError(f"Detection cua sensor {sensor} vuot ngoai V")
        available[sensor] = set(detection)

    groups: list[Group] = [(0, set(vertex_set))] if vertex_set else []
    selected: list[SensorID] = []
    history: list[tuple[SensorID, int, int, int]] = []
    while True:
        if sum(len(group_vertices) for _, group_vertices in groups) != len(vertex_set):
            raise RuntimeError("Partition khong bao phu dung n fire states K=1")
        before = remaining(groups)
        if before == 0:
            return selected, history

        best: SensorID | None = None
        best_gain = 0
        for sensor, detection in available.items():
            gain = candidate_gain(groups, detection)
            if gain > best_gain:
                best, best_gain = sensor, gain
        if best is None or best_gain <= 0:
            raise RuntimeError(
                "Khong co positive gain; thieu L-sensor hoac implementation sai"
            )
        detection = available.pop(best)
        groups = refine(groups, detection, 1 << len(selected))
        selected.append(best)
        after = remaining(groups)
        if after != before - best_gain:
            raise RuntimeError(
                f"Gain invariant sai: before={before}, gain={best_gain}, after={after}"
            )
        if trace:
            history.append((best, best_gain, before, after))


def sensor_detection(graph: Graph, sensor: SensorID) -> set[int]:
    kind, vertex = sensor
    if vertex not in graph:
        raise ValueError(f"Sensor {sensor} khong thuoc graph")
    if kind == "N":
        return set(graph[vertex]) | {vertex}
    if kind == "L":
        return {vertex}
    raise ValueError(f"Loai sensor khong hop le: {kind}")


def validate_solution(
    graph: Graph, selected: Sequence[SensorID]
) -> tuple[bool, str, int, int]:
    """Kiem tra moi singleton fire state co signature rieng va khac 0."""
    if len(set(selected)) != len(selected):
        return False, "Danh sach selected bi trung sensor", 0, 0
    closed_neighborhoods(graph)
    signatures = {vertex: 0 for vertex in graph}
    try:
        for index, sensor in enumerate(selected):
            for vertex in sensor_detection(graph, sensor):
                signatures[vertex] |= 1 << index
    except ValueError as exc:
        return False, str(exc), 0, 0

    by_signature: dict[int, set[int]] = {}
    for vertex in sorted(graph):
        by_signature.setdefault(signatures[vertex], set()).add(vertex)
    groups = [(signature, vertices) for signature, vertices in by_signature.items()]
    if sum(len(vertices) for _, vertices in groups) != len(graph):
        return False, "Validation dem sai tong fire states K=1", 0, 0
    unresolved = remaining(groups)
    maximum = max((len(vertices) for _, vertices in groups), default=0)
    if unresolved:
        return (
            False,
            f"Con {unresolved} constraints; max signature multiplicity={maximum}",
            unresolved,
            maximum,
        )
    return True, "Moi singleton fire state co signature rieng, khac 0", 0, maximum


def physical_vertices(selected: Iterable[SensorID]) -> set[int]:
    """N va L cung vi tri chi dong gop mot lan vao code_size."""
    return {vertex for _, vertex in selected}


def bundled_sensors(vertices: Iterable[int]) -> list[SensorID]:
    return [(kind, vertex) for vertex in vertices for kind in ("N", "L")]


def prune_bundled_vertices(graph: Graph, selected_vertices: Sequence[int]) -> list[int]:
    """Reverse-delete cac goi du thua; moi phep xoa deu validate K=1 doc lap."""
    kept = list(selected_vertices)
    for vertex in reversed(selected_vertices):
        trial = [kept_vertex for kept_vertex in kept if kept_vertex != vertex]
        if validate_solution(graph, bundled_sensors(trial))[0]:
            kept = trial
    return kept


def solve_two_sensor_types(graph: Graph, *, trace: bool = False) -> dict[str, object]:
    greedy_selected_vertices, history = greedy_k1_bundled(graph, trace=trace)
    selected_vertices = prune_bundled_vertices(graph, greedy_selected_vertices)
    selected = bundled_sensors(selected_vertices)
    locations = set(selected_vertices)
    return {
        "selected": selected,
        "selected_vertices": selected_vertices,
        "greedy_selected_vertices": greedy_selected_vertices,
        "removed_vertices": [
            vertex for vertex in greedy_selected_vertices if vertex not in locations
        ],
        "trace": history,
        "code_vertices": locations,
        "pre_prune_code_size": len(greedy_selected_vertices),
        "code_size": len(locations),
        "total_selected_types": len(selected),
        "n_sensor_count": len(locations),
        "l_sensor_count": len(locations),
        "dual_type_vertex_count": len(locations),
    }


def result_template(graph_path: Path) -> dict[str, object]:
    result: dict[str, object] = dict.fromkeys(RESULT_FIELDS, "")
    result.update(
        algorithm_version=ALGORITHM_VERSION,
        dataset=graph_path.name,
        K=K,
        status="INVALID",
    )
    return result


def run_worker(
    graph_path: Path,
    result_path: Path,
    solution_directory: Path,
    batch_support: ModuleType,
) -> int:
    """Chay mot graph, validation ngoai solver timer, ghi JSON atomic."""
    start = time.perf_counter()
    result = result_template(graph_path)
    try:
        graph = batch_support.read_graph(graph_path)
        n = len(graph)
        result.update(
            vertices=n,
            edges=sum(map(len, graph.values())) // 2,
            fire_states=n,
            candidate_type_count=2 * n,
            candidate_bundle_count=n,
        )
        algorithm_start = time.perf_counter()
        solution = solve_two_sensor_types(graph)
        result["elapsed_seconds"] = f"{time.perf_counter() - algorithm_start:.9f}"
        selected = solution["selected"]
        if not isinstance(selected, list):
            raise RuntimeError("Solver khong tra ve list selected")

        validation_start = time.perf_counter()
        valid, message, unresolved, maximum = validate_solution(graph, selected)
        result.update(
            status="VALID" if valid else "INVALID",
            code_size=solution["code_size"],
            total_selected_types=solution["total_selected_types"],
            n_sensor_count=solution["n_sensor_count"],
            l_sensor_count=solution["l_sensor_count"],
            dual_type_vertex_count=solution["dual_type_vertex_count"],
            pre_prune_code_size=solution["pre_prune_code_size"],
            remaining_constraints=unresolved,
            max_signature_multiplicity=maximum,
            validation_passed=valid,
            validation_seconds=f"{time.perf_counter() - validation_start:.9f}",
            detail=message,
        )
        if valid:
            solution_directory.mkdir(parents=True, exist_ok=True)
            sensor_path = (
                solution_directory / f"{graph_path.name}.K1.bundled.sensor-types.txt"
            )
            sensor_tmp = sensor_path.with_name(sensor_path.name + ".tmp")
            sensor_tmp.write_text(
                "".join(f"{kind}\t{vertex}\n" for kind, vertex in selected),
                encoding="utf-8",
            )
            sensor_tmp.replace(sensor_path)

            vertex_path = (
                solution_directory / f"{graph_path.name}.K1.bundled.vertices.txt"
            )
            vertex_tmp = vertex_path.with_name(vertex_path.name + ".tmp")
            vertex_tmp.write_text(
                "".join(f"{vertex}\n" for vertex in sorted(solution["code_vertices"])),
                encoding="utf-8",
            )
            vertex_tmp.replace(vertex_path)
            result.update(
                solution_file=str(sensor_path),
                vertex_solution_file=str(vertex_path),
            )
    except MemoryError as exc:
        result.update(status="RAM_LIMITED", detail=str(exc) or "Vuot gioi han RAM")
    except Exception as exc:
        result.update(status="INVALID", detail=f"{type(exc).__name__}: {exc}")
    finally:
        if not result["elapsed_seconds"]:
            result["elapsed_seconds"] = f"{time.perf_counter() - start:.9f}"
        result["peak_ram_bytes"] = batch_support._peak_ram_bytes()
        batch_support._write_json_atomic(result_path, result)
    return 0


def completed_datasets(csv_path: Path) -> set[str]:
    """Resume chi cac dong cua dung algorithm version va K=1."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != RESULT_FIELDS:
            raise ValueError(f"CSV {csv_path} khong khop schema cua script")
        return {
            row["dataset"]
            for row in reader
            if row.get("algorithm_version") == ALGORITHM_VERSION
            and row.get("K") == str(K)
            and row.get("dataset")
            and row.get("status")
        }


def append_result(csv_path: Path, result: Mapping[str, object]) -> None:
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
        help="graph tuy chon; mac dinh chay 50 graph standardized_dataset/",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--ram-limit-gb", type=float, default=DEFAULT_RAM_LIMIT_GB)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_path.with_name(
            "greedy_k1_partition_bundled_two_sensor_types-lazy-results.csv"
        ),
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=(
            script_path.parent
            / "greedy_k1_partition_bundled_two_sensor_types-lazy-solutions"
        ),
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-ram-bytes", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        batch_support = load_batch_support()
    except (OSError, ImportError) as exc:
        parser.exit(1, f"Loi tai batch harness: {exc}\n")

    if args.worker is not None:
        if args.result_file is None or args.worker_ram_bytes is None:
            parser.error("--worker requires --result-file and --worker-ram-bytes")
        if args.worker_ram_bytes <= 0:
            parser.error("--worker-ram-bytes phai lon hon 0")
        batch_support._set_ram_limit(args.worker_ram_bytes)
        raise SystemExit(
            run_worker(args.worker, args.result_file, args.solution_dir, batch_support)
        )

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout phai huu han va lon hon 0")
    if not math.isfinite(args.ram_limit_gb) or args.ram_limit_gb <= 0:
        parser.error("--ram-limit-gb phai huu han va lon hon 0")

    dataset_directory = project_directory / "standardized_dataset"
    try:
        graph_paths = args.graphs or [
            path
            for path in dataset_directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if not graph_paths:
            parser.error(f"Khong tim thay graph trong {dataset_directory}")
        if not args.graphs and len(graph_paths) != EXPECTED_DATASET_COUNT:
            parser.error(
                f"Can {EXPECTED_DATASET_COUNT} graph, tim thay {len(graph_paths)}"
            )
        if len({path.name for path in graph_paths}) != len(graph_paths):
            parser.error("Ten dataset phai duy nhat de resume CSV")
        graph_paths = sorted(
            graph_paths,
            key=lambda path: (
                batch_support.declared_vertex_count(path),
                path.name.lower(),
            ),
        )
        done = completed_datasets(args.output_csv)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(f"Khong the doc dataset/CSV: {exc}")

    pending = [path for path in graph_paths if path.name not in done]
    print(
        f"Tim thay {len(graph_paths)} graph; da co ket qua={len(graph_paths) - len(pending)}; "
        f"con lai={len(pending)}. K={K}; sensor_types=N,L; timeout={args.timeout:g}s; "
        f"RAM={args.ram_limit_gb:g} GiB/graph. code_size=unique vertex positions.",
        flush=True,
    )
    ram_limit_bytes = int(args.ram_limit_gb * 1024**3)
    for index, graph_path in enumerate(pending, start=1):
        result = result_template(graph_path)
        result.update(
            batch_support.run_dataset_process(
                script_path,
                graph_path.resolve(),
                args.solution_dir.resolve(),
                args.timeout,
                ram_limit_bytes,
            )
        )
        append_result(args.output_csv, result)
        code_size = result["code_size"]
        selected_types = result["total_selected_types"]
        print(
            f"[{index:02d}/{len(pending):02d}] {graph_path.name}: "
            f"STATUS={result['status']} | time={result['elapsed_seconds']}s | "
            f"code_size={code_size if code_size != '' else 'N/A'} | "
            f"selected_types={selected_types if selected_types != '' else 'N/A'}",
            flush=True,
        )
        if result["status"] != "VALID":
            print(f"  detail: {result['detail']}", flush=True)
    print(f"Da ghi tong ket: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
