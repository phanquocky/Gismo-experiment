#!/usr/bin/env python3
"""Thuc nghiem Greedy K=2 voi partition dinh goc va hai loai sensor.

Mac dinh chay 50 graph trong standardized_dataset/ theo so dinh tang dan.
Moi graph chay trong worker rieng, timeout 8 gio, RAM 64 GiB va resume tu
CSV. Moi ung vien greedy la mot dinh v: khi chon v, ca N-sensor
(closed-neighborhood) va L-sensor (single-vertex) tai v duoc them cung luc.
Moi goi nhu vay co chi phi 1 vi tri. Chon tat ca cac goi luon cho nghiem vi
chung chua day du L-sensors, neu khong bi timeout/RAM limit.

code_size dem so goi/vi tri da chon; total_selected_types = 2 * code_size.
Gain cua goi duoc tinh chung sau khi them ca hai bit, khong cong hai gain
rieng le vi cac constraints ma hai loai sensor giai quyet co the trung nhau.
Moi vong tinh gain theo batch: dung chung local-count theo partition group,
cache cac N[v] co cung residual profile, va chi refine goi thang cuoc.
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
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType


Graph = dict[int, set[int]]
SensorID = tuple[str, int]
Group = tuple[int, set[int]]
K = 2
ALGORITHM_VERSION = "partition-bundled-two-sensor-types-k2-pruned-batched-lazy-v3"
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


def aggregate(groups: Sequence[Group], weights: Sequence[int]) -> dict[int, int]:
    """Dem singleton/pair fire states theo OR signature ma khong liet ke F."""
    if len(groups) != len(weights):
        raise ValueError("So weights phai bang so groups")
    out: defaultdict[int, int] = defaultdict(int)
    for i, (signature, _) in enumerate(groups):
        weight = weights[i]
        if weight < 0:
            raise ValueError("Weight khong duoc am")
        out[signature] += weight + choose2(weight)
        for j in range(i):
            out[signature | groups[j][0]] += weight * weights[j]
    return {signature: count for signature, count in out.items() if count}


def remaining(counts: Mapping[int, int]) -> int:
    """So domination va pair-separation constraints chua duoc thoa."""
    return counts.get(0, 0) + sum(choose2(count) for count in counts.values())


def candidate_gain(
    groups: Sequence[Group],
    counts: Mapping[int, int],
    detection: set[int],
) -> int:
    """Exact gain: b_0 + sum_s b_s*z_s theo dac ta."""
    residual_weights = [len(vertices - detection) for _, vertices in groups]
    zeros = aggregate(groups, residual_weights)
    gain = counts.get(0, 0) - zeros.get(0, 0)
    for signature, count in counts.items():
        zero_count = zeros.get(signature, 0)
        gain += (count - zero_count) * zero_count
    return gain


def refine(groups: Sequence[Group], detection: set[int], bit: int) -> list[Group]:
    """Them mot signal bit va tach partition cua cac dinh goc."""
    refined: list[Group] = []
    for signature, vertices in groups:
        hit = vertices & detection
        miss = vertices - detection
        if miss:
            refined.append((signature, miss))
        if hit:
            refined.append((signature | bit, hit))
    return refined


def containing_state_counts_by_group(groups: Sequence[Group]) -> list[dict[int, int]]:
    """Dem states chua v; ket qua giong nhau cho moi v trong cung group."""
    signatures = [signature for signature, _ in groups]
    sizes = [len(vertices) for _, vertices in groups]
    state_count = sum(sizes)
    out: list[dict[int, int]] = []
    for i, signature in enumerate(signatures):
        counts: defaultdict[int, int] = defaultdict(int)
        counts[signature] += 1  # singleton {v}
        for j, other_signature in enumerate(signatures):
            pair_count = sizes[j] - (i == j)
            if pair_count:
                counts[signature | other_signature] += pair_count
        if sum(counts.values()) != state_count:
            raise RuntimeError("Dem sai so singleton/pair states chua mot dinh")
        out.append(dict(counts))
    return out


def prepare_bundled_gain_evaluator(
    groups: Sequence[Group],
    counts: Mapping[int, int],
    neighborhoods: Mapping[int, set[int]],
) -> Callable[[int], int]:
    """Chuan bi state chung va tra evaluator co cache cho mot vong greedy."""
    group_sizes = [len(vertices) for _, vertices in groups]
    group_of: dict[int, int] = {}
    for group_index, (_, group_vertices) in enumerate(groups):
        for vertex in group_vertices:
            group_of[vertex] = group_index
    if len(group_of) != sum(group_sizes):
        raise RuntimeError("Partition groups bi trung dinh")

    containing_by_group = containing_state_counts_by_group(groups)
    profile_cache: dict[tuple[int, ...], tuple[int, dict[int, int]]] = {}

    def evaluate(vertex: int) -> int:
        residual = group_sizes.copy()
        for detected_vertex in neighborhoods[vertex]:
            residual[group_of[detected_vertex]] -= 1
        profile = tuple(residual)

        cached = profile_cache.get(profile)
        if cached is None:
            zeros = aggregate(groups, profile)
            neighbor_gain = counts.get(0, 0) - zeros.get(0, 0)
            for signature, count in counts.items():
                zero_count = zeros.get(signature, 0)
                neighbor_gain += (count - zero_count) * zero_count
            cached = neighbor_gain, zeros
            profile_cache[profile] = cached
        neighbor_gain, zeros = cached

        local_gain = 0
        for signature, containing_count in containing_by_group[
            group_of[vertex]
        ].items():
            neighborhood_hit_count = counts[signature] - zeros.get(signature, 0)
            if containing_count > neighborhood_hit_count:
                raise RuntimeError("State chua v phai duoc N[v] phat hien")
            local_gain += containing_count * (neighborhood_hit_count - containing_count)
        return neighbor_gain + local_gain

    return evaluate


def all_bundled_gains(
    groups: Sequence[Group],
    counts: Mapping[int, int],
    neighborhoods: Mapping[int, set[int]],
    available: set[int],
) -> dict[int, int]:
    """Tinh gain moi goi trong mot batch; helper dung cho test/debug."""
    evaluate = prepare_bundled_gain_evaluator(groups, counts, neighborhoods)
    return {vertex: evaluate(vertex) for vertex in sorted(available)}


def greedy_k2_bundled(
    graph: Graph,
    *,
    trace: bool = False,
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    """Greedy tren n goi vi tri, moi goi them ca N(v) va L(v), chi phi 1."""
    neighborhoods = closed_neighborhoods(graph)
    vertices = sorted(graph)
    available = set(vertices)
    groups: list[Group] = [(0, set(vertices))] if vertices else []
    selected: list[int] = []
    history: list[tuple[int, int, int, int]] = []
    total_states = len(vertices) + choose2(len(vertices))
    counts = aggregate(groups, [len(group_vertices) for _, group_vertices in groups])
    gain_heap: list[tuple[int, int, int]] = []
    iteration = 0

    while True:
        if sum(counts.values()) != total_states:
            raise RuntimeError("Tong state counts khong bang n + C(n,2)")
        before = remaining(counts)
        if before == 0:
            return selected, history

        evaluate = prepare_bundled_gain_evaluator(groups, counts, neighborhoods)
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
        counts = aggregate(
            groups, [len(group_vertices) for _, group_vertices in groups]
        )
        actual_after = remaining(counts)
        if actual_after != before - best_gain:
            raise RuntimeError(
                f"Gain invariant sai: before={before}, gain={best_gain}, "
                f"actual_after={actual_after}"
            )
        if trace:
            history.append((best_vertex, best_gain, before, actual_after))
        iteration += 1


def greedy_k2(
    vertices: Iterable[int],
    candidates: Mapping[SensorID, set[int]],
    *,
    trace: bool = False,
) -> tuple[list[SensorID], list[tuple[SensorID, int, int, int]]]:
    """Greedy K=2 khong dung states/Universe; stable tie-break theo mapping."""
    vertex_set = set(vertices)
    available: dict[SensorID, set[int]] = {}
    for sensor, detection in candidates.items():
        if not detection <= vertex_set:
            raise ValueError(f"Detection cua sensor {sensor} vuot ngoai V")
        available[sensor] = set(detection)

    groups: list[Group] = [(0, set(vertex_set))] if vertex_set else []
    selected: list[SensorID] = []
    history: list[tuple[SensorID, int, int, int]] = []
    total_states = len(vertex_set) + choose2(len(vertex_set))
    while True:
        counts = aggregate(groups, [len(vertices) for _, vertices in groups])
        if sum(counts.values()) != total_states:
            raise RuntimeError("Tong state counts khong bang n + C(n,2)")
        before = remaining(counts)
        if before == 0:
            return selected, history

        best: SensorID | None = None
        best_gain = 0
        for sensor, detection in available.items():
            gain = candidate_gain(groups, counts, detection)
            if gain > best_gain:
                best, best_gain = sensor, gain
        if best is None or best_gain <= 0:
            raise RuntimeError(
                "Khong co positive gain; thieu L-sensor hoac implementation sai"
            )
        detection = available.pop(best)
        groups = refine(groups, detection, 1 << len(selected))
        selected.append(best)
        after_counts = aggregate(groups, [len(vertices) for _, vertices in groups])
        after = remaining(after_counts)
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
    """Rebuild vertex signatures, aggregate K=2 counts, khong dung solver state."""
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
    counts = aggregate(groups, [len(vertices) for _, vertices in groups])
    expected_states = len(graph) + choose2(len(graph))
    if sum(counts.values()) != expected_states:
        return False, "Validation dem sai tong fire states", 0, 0
    unresolved = remaining(counts)
    maximum = max(counts.values(), default=0)
    if unresolved:
        return (
            False,
            f"Con {unresolved} constraints; max signature multiplicity={maximum}",
            unresolved,
            maximum,
        )
    return True, "Moi singleton/pair fire state co signature rieng, khac 0", 0, maximum


def physical_vertices(selected: Iterable[SensorID]) -> set[int]:
    """N va L cung vi tri chi dong gop mot lan vao code_size."""
    return {vertex for _, vertex in selected}


def bundled_sensors(vertices: Iterable[int]) -> list[SensorID]:
    return [(kind, vertex) for vertex in vertices for kind in ("N", "L")]


def prune_bundled_vertices(graph: Graph, selected_vertices: Sequence[int]) -> list[int]:
    """Reverse-delete cac goi du thua; moi phep xoa deu duoc validate doc lap."""
    kept = list(selected_vertices)
    for vertex in reversed(selected_vertices):
        trial = [kept_vertex for kept_vertex in kept if kept_vertex != vertex]
        if validate_solution(graph, bundled_sensors(trial))[0]:
            kept = trial
    return kept


def solve_two_sensor_types(graph: Graph, *, trace: bool = False) -> dict[str, object]:
    greedy_selected_vertices, history = greedy_k2_bundled(graph, trace=trace)
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
            fire_states=n + choose2(n),
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
                solution_directory / f"{graph_path.name}.K2.bundled.sensor-types.txt"
            )
            sensor_tmp = sensor_path.with_name(sensor_path.name + ".tmp")
            sensor_tmp.write_text(
                "".join(f"{kind}\t{vertex}\n" for kind, vertex in selected),
                encoding="utf-8",
            )
            sensor_tmp.replace(sensor_path)

            vertex_path = (
                solution_directory / f"{graph_path.name}.K2.bundled.vertices.txt"
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
    """Resume chi cac dong cua dung algorithm version va K=2."""
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
            "greedy_k2_partition_bundled_two_sensor_types-lazy-results.csv"
        ),
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=(
            script_path.parent
            / "greedy_k2_partition_bundled_two_sensor_types-lazy-solutions"
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
