#!/usr/bin/env python3
"""Run the exact ILP for the minimum 1-identifying-code problem.

By default, every graph in ``standardized_dataset/`` is processed in
nondecreasing order of its declared number of vertices.  Each graph runs in
an isolated worker with an eight-hour wall-clock limit and a 64-GiB RAM
limit.  Completed runs are written to a resumable CSV file and every
optimal code is stored in a separate text file.

The ILP is the standard set-cover formulation.  A binary variable ``x_c``
indicates whether vertex ``c`` is selected.  Closed-neighborhood coverage is
enforced for every vertex and, for every pair ``u, v``, at least one selected
vertex must belong to ``N[u]`` symmetric-difference ``N[v]``.  CPLEX is used
through PuLP's ``CPLEX_CMD`` interface.
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import pulp


Graph: TypeAlias = dict[int, set[int]]
GraphInput: TypeAlias = Mapping[int, set[int]] | Sequence[Sequence[int]]

DEFAULT_TIMEOUT_SECONDS = 8 * 60 * 60
DEFAULT_RAM_LIMIT_GB = 64.0
SUPPORTED_SUFFIXES = {".txt", ".mtx", ".edges"}


class IdentifyingCodeInfeasible(ValueError):
    """Raised when two vertices have identical closed neighborhoods."""


class SolverLimitReached(RuntimeError):
    """Raised when CPLEX stops before proving optimality."""


def declared_vertex_count(path: Path) -> int:
    """Read just the header needed to sort standardized graphs by |V|."""
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
    """Read the standardized edge-list format, including isolated vertices."""
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
            raise ValueError(
                f"Dong {line_number}: nhan dinh phai la so nguyen"
            ) from exc

        u = vertices[0]
        graph.setdefault(u, set())
        if len(vertices) == 2:
            v = vertices[1]
            graph.setdefault(v, set())
            if u != v:
                graph[u].add(v)
                graph[v].add(u)

    if declared_vertices is not None:
        if graph and (min(graph) < 1 or max(graph) > declared_vertices):
            raise ValueError("Nhan dinh vuot qua kich thuoc graph da khai bao")
        for vertex in range(1, declared_vertices + 1):
            graph.setdefault(vertex, set())

    if not graph:
        raise ValueError("Graph rong")
    return graph


def _read_matrix_market(lines: list[str]) -> Graph:
    """Read a square Matrix Market coordinate graph as an undirected graph."""
    data_lines = [
        line.strip()
        for line in lines[1:]
        if line.strip() and not line.lstrip().startswith("%")
    ]
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
            raise ValueError(
                f"Entry Matrix Market {entry_number} khong hop le"
            ) from exc
        if not (1 <= u <= number_of_rows and 1 <= v <= number_of_rows):
            raise ValueError(
                f"Entry Matrix Market {entry_number} vuot ngoai kich thuoc"
            )
        actual_entries += 1
        if u != v:
            graph[u].add(v)
            graph[v].add(u)

    if actual_entries != declared_entries:
        raise ValueError(
            f"Matrix Market khai bao {declared_entries} entries, "
            f"doc duoc {actual_entries}"
        )
    return graph


def read_graph(path: str | Path) -> Graph:
    """Read a standardized edge-list or Matrix Market graph."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    first_line = next((line.strip() for line in lines if line.strip()), "")
    if first_line.lstrip("%").lower().startswith("matrixmarket"):
        return _read_matrix_market(lines)
    return _read_edge_list(lines)


def _as_graph(graph: GraphInput) -> Graph:
    """Normalize either an adjacency mapping or a 0/1 adjacency matrix."""
    if isinstance(graph, Mapping):
        normalized = {
            int(vertex): {int(neighbor) for neighbor in neighbors}
            for vertex, neighbors in graph.items()
        }
        if not normalized:
            raise ValueError("Graph rong")
        vertices = set(normalized)
        for vertex, neighbors in normalized.items():
            unknown = neighbors - vertices
            if unknown:
                raise ValueError(
                    f"Dinh {vertex} co lang gieng khong thuoc graph: {sorted(unknown)}"
                )
            neighbors.discard(vertex)
        for vertex, neighbors in normalized.items():
            for neighbor in neighbors:
                if vertex not in normalized[neighbor]:
                    raise ValueError("Graph phai vo huong")
        return normalized

    number_of_vertices = len(graph)
    if number_of_vertices == 0:
        raise ValueError("Graph rong")
    normalized = {vertex: set() for vertex in range(number_of_vertices)}
    for row_index, row in enumerate(graph):
        if len(row) != number_of_vertices:
            raise ValueError("Ma tran ke phai vuong")
        for column_index, entry in enumerate(row):
            if entry not in (0, 1, False, True):
                raise ValueError("Ma tran ke chi duoc chua 0/1")
            if row_index != column_index and bool(entry):
                normalized[row_index].add(column_index)
    for vertex, neighbors in normalized.items():
        for neighbor in neighbors:
            if vertex not in normalized[neighbor]:
                raise ValueError("Ma tran ke phai doi xung")
    return normalized


def closed_neighborhoods(graph: Mapping[int, set[int]]) -> dict[int, set[int]]:
    """Return ``N[v]`` for every vertex of an undirected graph."""
    return {vertex: set(neighbors) | {vertex} for vertex, neighbors in graph.items()}


def _check_twin_free(neighborhoods: Mapping[int, set[int]]) -> None:
    """Reject graphs for which no identifying code can exist."""
    seen: dict[frozenset[int], int] = {}
    for vertex in sorted(neighborhoods):
        neighborhood = frozenset(neighborhoods[vertex])
        if neighborhood in seen:
            raise IdentifyingCodeInfeasible(
                "Graph khong ton tai identifying code: hai dinh "
                f"{seen[neighborhood]} va {vertex} co cung lan can dong"
            )
        seen[neighborhood] = vertex


def minimum_identifying_code(
    graph: GraphInput,
    *,
    time_limit_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    max_memory_mb: int | None = None,
    cplex_path: str | None = None,
    msg: bool = False,
) -> tuple[list[int], int]:
    """Solve the minimum identifying-code ILP with ``pulp.CPLEX_CMD``.

    ``graph`` may be an adjacency mapping (the batch runner's native form) or
    a square 0/1 adjacency matrix (the form used by the reference program in
    ``BMF/identifying_code_ILP.py``).
    """
    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds phai lon hon 0")

    algorithm_start = time.perf_counter()
    normalized = _as_graph(graph)
    vertices = sorted(normalized)
    neighborhoods = closed_neighborhoods(normalized)
    _check_twin_free(neighborhoods)

    problem = pulp.LpProblem("Minimum_Identifying_Code", pulp.LpMinimize)
    selected = {
        vertex: pulp.LpVariable(f"x_{index}", cat=pulp.LpBinary)
        for index, vertex in enumerate(vertices)
    }
    problem += pulp.lpSum(selected.values()), "code_size"

    for index, vertex in enumerate(vertices):
        problem += (
            pulp.lpSum(selected[codeword] for codeword in neighborhoods[vertex]) >= 1,
            f"cover_{index}",
        )

    for left_index, left in enumerate(vertices):
        for right_index in range(left_index + 1, len(vertices)):
            right = vertices[right_index]
            distinguishing_vertices = neighborhoods[left] ^ neighborhoods[right]
            # _check_twin_free() guarantees this set is nonempty.
            problem += (
                pulp.lpSum(selected[codeword] for codeword in distinguishing_vertices)
                >= 1,
                f"separate_{left_index}_{right_index}",
            )

    solver_time_limit = time_limit_seconds
    if time_limit_seconds is not None:
        remaining = time_limit_seconds - (time.perf_counter() - algorithm_start)
        if remaining <= 0:
            raise SolverLimitReached("Het time limit trong khi xay dung mo hinh ILP")
        # Leave a short interval for PuLP to parse CPLEX's solution and for the
        # worker to persist its result before the parent wall-clock deadline.
        solver_time_limit = max(0.001, remaining - min(1.0, remaining / 10))

    solver = pulp.CPLEX_CMD(
        path=cplex_path,
        msg=msg,
        timeLimit=solver_time_limit,
        maxMemory=max_memory_mb,
    )
    if not solver.available():
        raise RuntimeError(
            "Khong tim thay CPLEX executable; hay them 'cplex' vao PATH "
            "hoac dung --cplex-path"
        )

    try:
        status = problem.solve(solver)
    except pulp.PulpSolverError as exc:
        solver_error = str(exc).lower()
        if "time limit" in solver_error:
            raise SolverLimitReached(f"CPLEX dung do time limit: {exc}") from exc
        if "memory limit" in solver_error:
            raise MemoryError(f"CPLEX dung do vuot gioi han RAM: {exc}") from exc
        raise

    if problem.sol_status == pulp.LpSolutionIntegerFeasible:
        raise SolverLimitReached(
            "CPLEX tim thay nghiem kha thi nhung chua chung minh toi uu "
            "truoc khi cham time limit"
        )
    if status == pulp.LpStatusInfeasible:
        raise IdentifyingCodeInfeasible("Mo hinh ILP khong kha thi")
    if status != pulp.LpStatusOptimal or problem.sol_status != pulp.LpSolutionOptimal:
        raise RuntimeError(
            "CPLEX khong chung minh duoc toi uu: "
            f"status={pulp.LpStatus.get(status, status)}, "
            f"solution_status={pulp.LpSolution.get(problem.sol_status, problem.sol_status)}"
        )

    code = [
        vertex
        for vertex in vertices
        if selected[vertex].varValue is not None and selected[vertex].varValue > 0.5
    ]
    objective = pulp.value(problem.objective)
    if objective is None:
        raise RuntimeError("CPLEX khong tra ve gia tri muc tieu")
    return code, int(round(objective))


def is_identifying_code(
    graph: Mapping[int, set[int]], code: set[int]
) -> tuple[bool, str]:
    """Independently validate domination and pairwise identification."""
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
    return True, "Nghiem toi uu thoa man tinh chat Identifying Code"


def _peak_ram_bytes() -> int:
    """Return peak RSS; macOS reports bytes while Linux reports KiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _set_ram_limit(limit_bytes: int) -> None:
    """Apply RLIMIT_AS where supported; the parent also monitors total RSS."""
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    effective_limit = (
        min(limit_bytes, hard) if hard != resource.RLIM_INFINITY else limit_bytes
    )
    try:
        resource.setrlimit(resource.RLIMIT_AS, (effective_limit, hard))
    except (OSError, ValueError):
        # Darwin commonly rejects RLIMIT_AS changes.  The parent still checks
        # the combined RSS of the worker and the CPLEX child process.
        pass


def _write_json_atomic(path: Path, result: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_solution_atomic(path: Path, code: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(map(str, sorted(code))) + "\n", encoding="utf-8")
    temporary.replace(path)


def result_template(graph_path: Path) -> dict[str, object]:
    return {
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


def run_worker(
    graph_path: Path,
    result_path: Path,
    solution_directory: Path,
    ram_limit_bytes: int,
    time_limit_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    cplex_path: str | None = None,
) -> int:
    """Solve and independently verify one dataset in a worker process."""
    worker_start = time.perf_counter()
    result = result_template(graph_path)
    try:
        graph = read_graph(graph_path)
        result["vertices"] = len(graph)
        result["edges"] = sum(map(len, graph.values())) // 2

        elapsed_before_algorithm = time.perf_counter() - worker_start
        remaining_time = time_limit_seconds - elapsed_before_algorithm
        if remaining_time <= 0:
            raise SolverLimitReached("Het time limit trong khi doc graph")

        algorithm_start = time.perf_counter()
        code, objective = minimum_identifying_code(
            graph,
            time_limit_seconds=remaining_time,
            max_memory_mb=max(1, ram_limit_bytes // 1024**2),
            cplex_path=cplex_path,
            msg=False,
        )
        algorithm_elapsed = time.perf_counter() - algorithm_start
        valid, message = is_identifying_code(graph, set(code))

        if objective != len(code):
            raise RuntimeError(
                f"Objective={objective} khong khop code_size={len(code)}"
            )
        if not valid:
            raise RuntimeError(f"CPLEX tra ve nghiem khong hop le: {message}")

        solution_path = solution_directory / f"{graph_path.name}.code.txt"
        _write_solution_atomic(solution_path, code)
        result.update(
            status="OPTIMAL",
            code_size=objective,
            elapsed_seconds=f"{algorithm_elapsed:.9f}",
            solution_file=str(solution_path),
            detail=message,
        )
    except IdentifyingCodeInfeasible as exc:
        result.update(
            status="INFEASIBLE",
            elapsed_seconds=f"{time.perf_counter() - worker_start:.9f}",
            detail=str(exc),
        )
    except SolverLimitReached as exc:
        result.update(
            status="TIMEOUT",
            elapsed_seconds=f"{time.perf_counter() - worker_start:.9f}",
            detail=str(exc),
        )
    except MemoryError as exc:
        result.update(
            status="RAM_LIMITED",
            elapsed_seconds=f"{time.perf_counter() - worker_start:.9f}",
            detail=str(exc) or "Vuot gioi han RAM",
        )
    except Exception as exc:
        result.update(
            status="INVALID",
            elapsed_seconds=f"{time.perf_counter() - worker_start:.9f}",
            detail=f"{type(exc).__name__}: {exc}",
        )
    finally:
        result["peak_ram_bytes"] = _peak_ram_bytes()
        _write_json_atomic(result_path, result)
    return 0


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_dataset_process(
    script_path: Path,
    graph_path: Path,
    solution_directory: Path,
    timeout_seconds: float,
    ram_limit_bytes: int,
    cplex_path: str | None = None,
) -> dict[str, object]:
    """Run one graph in isolation and enforce wall-time and aggregate RSS."""
    result_file_handle = tempfile.NamedTemporaryFile(
        prefix="k1-ilp-result-", suffix=".json", delete=False
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
        "--worker-time-limit",
        str(timeout_seconds),
    ]
    if cplex_path is not None:
        command.extend(("--cplex-path", cplex_path))

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
        _kill_process_group(process)
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
        _kill_process_group(process)
        _, stderr = process.communicate()
        result_path.unlink(missing_ok=True)
        return {
            **result_template(graph_path),
            "status": limit_status,
            "elapsed_seconds": f"{time.perf_counter() - start:.9f}",
            "peak_ram_bytes": observed_peak_ram,
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

        detail = stderr.strip()[-1000:] or f"worker exit code {process.returncode}"
        killed_for_memory = process.returncode in {
            -signal.SIGKILL,
            -signal.SIGABRT,
            -signal.SIGSEGV,
        }
        return {
            **result_template(graph_path),
            "status": "RAM_LIMITED" if killed_for_memory else "INVALID",
            "elapsed_seconds": f"{time.perf_counter() - start:.9f}",
            "peak_ram_bytes": observed_peak_ram,
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


def completed_datasets(
    csv_path: Path, rerun_statuses: set[str] | None = None
) -> set[str]:
    """Return datasets that should be skipped by a resumable batch run."""
    rerun_statuses = {status.upper() for status in (rerun_statuses or set())}
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"dataset", "status"} <= set(
            reader.fieldnames
        ):
            raise ValueError(f"CSV {csv_path} thieu cot bat buoc dataset/status")
        latest_status: dict[str, str] = {}
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            status = (row.get("status") or "").strip().upper()
            if dataset and status:
                latest_status[dataset] = status
        return {
            dataset
            for dataset, status in latest_status.items()
            if status not in rerun_statuses
        }


def append_csv_result(csv_path: Path, result: Mapping[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in RESULT_FIELDS})
        output.flush()


def upsert_csv_result(csv_path: Path, result: Mapping[str, object]) -> None:
    """Atomically replace a dataset row, or append it when it is new."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        append_csv_result(csv_path, result)
        return

    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not set(RESULT_FIELDS) <= set(reader.fieldnames):
            raise ValueError(f"CSV {csv_path} khong dung schema ket qua")
        rows = list(reader)

    dataset = str(result.get("dataset") or "").strip()
    replacement = {field: result.get(field, "") for field in RESULT_FIELDS}
    new_rows: list[Mapping[str, object]] = []
    replaced = False
    for row in rows:
        if (row.get("dataset") or "").strip() == dataset:
            if not replaced:
                new_rows.append(replacement)
                replaced = True
        else:
            new_rows.append(row)
    if not replaced:
        new_rows.append(replacement)

    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in RESULT_FIELDS}
            for row in new_rows
        )
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(csv_path)


def _resolve_cplex_path(path: str | None) -> str:
    solver = pulp.CPLEX_CMD(path=path, msg=False)
    available = solver.available()
    if not available:
        requested = path or "cplex"
        raise RuntimeError(f"Khong tim thay CPLEX executable: {requested}")
    return str(available)


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
    parser.add_argument(
        "--cplex-path",
        help="duong dan CPLEX executable; mac dinh tim 'cplex' trong PATH",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_path.parent / "k1-results.csv",
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=script_path.parent / "k1-solutions",
    )
    parser.add_argument(
        "--rerun-status",
        action="append",
        default=[],
        metavar="STATUS",
        help=(
            "chi chay lai va thay the cac dong co status nay trong CSV; "
            "co the lap option, vi du --rerun-status INVALID"
        ),
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-ram-bytes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-time-limit", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker is not None:
        if args.result_file is None:
            parser.error("--worker requires --result-file")
        if args.worker_ram_bytes is None:
            parser.error("--worker requires --worker-ram-bytes")
        if args.worker_time_limit is None:
            parser.error("--worker requires --worker-time-limit")
        _set_ram_limit(args.worker_ram_bytes)
        raise SystemExit(
            run_worker(
                args.worker,
                args.result_file,
                args.solution_dir,
                args.worker_ram_bytes,
                args.worker_time_limit,
                args.cplex_path,
            )
        )

    if args.timeout <= 0:
        parser.error("--timeout phai lon hon 0")
    if args.ram_limit_gb <= 0:
        parser.error("--ram-limit-gb phai lon hon 0")
    try:
        cplex_path = _resolve_cplex_path(args.cplex_path)
    except RuntimeError as exc:
        parser.error(str(exc))

    dataset_directory = project_directory / "standardized_dataset"
    graph_paths = args.graphs or [
        path
        for path in dataset_directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not graph_paths:
        parser.error(f"Khong tim thay graph trong {dataset_directory}")
    unsupported = [
        path for path in graph_paths if path.suffix.lower() not in SUPPORTED_SUFFIXES
    ]
    if unsupported:
        parser.error(
            "Dinh dang graph khong duoc ho tro: "
            + ", ".join(str(path) for path in unsupported)
        )
    try:
        graph_paths = sorted(
            graph_paths,
            key=lambda path: (declared_vertex_count(path), path.name.lower()),
        )
    except (OSError, ValueError) as exc:
        parser.error(f"Khong the sap xep graph theo so dinh: {exc}")

    ram_limit_bytes = int(args.ram_limit_gb * 1024**3)
    try:
        rerun_statuses = {
            status.strip().upper()
            for status in args.rerun_status
            if status.strip()
        }
        recorded_datasets = completed_datasets(args.output_csv)
        already_completed = completed_datasets(args.output_csv, rerun_statuses)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(f"Khong the doc CSV ket qua cu: {exc}")
    if rerun_statuses:
        pending_graphs = [
            path
            for path in graph_paths
            if path.name in recorded_datasets and path.name not in already_completed
        ]
    else:
        pending_graphs = [
            path for path in graph_paths if path.name not in already_completed
        ]
    selection_summary = (
        f"chon chay lai={len(pending_graphs)} case co status "
        f"{','.join(sorted(rerun_statuses))}"
        if rerun_statuses
        else (
            f"da co ket qua={len(graph_paths) - len(pending_graphs)}, "
            f"con lai={len(pending_graphs)}"
        )
    )
    print(
        f"Tim thay {len(graph_paths)} graph; {selection_summary}. "
        f"Timeout={args.timeout:g}s, RAM limit={args.ram_limit_gb:g} GiB/graph. "
        "Solver=CPLEX_CMD. Thu tu: so dinh tang dan.",
        flush=True,
    )
    if not pending_graphs:
        print(f"Khong co graph nao can chay. Ket qua: {args.output_csv}")
        return

    for index, graph_path in enumerate(pending_graphs, start=1):
        result = run_dataset_process(
            script_path,
            graph_path.resolve(),
            args.solution_dir.resolve(),
            args.timeout,
            ram_limit_bytes,
            cplex_path,
        )
        upsert_csv_result(args.output_csv, result)
        print(
            f"[{index:02d}/{len(pending_graphs):02d}] {graph_path.name}: "
            f"STATUS={result['status']} | time={result['elapsed_seconds']}s | "
            f"code_size={result['code_size'] or 'N/A'}",
            flush=True,
        )
        if result["status"] not in {"OPTIMAL", "INFEASIBLE"}:
            print(f"  detail: {result['detail']}", flush=True)
    print(f"Da ghi tong ket: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
