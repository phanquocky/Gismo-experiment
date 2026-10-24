#!/usr/bin/env python3
"""Run the explicit GICS ILP from ``k2.md`` with CPLEX.

For every selected physical sensor ``v``, the observation contains two bits:
whether ``v`` itself has failed at time t0 and whether a vertex in ``N[v]``
has failed at time t1.  Every failure scenario with cardinality from one to
``k`` must be covered and separated from every other scenario.  The default is
``k=2``.

This implementation intentionally builds every Equation (4) row from the new
``k2.md``.  It does not use lazy/cutting-plane generation.  The batch runner
keeps the experimental environment of ``k1.py``: increasing graph size,
isolated workers, an eight-hour wall-clock limit, a 64-GiB aggregate RAM limit,
resumable CSV output, and CPLEX through PuLP's ``CPLEX_CMD`` interface.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Hashable, Mapping, Sequence
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, TypeAlias

import pulp


Node: TypeAlias = Hashable
Graph: TypeAlias = dict[Node, set[Node]]
GraphInput: TypeAlias = Mapping[Node, set[Node]] | Sequence[Sequence[int]] | Any
Scenario: TypeAlias = frozenset[Node]

DEFAULT_K = 2
DEFAULT_TIMEOUT_SECONDS = 8 * 60 * 60
DEFAULT_RAM_LIMIT_GB = 64.0
SUPPORTED_SUFFIXES = {".txt", ".mtx", ".edges"}


class IdentifyingCodeInfeasible(ValueError):
    """Raised when CPLEX proves the explicit GICS model infeasible."""


class SolverLimitReached(RuntimeError):
    """Raised when model construction or CPLEX reaches the time limit."""


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
    """Normalize a NetworkX-like graph, adjacency mapping, or 0/1 matrix."""
    if all(
        hasattr(graph, attribute)
        for attribute in ("is_directed", "is_multigraph", "nodes", "neighbors")
    ):
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError("Expected a simple undirected graph.")
        nodes = list(graph.nodes())
        if not nodes:
            raise ValueError("Require a nonempty graph.")
        normalized = {vertex: set(graph.neighbors(vertex)) for vertex in nodes}
        for vertex, neighbors in normalized.items():
            neighbors.discard(vertex)
        return normalized

    if isinstance(graph, Mapping):
        normalized = {vertex: set(neighbors) for vertex, neighbors in graph.items()}
        if not normalized:
            raise ValueError("Require a nonempty graph.")
        vertices = set(normalized)
        for vertex, neighbors in normalized.items():
            unknown = neighbors - vertices
            if unknown:
                raise ValueError(
                    f"Dinh {vertex!r} co lang gieng khong thuoc graph: "
                    f"{sorted(map(repr, unknown))}"
                )
            neighbors.discard(vertex)
        for vertex, neighbors in normalized.items():
            for neighbor in neighbors:
                if vertex not in normalized[neighbor]:
                    raise ValueError("Expected a simple undirected graph.")
        return normalized

    number_of_vertices = len(graph)
    if number_of_vertices == 0:
        raise ValueError("Require a nonempty graph.")
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


def _validate_k(k: int, number_of_vertices: int) -> None:
    if (
        not isinstance(k, int)
        or isinstance(k, bool)
        or not 1 <= k <= number_of_vertices
    ):
        raise ValueError("Require a nonempty graph and integer 1 <= k <= |V|.")


def prepare_scenarios(
    graph: GraphInput, k: int
) -> tuple[list[Node], dict[Node, frozenset[Node]], list[Scenario], list[Scenario]]:
    """Build exactly the scenario and union-neighbourhood lists in k2.md."""
    normalized = _as_graph(graph)
    nodes = list(normalized)
    _validate_k(k, len(nodes))
    closed = {vertex: frozenset(normalized[vertex]) | {vertex} for vertex in nodes}
    scenarios = [
        frozenset(state)
        for size in range(1, k + 1)
        for state in combinations(nodes, size)
    ]
    neighborhoods = [
        frozenset().union(*(closed[vertex] for vertex in state)) for state in scenarios
    ]
    return nodes, closed, scenarios, neighborhoods


def scenario_count(number_of_vertices: int, k: int) -> int:
    """Return sum(C(n,i), i=1..k)."""
    _validate_k(k, number_of_vertices)
    return sum(comb(number_of_vertices, size) for size in range(1, k + 1))


def verify_code(graph: GraphInput, selected: Sequence[Node], k: int) -> bool:
    """Verify actual two-time signatures, including the no-failure state."""
    normalized = _as_graph(graph)
    nodes = list(normalized)
    _validate_k(k, len(nodes))
    selected_set = set(selected)
    if not selected_set <= set(nodes):
        return False

    seen: set[tuple[frozenset[Node], frozenset[Node]]] = set()
    for size in range(k + 1):
        for state_tuple in combinations(nodes, size):
            state = set(state_tuple)
            at_t0 = frozenset(selected_set & state)
            at_t1 = frozenset(
                sensor
                for sensor in selected_set
                if any(
                    sensor == failed or sensor in normalized[failed] for failed in state
                )
            )
            signature = (at_t0, at_t1)
            if signature in seen:
                return False
            seen.add(signature)
    return True


def _check_deadline(deadline: float | None, message: str) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise SolverLimitReached(message)


def build_gics_model(
    graph: GraphInput,
    k: int = DEFAULT_K,
    *,
    deadline: float | None = None,
) -> tuple[
    pulp.LpProblem,
    dict[Node, pulp.LpVariable],
    dict[Node, pulp.LpVariable],
    dict[str, int],
]:
    """Build Equations (2)--(5) from the new ``k2.md`` explicitly."""
    nodes, closed, scenarios, neighborhoods = prepare_scenarios(graph, k)
    _check_deadline(deadline, "Het time limit khi tao cac trang thai")

    model = pulp.LpProblem(f"GICS_k{k}", pulp.LpMinimize)
    x = {
        vertex: pulp.LpVariable(f"x_{index}", cat=pulp.LpBinary)
        for index, vertex in enumerate(nodes)
    }
    y = {
        vertex: pulp.LpVariable(
            f"y_{index}", lowBound=0, upBound=len(closed[vertex]), cat=pulp.LpInteger
        )
        for index, vertex in enumerate(nodes)
    }

    # Equation (2): y[v] counts selected sensors in N[v].
    for index, vertex in enumerate(nodes):
        model += (
            y[vertex] == pulp.lpSum(x[neighbor] for neighbor in closed[vertex]),
            f"seepage_{index}",
        )

    # Equation (3): every vertex is observed at t1.
    for index, vertex in enumerate(nodes):
        model += y[vertex] >= 1, f"coverage_{index}"

    # Equation (4): add every pair of nonempty scenarios explicitly.
    for pair_index, (left_index, right_index) in enumerate(
        combinations(range(len(scenarios)), 2)
    ):
        if pair_index & 0x3FFF == 0:
            _check_deadline(
                deadline,
                "Het time limit khi dung rang buoc uniqueness Equation (4)",
            )
        support = (scenarios[left_index] ^ scenarios[right_index]) | (
            neighborhoods[left_index] ^ neighborhoods[right_index]
        )
        model += (
            pulp.lpSum(x[vertex] for vertex in support) >= 1,
            f"unique_{left_index}_{right_index}",
        )

    # Equation (5): one unit of cost per physical sensor.
    model += pulp.lpSum(x.values()), "sensor_count"

    number_of_scenarios = len(scenarios)
    metadata = {
        "nodes": len(nodes),
        "k": k,
        "nonempty_scenarios": number_of_scenarios,
        "variables": 2 * len(nodes),
        "constraints": 2 * len(nodes) + comb(number_of_scenarios, 2),
    }
    return model, x, y, metadata


def _solver_time_limit(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise SolverLimitReached("Het time limit trong khi xay dung mo hinh ILP")
    return max(0.001, remaining - min(1.0, remaining / 10))


def solve_gics(
    graph: GraphInput,
    k: int = DEFAULT_K,
    time_limit: float | None = None,
    threads: int | None = None,
    log_output: bool = True,
    *,
    max_memory_mb: int | None = None,
    cplex_path: str | None = None,
) -> dict[str, object]:
    """Build and solve the explicit GICS model using ``CPLEX_CMD``."""
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be finite and positive.")
    if threads is not None and (
        not isinstance(threads, int) or isinstance(threads, bool) or threads < 0
    ):
        raise ValueError("threads must be a nonnegative integer (0 = automatic).")

    started = time.perf_counter()
    deadline = None if time_limit is None else started + time_limit
    model, x, _y, metadata = build_gics_model(graph, k, deadline=deadline)
    build_seconds = time.perf_counter() - started

    solver = pulp.CPLEX_CMD(
        path=cplex_path,
        msg=log_output,
        timeLimit=_solver_time_limit(deadline),
        gapRel=0,
        gapAbs=0,
        threads=threads,
        maxMemory=max_memory_mb,
    )
    if not solver.available():
        raise RuntimeError(
            "Khong tim thay CPLEX executable; hay them 'cplex' vao PATH "
            "hoac dung --cplex-path"
        )

    solve_started = time.perf_counter()
    try:
        status_code = model.solve(solver)
    except pulp.PulpSolverError as exc:
        message = str(exc).lower()
        if "time limit" in message:
            raise SolverLimitReached(f"CPLEX dung do time limit: {exc}") from exc
        if "memory limit" in message:
            raise MemoryError(f"CPLEX dung do vuot gioi han RAM: {exc}") from exc
        raise
    solve_seconds = time.perf_counter() - solve_started

    is_optimal = (
        status_code == pulp.LpStatusOptimal
        and model.sol_status == pulp.LpSolutionOptimal
    )
    has_solution = model.sol_status in {
        pulp.LpSolutionOptimal,
        pulp.LpSolutionIntegerFeasible,
    }
    result: dict[str, object] = {
        **metadata,
        "status": pulp.LpStatus.get(status_code, str(status_code)),
        "is_optimal": is_optimal,
        "has_solution": has_solution,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    if not has_solution:
        result.update(selected=None, objective=None, verified=False)
        return result

    selected = [
        vertex
        for vertex in x
        if x[vertex].varValue is not None and x[vertex].varValue > 0.5
    ]
    if not verify_code(graph, selected, k):
        raise RuntimeError(
            "CPLEX solution failed independent two-time signature verification."
        )
    result.update(
        selected=selected,
        objective=len(selected),
        verified=True,
        best_bound=None,
        relative_gap=0.0 if is_optimal else None,
    )
    return result


def minimum_identifying_code(
    graph: GraphInput,
    *,
    k: int = DEFAULT_K,
    time_limit_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    max_memory_mb: int | None = None,
    cplex_path: str | None = None,
    threads: int | None = None,
    msg: bool = False,
) -> tuple[list[Node], int]:
    """Compatibility wrapper returning only a proven-optimal code."""
    result = solve_gics(
        graph,
        k=k,
        time_limit=time_limit_seconds,
        threads=threads,
        log_output=msg,
        max_memory_mb=max_memory_mb,
        cplex_path=cplex_path,
    )
    if result["is_optimal"]:
        selected = result["selected"]
        assert isinstance(selected, list)
        return selected, int(result["objective"])
    if result["status"] == "Infeasible":
        raise IdentifyingCodeInfeasible("Mo hinh GICS khong kha thi")
    if result["has_solution"]:
        raise SolverLimitReached(
            "CPLEX tim thay nghiem kha thi nhung chua chung minh toi uu"
        )
    raise SolverLimitReached(
        f"CPLEX khong tra ve nghiem toi uu: status={result['status']}"
    )


def is_identifying_code(
    graph: GraphInput, code: Sequence[Node], k: int = DEFAULT_K
) -> tuple[bool, str]:
    """Compatibility validator with an explanatory message."""
    try:
        valid = verify_code(graph, code, k)
    except ValueError as exc:
        return False, str(exc)
    if not valid:
        return False, "Ton tai hai kich ban co cung chu ky hai thoi diem"
    return True, f"Moi kich ban tu 0 den {k} dinh loi co chu ky rieng"


def _peak_ram_bytes() -> int:
    """Return peak RSS; macOS reports bytes while Linux reports KiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _set_ram_limit(limit_bytes: int) -> None:
    """Apply RLIMIT_AS where supported; parent also monitors aggregate RSS."""
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    effective_limit = (
        min(limit_bytes, hard) if hard != resource.RLIM_INFINITY else limit_bytes
    )
    try:
        resource.setrlimit(resource.RLIMIT_AS, (effective_limit, hard))
    except (OSError, ValueError):
        pass


def _write_json_atomic(path: Path, result: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _node_sort_key(node: Node) -> tuple[str, str]:
    return type(node).__name__, repr(node)


def _write_solution_atomic(path: Path, code: Sequence[Node]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "\n".join(map(str, sorted(code, key=_node_sort_key))) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def result_template(graph_path: Path, k: int) -> dict[str, object]:
    return {
        "dataset": graph_path.name,
        "k": k,
        "status": "INVALID",
        "vertices": "",
        "edges": "",
        "nonempty_scenarios": "",
        "variables": "",
        "constraints": "",
        "code_size": "",
        "build_seconds": "",
        "solve_seconds": "",
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
    k: int = DEFAULT_K,
    threads: int | None = None,
) -> int:
    """Solve and independently verify one dataset in a worker process."""
    worker_started = time.perf_counter()
    output = result_template(graph_path, k)
    try:
        graph = read_graph(graph_path)
        output["vertices"] = len(graph)
        output["edges"] = sum(map(len, graph.values())) // 2
        output["nonempty_scenarios"] = scenario_count(len(graph), k)

        remaining = time_limit_seconds - (time.perf_counter() - worker_started)
        if remaining <= 0:
            raise SolverLimitReached("Het time limit trong khi doc graph")
        solved = solve_gics(
            graph,
            k=k,
            time_limit=remaining,
            threads=threads,
            log_output=False,
            max_memory_mb=max(1, ram_limit_bytes // 1024**2),
            cplex_path=cplex_path,
        )
        output.update(
            variables=solved["variables"],
            constraints=solved["constraints"],
            build_seconds=f"{float(solved['build_seconds']):.9f}",
            solve_seconds=f"{float(solved['solve_seconds']):.9f}",
            elapsed_seconds=f"{float(solved['total_seconds']):.9f}",
        )

        if solved["is_optimal"]:
            selected = solved["selected"]
            assert isinstance(selected, list)
            solution_path = solution_directory / f"{graph_path.name}.k{k}.code.txt"
            _write_solution_atomic(solution_path, selected)
            output.update(
                status="OPTIMAL",
                code_size=solved["objective"],
                solution_file=str(solution_path),
                detail=f"Da verify moi kich ban tu 0 den {k} dinh loi",
            )
        elif solved["status"] == "Infeasible":
            output.update(
                status="INFEASIBLE",
                detail="CPLEX da chung minh mo hinh khong kha thi",
            )
        else:
            incumbent = solved["objective"] if solved["has_solution"] else None
            raise SolverLimitReached(
                "CPLEX chua chung minh toi uu"
                + (f"; incumbent={incumbent}" if incumbent is not None else "")
            )
    except IdentifyingCodeInfeasible as exc:
        output.update(status="INFEASIBLE", detail=str(exc))
    except SolverLimitReached as exc:
        output.update(
            status="TIMEOUT",
            elapsed_seconds=output["elapsed_seconds"]
            or f"{time.perf_counter() - worker_started:.9f}",
            detail=str(exc),
        )
    except MemoryError as exc:
        output.update(
            status="RAM_LIMITED",
            elapsed_seconds=f"{time.perf_counter() - worker_started:.9f}",
            detail=str(exc) or "Vuot gioi han RAM",
        )
    except Exception as exc:
        output.update(
            status="INVALID",
            elapsed_seconds=f"{time.perf_counter() - worker_started:.9f}",
            detail=f"{type(exc).__name__}: {exc}",
        )
    finally:
        output["peak_ram_bytes"] = _peak_ram_bytes()
        _write_json_atomic(result_path, output)
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
    k: int = DEFAULT_K,
    threads: int | None = None,
) -> dict[str, object]:
    """Run one graph in isolation and enforce wall-time and aggregate RSS."""
    result_file_handle = tempfile.NamedTemporaryFile(
        prefix=f"k{k}-ilp-result-", suffix=".json", delete=False
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
        "--k",
        str(k),
    ]
    if cplex_path is not None:
        command.extend(("--cplex-path", cplex_path))
    if threads is not None:
        command.extend(("--threads", str(threads)))

    started = time.perf_counter()
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
        elapsed = time.perf_counter() - started
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
        process.communicate()
        result_path.unlink(missing_ok=True)
        return {
            **result_template(graph_path, k),
            "status": limit_status,
            "elapsed_seconds": f"{time.perf_counter() - started:.9f}",
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
            **result_template(graph_path, k),
            "status": "RAM_LIMITED" if killed_for_memory else "INVALID",
            "elapsed_seconds": f"{time.perf_counter() - started:.9f}",
            "peak_ram_bytes": observed_peak_ram,
            "detail": detail,
        }
    finally:
        result_path.unlink(missing_ok=True)


RESULT_FIELDS = [
    "dataset",
    "k",
    "status",
    "vertices",
    "edges",
    "nonempty_scenarios",
    "variables",
    "constraints",
    "code_size",
    "build_seconds",
    "solve_seconds",
    "elapsed_seconds",
    "peak_ram_bytes",
    "solution_file",
    "detail",
]


def completed_runs(
    csv_path: Path, rerun_statuses: set[str] | None = None
) -> set[tuple[str, int]]:
    """Return (dataset, k) runs that should be skipped when resuming."""
    rerun_statuses = {status.upper() for status in (rerun_statuses or set())}
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"dataset", "k", "status"} <= set(
            reader.fieldnames
        ):
            raise ValueError(f"CSV {csv_path} thieu cot dataset/k/status")
        latest_status: dict[tuple[str, int], str] = {}
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            status = (row.get("status") or "").strip().upper()
            if dataset and status:
                latest_status[(dataset, int(row["k"]))] = status
        return {
            run for run, status in latest_status.items() if status not in rerun_statuses
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
    """Atomically replace one (dataset, k) row, or append a new run."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        append_csv_result(csv_path, result)
        return

    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not set(RESULT_FIELDS) <= set(reader.fieldnames):
            raise ValueError(f"CSV {csv_path} khong dung schema ket qua")
        rows = list(reader)

    dataset = str(result.get("dataset") or "").strip()
    run_k = int(result["k"])
    replacement = {field: result.get(field, "") for field in RESULT_FIELDS}
    new_rows: list[Mapping[str, object]] = []
    replaced = False
    for row in rows:
        same_run = (
            (row.get("dataset") or "").strip() == dataset
            and int(row["k"]) == run_k
        )
        if same_run:
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
        raise RuntimeError(f"Khong tim thay CPLEX executable: {path or 'cplex'}")
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
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--ram-limit-gb", type=float, default=DEFAULT_RAM_LIMIT_GB)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--cplex-path",
        help="duong dan CPLEX executable; mac dinh tim 'cplex' trong PATH",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_path.parent / "k2-results.csv",
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=script_path.parent / "k2-solutions",
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
        if (
            args.result_file is None
            or args.worker_ram_bytes is None
            or args.worker_time_limit is None
        ):
            parser.error("worker thieu result-file/RAM/time-limit")
        _set_ram_limit(args.worker_ram_bytes)
        raise SystemExit(
            run_worker(
                args.worker,
                args.result_file,
                args.solution_dir,
                args.worker_ram_bytes,
                args.worker_time_limit,
                args.cplex_path,
                args.k,
                args.threads,
            )
        )

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout phai huu han va lon hon 0")
    if not math.isfinite(args.ram_limit_gb) or args.ram_limit_gb <= 0:
        parser.error("--ram-limit-gb phai huu han va lon hon 0")
    if args.k < 1:
        parser.error("--k phai lon hon hoac bang 1")
    if args.threads is not None and args.threads < 0:
        parser.error("--threads phai khong am")
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

    smallest_graph_size = declared_vertex_count(graph_paths[0])
    if args.k > smallest_graph_size:
        parser.error(
            f"--k={args.k} lon hon graph nho nhat co {smallest_graph_size} dinh"
        )

    ram_limit_bytes = int(args.ram_limit_gb * 1024**3)
    try:
        rerun_statuses = {
            status.strip().upper()
            for status in args.rerun_status
            if status.strip()
        }
        recorded_runs = completed_runs(args.output_csv)
        already_completed = completed_runs(args.output_csv, rerun_statuses)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(f"Khong the doc CSV ket qua cu: {exc}")
    if rerun_statuses:
        pending_graphs = [
            path
            for path in graph_paths
            if (path.name, args.k) in recorded_runs
            and (path.name, args.k) not in already_completed
        ]
    else:
        pending_graphs = [
            path for path in graph_paths if (path.name, args.k) not in already_completed
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
        f"Tim thay {len(graph_paths)} graph; {selection_summary}. K={args.k}, "
        f"timeout={args.timeout:g}s, RAM={args.ram_limit_gb:g} GiB/graph. "
        "Mo hinh explicit Equation (2)-(5), solver=CPLEX_CMD.",
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
            args.k,
            args.threads,
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
