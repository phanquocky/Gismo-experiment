#!/usr/bin/env python3
"""Chay thuc nghiem GiSMo voi moi k trong K tren standardized_dataset/."""

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
from itertools import combinations
from pathlib import Path

import dataset_standardize


Graph = dict[int, set[int]]
FireState = tuple[int, ...]


K = [2]
DEFAULT_TIMEOUT_SECONDS = 8 * 60 * 60
DEFAULT_RAM_LIMIT_GB = 64.0
SUPPORTED_SUFFIXES = {".txt", ".mtx", ".edges"}

# Sua hai duong dan nay tren server, hoac truyen bang command line.
GISMO_BINARY = Path("/home/ndthuc/bvthach/gismo-experiment/Gismo-experiment/gismo-env/gismo/build/gismo")
ENCODE_NETWORK_PATH = Path(
    "/home/ndthuc/bvthach/gismo-experiment/Gismo-experiment/gismo-env/identifying-codes/scripts/encoding/encode_network.py"
)

RESULT_FIELDS = [
    "dataset",
    "status",
    "vertices",
    "edges",
    "code_size",
    "elapsed_seconds",
    "encoding_seconds",
    "solver_seconds",
    "peak_ram_bytes",
    "gcnf_bytes",
    "solution_file",
    "detail",
]


def temp_directory_for_k(project_directory: Path, k: int) -> Path:
    """Tra ve thu muc scratch dung chung ``temp/k<k>/`` va tao neu can."""
    temp_directory = project_directory / "temp" / f"k{k}"
    temp_directory.mkdir(parents=True, exist_ok=True)
    return temp_directory


def closed_neighborhoods(graph: Graph) -> dict[int, set[int]]:
    """Tra ve N[v] cho moi dinh va kiem tra nhan dinh ke."""
    vertices = set(graph)
    if not vertices:
        raise ValueError("Graph rong")

    for vertex, neighbors in graph.items():
        unknown = neighbors - vertices
        if unknown:
            raise ValueError(
                f"Dinh {vertex} co dinh ke khong thuoc graph: {sorted(unknown)}"
            )
    return {vertex: set(neighbors) | {vertex} for vertex, neighbors in graph.items()}


def fire_states(vertices: list[int], k: int) -> list[FireState]:
    """Liet ke moi trang thai co tu 1 den k dinh bi chay."""
    number_of_vertices = len(vertices)
    if not 1 <= k <= number_of_vertices:
        raise ValueError(f"k phai nam trong [1, {number_of_vertices}], nhan duoc {k}")

    return [
        state
        for number_of_fires in range(1, k + 1)
        for state in combinations(vertices, number_of_fires)
    ]


def build_or_output_matrix(
    graph: Graph, sensors: set[int], k: int
) -> dict[FireState, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Dung cac hang ``x | y`` bang OR cho moi trang thai chay 1..k dinh.

    Voi trang thai ``F``:

    * ``x_i = 1`` khi sensor ``i`` nam trong F;
    * ``y_i = 1`` khi sensor ``i`` quan sat duoc it nhat mot dinh trong F.

    Hai cong thuc nay tuong duong lay OR cac hang singleton cua tung dinh
    dang chay, cho ca thoi diem t0 va t1.
    """
    vertices = sorted(graph)
    neighborhoods = closed_neighborhoods(graph)
    unknown_sensors = sensors - graph.keys()
    if unknown_sensors:
        raise ValueError(f"Sensor khong thuoc graph: {sorted(unknown_sensors)}")

    states = fire_states(vertices, k)
    ordered_sensors = sorted(sensors)
    matrix: dict[FireState, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for state in states:
        x_row = tuple(int(sensor in state) for sensor in ordered_sensors)
        y_row = tuple(
            int(any(sensor in neighborhoods[vertex] for vertex in state))
            for sensor in ordered_sensors
        )
        matrix[state] = x_row, y_row
    return matrix


def _format_fire_state(state: FireState) -> str:
    return "{" + ",".join(f"v_{vertex}" for vertex in state) + "}"


def evaluate_output(
    graph: Graph, sensors: set[int], k: int
) -> tuple[bool, str]:
    """Kiem tra domination va separation bang chu ky OR ``x | y``."""
    try:
        matrix = build_or_output_matrix(graph, sensors, k)
    except ValueError as exc:
        return False, str(exc)

    seen: dict[tuple[int, ...], FireState] = {}
    for state, (x_row, y_row) in matrix.items():
        if not any(y_row):
            return (
                False,
                f"Trang thai {_format_fire_state(state)} khong duoc dominate",
            )

        signature = x_row + y_row
        if signature in seen:
            return (
                False,
                f"Hai trang thai {_format_fire_state(seen[signature])} va "
                f"{_format_fire_state(state)} khong duoc phan biet",
            )
        seen[signature] = state

    return (
        True,
        f"Output hop le cho moi trang thai co tu 1 den {k} dinh chay",
    )


def parse_gismo_ind_from_text(text: str) -> list[int]:
    """Tuong duong web-gcnf/app/utils/parse_gismo_output.py."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("c ind "):
            result: list[int] = []
            for token in line.split()[2:]:
                try:
                    variable = int(token)
                except ValueError:
                    continue
                if variable != 0:
                    result.append(variable)
            return result
    raise RuntimeError("GiSMo output khong co dong 'c ind'")


def parse_groups_from_gcnf(gcnf_path: Path) -> dict[int, int]:
    variable_to_group: dict[int, int] = {}
    group_id = 0
    with gcnf_path.open("r", encoding="utf-8") as source:
        for raw_line in source:
            if not raw_line.startswith("c grp "):
                continue
            group_id += 1
            for token in raw_line.split()[2:]:
                try:
                    variable = int(token)
                except ValueError:
                    continue
                if variable != 0:
                    variable_to_group[variable] = group_id
    if not variable_to_group:
        raise RuntimeError("GCNF khong co 'c grp'; can encode voi --two_step")
    return variable_to_group


def parse_sensor_set(gismo_output: str, gcnf_path: Path) -> list[int]:
    independent_variables = parse_gismo_ind_from_text(gismo_output)
    variable_to_group = parse_groups_from_gcnf(gcnf_path)
    try:
        return sorted({variable_to_group[var] for var in independent_variables})
    except KeyError as exc:
        raise RuntimeError(
            f"Bien {exc.args[0]} trong GiSMo output khong thuoc group nao"
        ) from exc


def edge_list_group_to_vertex(path: Path) -> dict[int, int]:
    """Khoi phuc cach encoder sap xep nhan chuoi cua file edge-list."""
    labels: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith(("%", "#")):
                continue
            fields = line.split()
            if len(fields) >= 2:
                labels.update(fields[:2])
    return {
        group: int(label)
        for group, label in enumerate(sorted(labels), start=1)
    }


def declared_vertex_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as source:
        if path.suffix.lower() == ".mtx":
            for raw_line in source:
                line = raw_line.strip()
                if not line or line.startswith("%"):
                    continue
                rows, columns = map(int, line.split()[:2])
                if rows != columns:
                    raise ValueError(f"Matrix khong vuong: {path}")
                return rows
        else:
            for raw_line in source:
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
    raise ValueError(f"Khong tim thay so dinh trong {path}")


def peak_ram_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def set_ram_limit(limit_bytes: int) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    effective = min(limit_bytes, hard) if hard != resource.RLIM_INFINITY else limit_bytes
    try:
        resource.setrlimit(resource.RLIMIT_AS, (effective, hard))
    except (OSError, ValueError):
        # Parent van theo doi RSS; macOS thuong khong cho doi RLIMIT_AS.
        pass


def write_json_atomic(path: Path, result: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def result_template(graph_path: Path) -> dict[str, object]:
    return {
        "dataset": graph_path.name,
        "status": "INVALID",
        "vertices": "",
        "edges": "",
        "code_size": "",
        "elapsed_seconds": "",
        "encoding_seconds": "",
        "solver_seconds": "",
        "peak_ram_bytes": "",
        "gcnf_bytes": "",
        "solution_file": "",
        "detail": "",
    }


def run_worker(
    graph_path: Path,
    result_path: Path,
    solution_directory: Path,
    encoder_path: Path,
    gismo_binary: Path,
    project_directory: Path,
    k: int,
) -> int:
    """Dung dung pipeline trong web-gcnf/app/routes.py cho mot graph."""
    start = time.perf_counter()
    result = result_template(graph_path)
    try:
        graph = dataset_standardize.read_graph(graph_path)
        result["vertices"] = len(graph)
        result["edges"] = sum(map(len, graph.values())) // 2

        temp_directory = temp_directory_for_k(project_directory, k)
        with tempfile.TemporaryDirectory(
            prefix=f"gismo-{graph_path.stem}-",
            dir=temp_directory,
        ) as temp:
            work_directory = Path(temp)
            gcnf_name = f"{graph_path.stem}.gcnf"
            gcnf_path = work_directory / f"k{k}" / gcnf_name

            environment = os.environ.copy()
            environment.setdefault(
                "PROJECT_DIR",
                str(project_directory / "web-gcnf" / "identifying-codes"),
            )
            # identifying_codes.py tao TEMP_*_pbs.cnf/.pbo trong os.getcwd().
            # Dat ca working directory va cac bien temp vao scratch cua graph
            # de khong phat sinh file trung gian trong repository.
            environment.update(
                TMPDIR=str(work_directory),
                TMP=str(work_directory),
                TEMP=str(work_directory),
            )

            encoding_start = time.perf_counter()
            encoded = subprocess.run(
                [
                    sys.executable,
                    str(encoder_path),
                    "-n",
                    str(graph_path),
                    "--out_dir",
                    str(work_directory),
                    "--out_file",
                    gcnf_name,
                    "--encoding",
                    "gis",
                    "--two_step",
                    "-k",
                    str(k),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                cwd=work_directory,
            )
            encoding_seconds = time.perf_counter() - encoding_start
            result["encoding_seconds"] = f"{encoding_seconds:.9f}"
            if encoded.returncode != 0 or not gcnf_path.is_file():
                message = (encoded.stderr or encoded.stdout).strip()[-2000:]
                raise RuntimeError(
                    f"GCNF encoding failed (exit={encoded.returncode}): {message}"
                )
            result["gcnf_bytes"] = gcnf_path.stat().st_size

            solver_start = time.perf_counter()
            solved = subprocess.run(
                [str(gismo_binary), str(gcnf_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                cwd=work_directory,
            )
            solver_seconds = time.perf_counter() - solver_start
            result["solver_seconds"] = f"{solver_seconds:.9f}"
            if solved.returncode != 0:
                message = (solved.stderr or solved.stdout).strip()[-2000:]
                raise RuntimeError(
                    f"GiSMo failed (exit={solved.returncode}): {message}"
                )

            groups = parse_sensor_set(solved.stdout, gcnf_path)
            if graph_path.suffix.lower() == ".mtx":
                code = set(groups)
            else:
                mapping = edge_list_group_to_vertex(graph_path)
                code = {mapping[group] for group in groups}

            evaluation_start = time.perf_counter()
            valid, evaluation_message = evaluate_output(graph, code, k)
            evaluation_seconds = time.perf_counter() - evaluation_start

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
                elapsed_seconds=(
                    f"{encoding_seconds + solver_seconds + evaluation_seconds:.9f}"
                ),
                solution_file=str(solution_path),
                detail=(
                    f"GiSMo exit=0; {evaluation_message}; "
                    f"evaluate={evaluation_seconds:.9f}s"
                ),
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
        result["peak_ram_bytes"] = peak_ram_bytes()
        write_json_atomic(result_path, result)
    return 0


def run_isolated(
    script_path: Path,
    graph_path: Path,
    solution_directory: Path,
    encoder_path: Path,
    gismo_binary: Path,
    timeout_seconds: float,
    ram_limit_bytes: int,
    k: int,
) -> dict[str, object]:
    temp_directory = temp_directory_for_k(script_path.parent, k)
    handle = tempfile.NamedTemporaryFile(
        prefix="gismo-result-",
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
        "--encoder-path",
        str(encoder_path),
        "--gismo-binary",
        str(gismo_binary),
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
    observed_peak = 0
    limit_status = ""
    try:
        import psutil
    except ImportError as exc:
        process.kill()
        process.communicate()
        raise RuntimeError("Can psutil de gioi han RAM") from exc

    monitored = psutil.Process(process.pid)
    while process.poll() is None:
        if time.perf_counter() - start > timeout_seconds:
            limit_status = "TIMEOUT"
            break
        try:
            try:
                processes = [monitored, *monitored.children(recursive=True)]
            except (psutil.Error, OSError):
                processes = [monitored]
            current_ram = 0
            for item in processes:
                try:
                    current_ram += item.memory_info().rss
                except (psutil.Error, OSError):
                    pass
            observed_peak = max(observed_peak, current_ram)
        except (psutil.Error, OSError):
            pass
        if observed_peak > ram_limit_bytes:
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
            peak_ram_bytes=observed_peak,
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
                int(result.get("peak_ram_bytes") or 0), observed_peak
            )
            return result
        result = result_template(graph_path)
        killed_by_memory = process.returncode in {
            -signal.SIGKILL,
            -signal.SIGABRT,
            -signal.SIGSEGV,
        }
        result.update(
            status="RAM_LIMITED" if killed_by_memory else "INVALID",
            elapsed_seconds=f"{time.perf_counter() - start:.9f}",
            peak_ram_bytes=observed_peak,
            detail=stderr.strip()[-2000:] or f"worker exit code {process.returncode}",
        )
        return result
    finally:
        result_path.unlink(missing_ok=True)


def completed_datasets(csv_path: Path, retry_invalid: bool = False) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"dataset", "status"} <= set(reader.fieldnames):
            raise ValueError(f"CSV {csv_path} thieu cot dataset/status")
        completed: set[str] = set()
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            status = (row.get("status") or "").strip()
            if dataset and status and not (retry_invalid and status == "INVALID"):
                completed.add(dataset)
        return completed


def append_result(csv_path: Path, result: dict[str, object]) -> None:
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in RESULT_FIELDS})
        output.flush()


def remove_results(csv_path: Path, datasets: set[str]) -> None:
    """Xoa cac dong cu sap retry de CSV chi co mot ket qua moi dataset."""
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
    """Tao ten CSV rieng cho moi k de ket qua khong bi tron/ghi de."""
    suffix = base_path.suffix or ".csv"
    stem = base_path.stem if base_path.suffix else base_path.name
    return base_path.with_name(f"{stem}-k{k}{suffix}")


def main() -> None:
    script_path = Path(__file__).resolve()
    project_directory = script_path.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", nargs="*", type=Path)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--ram-limit-gb", type=float, default=DEFAULT_RAM_LIMIT_GB)
    parser.add_argument("--gismo-binary", type=Path, default=GISMO_BINARY)
    parser.add_argument("--encoder-path", type=Path, default=ENCODE_NETWORK_PATH)
    parser.add_argument("--output-csv", type=Path, default=project_directory / "gismo-results.csv")
    parser.add_argument("--solution-dir", type=Path, default=project_directory / "gismo-solutions")
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="chay lai cac dataset dang co STATUS=INVALID trong CSV",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-ram-bytes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-k", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker is not None:
        if (
            args.result_file is None
            or args.worker_ram_bytes is None
            or args.worker_k is None
        ):
            parser.error("worker thieu tham so")
        set_ram_limit(args.worker_ram_bytes)
        raise SystemExit(
            run_worker(
                args.worker,
                args.result_file,
                args.solution_dir,
                args.encoder_path,
                args.gismo_binary,
                project_directory,
                args.worker_k,
            )
        )
    if args.timeout <= 0 or args.ram_limit_gb <= 0:
        parser.error("timeout va RAM limit phai lon hon 0")
    if not args.encoder_path.is_file():
        parser.error(
            f"Khong tim thay encode_network.py: {args.encoder_path}. "
            "Sua ENCODE_NETWORK_PATH hoac dung --encoder-path."
        )
    if not args.gismo_binary.is_file():
        parser.error(
            f"Khong tim thay GiSMo binary: {args.gismo_binary}. "
            "Sua GISMO_BINARY hoac dung --gismo-binary."
        )

    dataset_directory = project_directory / "standardized_dataset"
    graph_paths = args.graphs or [
        path for path in dataset_directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    graph_paths = sorted(
        graph_paths,
        key=lambda path: (declared_vertex_count(path), path.name.lower()),
    )
    ram_limit_bytes = int(args.ram_limit_gb * 1024**3)
    for k_index, k in enumerate(K, start=1):
        output_csv = result_csv_for_k(args.output_csv, k)
        solution_directory = args.solution_dir / f"k{k}"
        done = completed_datasets(
            output_csv, retry_invalid=args.retry_invalid
        )
        pending = [path for path in graph_paths if path.name not in done]
        if args.retry_invalid:
            remove_results(output_csv, {path.name for path in pending})

        print(
            f"[K {k_index}/{len(K)}] Tim thay {len(graph_paths)} graph; "
            f"da co ket qua={len(graph_paths)-len(pending)}, "
            f"con lai={len(pending)}. k={k}, timeout={args.timeout:g}s, "
            f"RAM limit={args.ram_limit_gb:g} GiB/graph."
        )
        if not pending:
            print(f"Khong co graph nao can chay voi k={k}. Ket qua: {output_csv}")
            continue

        for index, graph_path in enumerate(pending, start=1):
            result = run_isolated(
                script_path,
                graph_path.resolve(),
                solution_directory.resolve(),
                args.encoder_path.resolve(),
                args.gismo_binary.resolve(),
                args.timeout,
                ram_limit_bytes,
                k,
            )
            append_result(output_csv, result)
            print(
                f"[k={k} | {index:02d}/{len(pending):02d}] {graph_path.name}: "
                f"STATUS={result['status']} | time={result['elapsed_seconds']}s | "
                f"code_size={result['code_size'] or 'N/A'}"
            )
            if result["status"] != "VALID":
                detail = str(result.get("detail") or "Khong co thong tin loi")
                print(f"  detail: {detail}")
        print(f"Da ghi tong ket k={k}: {output_csv}")


if __name__ == "__main__":
    main()
