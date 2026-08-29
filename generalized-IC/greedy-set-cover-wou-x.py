#!/usr/bin/env python3
"""Greedy Set Cover without Universe, co them cot t0 (x_i).

Moi sensor ``i`` cung cap hai cot quan sat:

* ``x_i`` o thoi diem t0: chi dinh ``i`` co gia tri 1;
* ``y_i`` o thoi diem t1: cac dinh trong closed-neighborhood cua ``i`` co
  gia tri 1.

Thuat toan khong materialize cac constraint domination va separation. Thay
vao do, no duy tri cac nhom dinh dang co cung chu ky tren cac sensor da chon.
"""

from __future__ import annotations

import argparse


Graph = dict[int, set[int]]


SAMPLE_GRAPH: Graph = {
    1: {2, 5},
    2: {1, 3, 4},
    3: {2, 4},
    4: {2, 3, 5},
    5: {1, 4},
}


def closed_neighborhoods(graph: Graph) -> dict[int, set[int]]:
    """Tra ve N[v] cho moi dinh v."""
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


def candidate_gains_with_t0(
    groups: list[set[int]],
    vector_0: set[int],
    neighborhoods: dict[int, set[int]],
) -> dict[int, int]:
    """Tinh gain khi them dong thoi ``x_i`` va ``y_i`` cho moi candidate.

    ``groups`` la cac nhom hang chua duoc phan biet. Cot ``y_i`` chia mot
    nhom thanh ``ones`` va ``zeros``, nen phan biet them
    ``len(ones) * len(zeros)`` cap. Sau do ``x_i`` tach rieng hang ``i``
    khoi nhom ``ones``, nen phan biet them ``len(ones) - 1`` cap neu ``i``
    nam trong nhom dang xet.

    ``vector_0`` chua cac dinh chua duoc dominate. Vi ``x_i`` la tap con
    cua ``y_i``, chi cot ``y_i`` co the them constraint domination moi.
    """
    gains = {candidate: 0 for candidate in neighborhoods}

    for group in groups:
        # number_of_ones[i] = so hang trong group co y_i = 1.
        number_of_ones: dict[int, int] = {}
        for vertex in group:
            for candidate in neighborhoods[vertex]:
                number_of_ones[candidate] = number_of_ones.get(candidate, 0) + 1

        group_size = len(group)
        for candidate, ones in number_of_ones.items():
            # Cac cap duoc tach boi cot y_i.
            gains[candidate] += ones * (group_size - ones)

        # y_i luon bang 1 tai hang i. Cot x_i tiep tuc tach hang i khoi
        # nhung hang van co y_i = 1 trong cung group.
        for candidate in group:
            gains[candidate] += number_of_ones[candidate] - 1

    # Them cac constraint domination duoc y_i phu lan dau.
    for vertex in vector_0:
        for candidate in neighborhoods[vertex]:
            gains[candidate] += 1

    return gains


def split_groups_with_t0(
    groups: list[set[int]],
    sensor: int,
    sensor_y_column: set[int],
) -> list[set[int]]:
    """Chia cac nhom theo ba pattern co the co: (1,1), (0,1), (0,0)."""
    new_groups: list[set[int]] = []
    for group in groups:
        y_ones = group & sensor_y_column
        x_and_y_one = {sensor} if sensor in group else set()
        y_only = y_ones - x_and_y_one
        zeros = group - y_ones

        if x_and_y_one:
            new_groups.append(x_and_y_one)
        if y_only:
            new_groups.append(y_only)
        if zeros:
            new_groups.append(zeros)
    return new_groups


def _remaining_constraints(groups: list[set[int]], vector_0: set[int]) -> int:
    """Dem constraint chua phu ma khong liet ke tung cap dinh."""
    unseparated_pairs = sum(
        len(group) * (len(group) - 1) // 2 for group in groups
    )
    return len(vector_0) + unseparated_pairs


def id_greedy_without_universe_with_t0(
    graph: Graph, *, verbose: bool = False
) -> set[int]:
    """Chon sensor bang greedy gain tren hai cot ``x_i`` va ``y_i``.

    Khi bang diem, dinh co nhan nho hon duoc chon de ket qua co tinh xac
    dinh. Ham tra ve tap sensor da chon.
    """
    vertices = sorted(graph)
    neighborhoods = closed_neighborhoods(graph)

    # Ban dau tat ca cac hang co chu ky rong giong nhau va chua hang nao
    # duoc dominate.
    groups = [set(vertices)]
    vector_0 = set(vertices)
    selected: set[int] = set()
    total_constraints = len(vertices) + len(vertices) * (len(vertices) - 1) // 2
    step = 0

    while total_constraints > 0:
        gains = candidate_gains_with_t0(groups, vector_0, neighborhoods)
        available_gains = {
            vertex: gains[vertex]
            for vertex in vertices
            if vertex not in selected
        }
        candidate, gain = max(
            available_gains.items(),
            key=lambda item: (item[1], -item[0]),
            default=(-1, 0),
        )
        if gain == 0:
            raise ValueError("Khong con candidate nao phu duoc constraint moi")

        step += 1
        selected.add(candidate)
        sensor_y_column = {
            vertex
            for vertex in vertices
            if candidate in neighborhoods[vertex]
        }
        groups = split_groups_with_t0(groups, candidate, sensor_y_column)
        vector_0.difference_update(sensor_y_column)
        total_constraints -= gain

        # Day la invariant cua cach tinh gain; no cung giup phat hien sai
        # lech giua viec cham diem va viec cap nhat cac nhom.
        actual_remaining = _remaining_constraints(groups, vector_0)
        if total_constraints != actual_remaining:
            raise AssertionError(
                "Gain khong khop voi so constraint con lai: "
                f"{total_constraints} != {actual_remaining}"
            )

        if verbose:
            score_text = ", ".join(
                f"v_{vertex}={score}"
                for vertex, score in sorted(available_gains.items())
            )
            print(
                f"Buoc {step}: gain {{{score_text}}}; "
                f"chon v_{candidate} (gain={gain}), "
                f"constraint con lai={total_constraints}"
            )

    return selected


# Alias ngan gon de thuan tien khi import file nhu mot module thuat toan.
id_greedy = id_greedy_without_universe_with_t0


def build_output_matrix(
    graph: Graph, sensors: set[int]
) -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Dung lai cac hang ``x | y`` chi tren nhung sensor duoc chon."""
    neighborhoods = closed_neighborhoods(graph)
    unknown_sensors = sensors - graph.keys()
    if unknown_sensors:
        raise ValueError(f"Sensor khong thuoc graph: {sorted(unknown_sensors)}")

    ordered_sensors = sorted(sensors)
    matrix: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for vertex in sorted(graph):
        x_row = tuple(int(vertex == sensor) for sensor in ordered_sensors)
        y_row = tuple(
            int(sensor in neighborhoods[vertex]) for sensor in ordered_sensors
        )
        matrix[vertex] = x_row, y_row
    return matrix


def evaluate_output(graph: Graph, sensors: set[int]) -> tuple[bool, str]:
    """Kiem tra output bang ma tran ``x_i | y_i`` theo dung mo ta.

    Moi hang phai duoc dominate (co it nhat mot bit 1 o phan ``y``) va hai
    hang bat ky phai co chu ky ``x | y`` khac nhau.
    """
    try:
        matrix = build_output_matrix(graph, sensors)
    except ValueError as exc:
        return False, str(exc)

    seen: dict[tuple[int, ...], int] = {}
    for vertex, (x_row, y_row) in matrix.items():
        if not any(y_row):
            return False, f"Dinh {vertex} khong duoc dominate"

        signature = x_row + y_row
        if signature in seen:
            return (
                False,
                f"Hai dinh {seen[signature]} va {vertex} khong duoc phan biet",
            )
        seen[signature] = vertex

    return True, "Output hop le: moi hang duoc phu va doi mot phan biet"


def print_output_matrix(graph: Graph, sensors: set[int]) -> None:
    """In ma tran ``x | y`` rut gon cua output de de kiem tra bang mat."""
    ordered_sensors = sorted(sensors)
    matrix = build_output_matrix(graph, sensors)
    x_header = " ".join(f"x_{sensor}" for sensor in ordered_sensors)
    y_header = " ".join(f"y_{sensor}" for sensor in ordered_sensors)
    print(f"      {x_header} | {y_header}")
    for vertex, (x_row, y_row) in matrix.items():
        x_values = "   ".join(map(str, x_row))
        y_values = "   ".join(map(str, y_row))
        print(f"v_{vertex} | {x_values} | {y_values}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet", action="store_true", help="khong in gain cua tung buoc"
    )
    args = parser.parse_args()

    print("Sample graph: V={1,2,3,4,5}")
    print("E={(1,2),(1,5),(2,3),(2,4),(3,4),(4,5)}")
    sensors = id_greedy_without_universe_with_t0(
        SAMPLE_GRAPH, verbose=not args.quiet
    )
    valid, message = evaluate_output(SAMPLE_GRAPH, sensors)

    print(f"Sensors duoc chon: {sorted(sensors)}")
    print_output_matrix(SAMPLE_GRAPH, sensors)
    print(f"Evaluate output: {'PASS' if valid else 'FAIL'} - {message}")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
