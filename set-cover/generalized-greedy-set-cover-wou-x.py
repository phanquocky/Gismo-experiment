#!/usr/bin/env python3
"""Generalized Greedy Set Cover without Universe, co t0 va 1 <= |F| <= k.

Theo phep quy trong ``set-cover-without-universe.md``, moi trang thai chay
``F`` co tu 1 den ``k`` dinh duoc xem nhu mot hang. Voi hang F, cot x_i cho
biet i co thuoc F hay khong; cot y_i la OR closed-neighborhood cua moi dinh
trong F. Thuat toan chi materialize cac hang va cot sensor; no khong tao
universe gom moi constraint domination/separation.

Phan evaluate dung mo hinh quan sat that: hang cua trang thai nhieu dam chay
la phep OR cac hang cua tung dinh dang chay, dung nhu note trong tai lieu.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from math import comb
from typing import Iterator


Graph = dict[int, set[int]]
FireState = tuple[int, ...]


SAMPLE_GRAPH: Graph = {
    1: {2, 5},
    2: {1, 3, 4},
    3: {2, 4},
    4: {2, 3, 5},
    5: {1, 4},
}


def closed_neighborhoods(graph: Graph) -> dict[int, set[int]]:
    """Tra ve N[v] cho moi dinh va kiem tra cac nhan dinh ke."""
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
    """Liet ke cac dinh cua graph mo rong: moi F voi 1 <= |F| <= k."""
    number_of_vertices = len(vertices)
    if not 1 <= k <= number_of_vertices:
        raise ValueError(f"k phai nam trong [1, {number_of_vertices}], nhan duoc {k}")

    return [
        state
        for number_of_fires in range(1, k + 1)
        for state in combinations(vertices, number_of_fires)
    ]


def expanded_graph_edges(
    graph: Graph, states: list[FireState]
) -> Iterator[tuple[FireState, FireState]]:
    """Sinh cac canh cua graph mo rong ma khong materialize adjacency.

    Graph mo rong giu nguyen canh giua cac singleton cua graph goc. Moi dinh
    phu tro ``F`` co ``|F| > 1`` duoc noi voi moi singleton sensor trong
    ``union(N[v] for v in F)``. Day chinh la hang y(F) tao boi phep OR. Khong
    co canh giua hai dinh phu tro.
    """
    neighborhoods = closed_neighborhoods(graph)
    for u in sorted(graph):
        for v in sorted(graph[u]):
            if u < v:
                yield (u,), (v,)

    for state in states:
        if len(state) > 1:
            observed_sensors: set[int] = set()
            for vertex in state:
                observed_sensors.update(neighborhoods[vertex])
            for sensor in sorted(observed_sensors):
                yield (sensor,), state


def expanded_graph_statistics(graph: Graph, k: int) -> tuple[int, int, int]:
    """Tra ve ``(|V'|, |E'|, TOTAL_CONSTRAIN)`` cua graph mo rong."""
    closed_neighborhoods(graph)
    number_of_vertices = len(graph)
    if not 1 <= k <= number_of_vertices:
        raise ValueError(f"k phai nam trong [1, {number_of_vertices}], nhan duoc {k}")
    expanded_vertices = sum(
        comb(number_of_vertices, size) for size in range(1, k + 1)
    )
    states = fire_states(sorted(graph), k)
    expanded_edges = sum(1 for _ in expanded_graph_edges(graph, states))
    total_constraints = (
        expanded_vertices + expanded_vertices * (expanded_vertices - 1) // 2
    )
    return expanded_vertices, expanded_edges, total_constraints


def build_reduction_columns(
    graph: Graph, k: int
) -> tuple[
    list[FireState],
    dict[int, set[FireState]],
    dict[int, set[FireState]],
]:
    """Dung cac cot ``x_i`` va ``y_i`` cua graph phu tro trong tai lieu.

    Sensor chi la dinh goc ``i``:

    Voi moi hang ``F`` (singleton hoac trang thai to hop):

    * ``x_i(F) = 1`` khi ``i in F``;
    * ``y_i(F) = 1`` khi co ``v in F`` sao cho ``i in N[v]``.

    Tuc la ca hang ``x(F)`` va ``y(F)`` deu la OR cac hang singleton cua
    tung dinh trong F.

    Vi vay cac trang thai to hop chi la hang can phan biet, khong bao gio
    duoc them vao danh sach candidate sensor.
    """
    vertices = sorted(graph)
    closed_neighborhoods(graph)
    states = fire_states(vertices, k)
    x_columns: dict[int, set[FireState]] = {
        sensor: set() for sensor in vertices
    }
    y_columns: dict[int, set[FireState]] = {
        sensor: set() for sensor in vertices
    }

    # x_i(F)=1 neu sensor i la mot trong cac dinh dang chay.
    for state in states:
        for sensor in state:
            x_columns[sensor].add(state)

    # Closed-neighborhood: moi cot y_i chua chinh singleton (i,).
    for sensor in vertices:
        y_columns[sensor].add((sensor,))

    # Voi moi canh (a, b), neu mot dau la singleton sensor thi cot y cua
    # sensor do co gia tri 1 tai hang dau con lai. Day dong thoi xu ly cac
    # canh cu va cac canh moi noi F voi union closed-neighborhood cua F.
    for left, right in expanded_graph_edges(graph, states):
        if len(left) == 1:
            y_columns[left[0]].add(right)
        if len(right) == 1:
            y_columns[right[0]].add(left)

    return states, x_columns, y_columns


def candidate_gains(
    groups: list[set[FireState]],
    vector_0: set[FireState],
    x_columns: dict[int, set[FireState]],
    y_columns: dict[int, set[FireState]],
) -> dict[int, int]:
    """Tinh gain cua cap cot ``(x_i, y_i)`` ma khong tao universe.

    Moi group chua cac hang dang co cung chu ky. Cap cot cua sensor ``i``
    chia group theo ba pattern co the co: ``(1,1)``, ``(0,1)`` va ``(0,0)``.
    So constraint separation moi bang so cap nam trong hai phan khac nhau.
    """
    gains = {sensor: 0 for sensor in y_columns}

    for group in groups:
        group_size = len(group)
        pairs_before = group_size * (group_size - 1) // 2
        for sensor in y_columns:
            x_and_y_one = group & x_columns[sensor]
            y_only = (group & y_columns[sensor]) - x_and_y_one
            zeros = group - y_columns[sensor]
            pairs_after = sum(
                len(part) * (len(part) - 1) // 2
                for part in (x_and_y_one, y_only, zeros)
            )
            gains[sensor] += pairs_before - pairs_after

    # x_i la tap con cua y_i, nen y_i quyet dinh cac constraint domination
    # moi duoc phu.
    for sensor, y_column in y_columns.items():
        gains[sensor] += len(vector_0 & y_column)

    return gains


def split_groups(
    groups: list[set[FireState]],
    x_column: set[FireState],
    y_column: set[FireState],
) -> list[set[FireState]]:
    """Cap nhat cac nhom hang sau khi chon mot sensor."""
    new_groups: list[set[FireState]] = []
    for group in groups:
        x_and_y_one = group & x_column
        y_only = (group & y_column) - x_and_y_one
        zeros = group - y_column
        for part in (x_and_y_one, y_only, zeros):
            if part:
                new_groups.append(part)
    return new_groups


def _remaining_constraints(
    groups: list[set[FireState]], vector_0: set[FireState]
) -> int:
    """Dem constraint domination va separation van chua duoc phu."""
    unseparated_pairs = sum(
        len(group) * (len(group) - 1) // 2 for group in groups
    )
    return len(vector_0) + unseparated_pairs


def generalized_id_greedy_without_universe_with_t0(
    graph: Graph, k: int, *, verbose: bool = False
) -> set[int]:
    """Chay greedy tren cac trang thai co ``1 <= |F| <= k``.

    Candidate sensor luon chi lay tu cac dinh cua graph goc. Khi bang gain,
    dinh co nhan nho hon duoc chon.
    """
    vertices = sorted(graph)
    states, x_columns, y_columns = build_reduction_columns(graph, k)
    groups = [set(states)]
    vector_0 = set(states)
    selected: set[int] = set()
    number_of_states = len(states)
    expected_states, _, total_constraints = expanded_graph_statistics(graph, k)
    if number_of_states != expected_states:
        raise AssertionError(
            f"So dinh graph mo rong khong khop: {number_of_states} != "
            f"{expected_states}"
        )
    step = 0

    while total_constraints > 0:
        gains = candidate_gains(groups, vector_0, x_columns, y_columns)
        available_gains = {
            sensor: gains[sensor]
            for sensor in vertices
            if sensor not in selected
        }
        sensor, gain = max(
            available_gains.items(),
            key=lambda item: (item[1], -item[0]),
            default=(-1, 0),
        )
        if gain == 0:
            raise ValueError("Khong con sensor nao phu duoc constraint moi")

        step += 1
        selected.add(sensor)
        groups = split_groups(groups, x_columns[sensor], y_columns[sensor])
        vector_0.difference_update(y_columns[sensor])
        total_constraints -= gain

        actual_remaining = _remaining_constraints(groups, vector_0)
        if total_constraints != actual_remaining:
            raise AssertionError(
                "Gain khong khop voi so constraint con lai: "
                f"{total_constraints} != {actual_remaining}"
            )

        if verbose:
            score_text = ", ".join(
                f"v_{candidate}={score}"
                for candidate, score in sorted(available_gains.items())
            )
            print(
                f"Buoc {step}: gain {{{score_text}}}; "
                f"chon v_{sensor} (gain={gain}), "
                f"constraint con lai={total_constraints}"
            )

    return selected


# Cac alias ngan gon de thuan tien khi import.
generalized_id_greedy = generalized_id_greedy_without_universe_with_t0
id_greedy = generalized_id_greedy_without_universe_with_t0


def build_or_output_matrix(
    graph: Graph, sensors: set[int], k: int
) -> dict[FireState, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Dung ma tran output that bang OR cac hang singleton.

    Neu ``F=(u,v)`` thi moi bit cua hang F bang bit hang u OR bit hang v,
    cho ca phan ``x`` va phan ``y``. Day cung la ma tran ma greedy su dung.
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
        # OR cua cac cot identity x_i tren nhung hang dang chay.
        x_row = tuple(int(sensor in state) for sensor in ordered_sensors)
        # OR cac hang closed-neighborhood cua moi dinh trong state.
        y_row = tuple(
            int(any(sensor in neighborhoods[vertex] for vertex in state))
            for sensor in ordered_sensors
        )
        matrix[state] = x_row, y_row
    return matrix


def _format_state(state: FireState) -> str:
    return "{" + ",".join(f"v_{vertex}" for vertex in state) + "}"


def evaluate_output(
    graph: Graph, sensors: set[int], k: int
) -> tuple[bool, str]:
    """Kiem tra moi trang thai 1..k dam chay bang chu ky OR ``x | y``."""
    try:
        matrix = build_or_output_matrix(graph, sensors, k)
    except ValueError as exc:
        return False, str(exc)

    seen: dict[tuple[int, ...], FireState] = {}
    for state, (x_row, y_row) in matrix.items():
        if not any(y_row):
            return False, f"Trang thai {_format_state(state)} khong duoc dominate"

        signature = x_row + y_row
        if signature in seen:
            return (
                False,
                f"Hai trang thai {_format_state(seen[signature])} va "
                f"{_format_state(state)} khong duoc phan biet",
            )
        seen[signature] = state

    return (
        True,
        f"Output hop le cho moi trang thai co tu 1 den {k} dinh chay",
    )


def print_output_matrix(graph: Graph, sensors: set[int], k: int) -> None:
    """In ma tran OR ``x | y`` ma evaluate su dung."""
    ordered_sensors = sorted(sensors)
    matrix = build_or_output_matrix(graph, sensors, k)
    x_header = " ".join(f"x_{sensor}" for sensor in ordered_sensors)
    y_header = " ".join(f"y_{sensor}" for sensor in ordered_sensors)
    print(f"              {x_header} | {y_header}")
    for state, (x_row, y_row) in matrix.items():
        x_values = "   ".join(map(str, x_row))
        y_values = "   ".join(map(str, y_row))
        print(f"{_format_state(state):>12} | {x_values} | {y_values}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=2, help="so dam chay toi da")
    parser.add_argument(
        "--quiet", action="store_true", help="khong in gain cua tung buoc"
    )
    args = parser.parse_args()

    print("Sample graph: V={1,2,3,4,5}")
    print("E={(1,2),(1,5),(2,3),(2,4),(3,4),(4,5)}")
    print(f"Generalized identifying code voi k={args.k}")
    expanded_vertices, expanded_edges, total_constraints = (
        expanded_graph_statistics(SAMPLE_GRAPH, args.k)
    )
    print(
        f"Graph mo rong: |V'|={expanded_vertices}, |E'|={expanded_edges}, "
        f"TOTAL_CONSTRAIN={total_constraints}"
    )
    sensors = generalized_id_greedy_without_universe_with_t0(
        SAMPLE_GRAPH, args.k, verbose=not args.quiet
    )
    valid, message = evaluate_output(SAMPLE_GRAPH, sensors, args.k)

    print(f"Sensors duoc chon: {sorted(sensors)}")
    print_output_matrix(SAMPLE_GRAPH, sensors, args.k)
    print(f"Evaluate output: {'PASS' if valid else 'FAIL'} - {message}")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
