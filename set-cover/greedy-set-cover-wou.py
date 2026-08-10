#!/usr/bin/env python3
"""ID-Greedy khong khoi tao universe va cac distinguishing set.

File graph la edge-list cua graph vo huong. Moi dong chua hai nhan dinh;
dong chi co mot nhan co the duoc dung de khai bao mot dinh co lap.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


Graph = dict[int, set[int]]


def _read_edge_list(lines: list[str]) -> Graph:
    """Doc cac dong edge-list."""
    graph: Graph = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
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


def main() -> None:
    default_graph = '/Users/admin/Documents/master/AnhThach/experiment/datasets/socfb-Amherst41.mtx'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graph", nargs="?", type=Path, default=default_graph, help="file edge-list"
    )
    args = parser.parse_args()

    try:
        graph = read_graph(args.graph)
        start = time.perf_counter()
        code = id_greedy_without_universe(graph)
        elapsed = time.perf_counter() - start
        valid, message = is_identifying_code(graph, code)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Loi: {exc}\n")

    print(f"C_greedy_wou: {sorted(code)}")
    print(f"Size cua output: {len(code)}")
    print(f"Thoi gian chay thuat toan: {elapsed:.9f} giay")
    print(f"Kiem tra Identifying Code: {'DAT' if valid else 'KHONG DAT'} - {message}")


if __name__ == "__main__":
    main()
