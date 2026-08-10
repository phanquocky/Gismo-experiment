#!/usr/bin/env python3
"""Remove closed twins from every graph in ``datasets/``.

Only the largest connected component of each input graph is retained. Then, for
every closed-twin equivalence class, the smallest-labelled vertex is kept. The
process is repeated because removing vertices can create new closed twins.
Output vertices are relabelled consecutively from 1 and written to
``standardized_dataset/`` using the original file names.
"""

from __future__ import annotations

import argparse
from itertools import chain
from pathlib import Path
from typing import Iterator, TextIO


Graph = dict[int, set[int]]
SUPPORTED_SUFFIXES = {".txt", ".mtx", ".edges"}


def _integer_triple(text: str) -> tuple[int, int, int] | None:
    fields = text.split()
    if len(fields) != 3:
        return None
    try:
        return int(fields[0]), int(fields[1]), int(fields[2])
    except ValueError:
        return None


def _matrix_market_entries(
    graph_file: TextIO,
) -> tuple[tuple[int, int, int], str | None, Iterator[str]]:
    """Find dimensions, including datasets that put them in a comment."""
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
        raise ValueError("cannot find valid Matrix Market dimensions")
    raise ValueError("Matrix Market file contains no data")


def read_matrix_market(path: Path) -> Graph:
    with path.open("r", encoding="utf-8") as graph_file:
        header = graph_file.readline().strip().lstrip("%").lower()
        if not header.startswith("matrixmarket matrix coordinate"):
            raise ValueError("invalid Matrix Market header")
        if "symmetric" not in header:
            raise ValueError("only symmetric Matrix Market graphs are supported")

        dimensions, first_entry, remaining = _matrix_market_entries(graph_file)
        rows, columns, declared_entries = dimensions
        if rows != columns:
            raise ValueError("graph adjacency matrix must be square")

        graph = {vertex: set() for vertex in range(1, rows + 1)}
        entries = 0
        for raw_line in chain([first_entry] if first_entry else [], remaining):
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"invalid Matrix Market entry: {line!r}")
            u, v = int(fields[0]), int(fields[1])
            if not (1 <= u <= rows and 1 <= v <= rows):
                raise ValueError(f"vertex outside declared dimensions: {u} {v}")
            entries += 1
            if u != v:
                graph[u].add(v)
                graph[v].add(u)
        if entries != declared_entries:
            raise ValueError(
                f"declared {declared_entries} entries but read {entries}"
            )
        return graph


def read_edge_list(path: Path) -> Graph:
    graph: Graph = {}
    declared_vertices: int | None = None
    maximum_vertex = 0

    with path.open("r", encoding="utf-8") as graph_file:
        for line_number, raw_line in enumerate(graph_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("%", "#")):
                metadata = _integer_triple(line.lstrip("%#").strip())
                if metadata is not None and metadata[1] == metadata[2]:
                    declared_vertices = metadata[1]
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"line {line_number}: expected two vertices")
            u, v = int(fields[0]), int(fields[1])
            if u <= 0 or v <= 0:
                raise ValueError(f"line {line_number}: vertices must be positive")
            maximum_vertex = max(maximum_vertex, u, v)
            graph.setdefault(u, set())
            graph.setdefault(v, set())
            if u != v:
                graph[u].add(v)
                graph[v].add(u)

    number_of_vertices = declared_vertices or maximum_vertex
    if number_of_vertices == 0:
        raise ValueError("empty graph")
    if maximum_vertex > number_of_vertices:
        raise ValueError("vertex exceeds the declared graph size")
    for vertex in range(1, number_of_vertices + 1):
        graph.setdefault(vertex, set())
    return graph


def read_graph(path: Path) -> Graph:
    return read_matrix_market(path) if path.suffix.lower() == ".mtx" else read_edge_list(path)


def remove_closed_twins(graph: Graph) -> tuple[Graph, int, int]:
    """Keep one vertex per closed-neighborhood class until the graph is twin-free."""
    graph = {vertex: set(neighbors) for vertex, neighbors in graph.items()}
    total_removed = 0
    rounds = 0

    while True:
        representative_by_signature: dict[frozenset[int], int] = {}
        to_remove: set[int] = set()
        for vertex in sorted(graph):
            signature = frozenset(graph[vertex] | {vertex})
            if signature in representative_by_signature:
                to_remove.add(vertex)
            else:
                representative_by_signature[signature] = vertex

        if not to_remove:
            return graph, total_removed, rounds

        rounds += 1
        total_removed += len(to_remove)
        for vertex in to_remove:
            for neighbor in graph[vertex]:
                graph[neighbor].discard(vertex)
            del graph[vertex]


def keep_largest_connected_component(graph: Graph) -> tuple[Graph, int, int]:
    """Return the largest component, number of components, and removed vertices.

    Ties are resolved deterministically by keeping the component whose smallest
    vertex label is smallest.
    """
    unseen = set(graph)
    components: list[set[int]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        component = {root}
        stack = [root]
        while stack:
            vertex = stack.pop()
            new_vertices = graph[vertex] & unseen
            unseen.difference_update(new_vertices)
            component.update(new_vertices)
            stack.extend(new_vertices)
        components.append(component)

    largest = min(components, key=lambda component: (-len(component), min(component)))
    reduced_graph = {
        vertex: graph[vertex] & largest
        for vertex in largest
    }
    return reduced_graph, len(components), len(graph) - len(largest)


def relabel_graph(graph: Graph) -> Graph:
    labels = {old: new for new, old in enumerate(sorted(graph), start=1)}
    return {
        labels[vertex]: {labels[neighbor] for neighbor in neighbors}
        for vertex, neighbors in graph.items()
    }


def graph_edges(graph: Graph) -> Iterator[tuple[int, int]]:
    for u in sorted(graph):
        for v in sorted(graph[u]):
            if u < v:
                yield u, v


def write_graph(path: Path, graph: Graph) -> None:
    number_of_edges = sum(map(len, graph.values())) // 2
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        if path.suffix.lower() == ".mtx":
            output.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
            output.write(f"{len(graph)} {len(graph)} {number_of_edges}\n")
        else:
            # Metadata preserves isolated vertices in edge-list based datasets.
            output.write(f"% {number_of_edges} {len(graph)} {len(graph)}\n")
        for u, v in graph_edges(graph):
            output.write(f"{u} {v}\n")
    temporary_path.replace(path)


def has_closed_twins(graph: Graph) -> bool:
    signatures: set[frozenset[int]] = set()
    for vertex, neighbors in graph.items():
        signature = frozenset(neighbors | {vertex})
        if signature in signatures:
            return True
        signatures.add(signature)
    return False


def is_connected(graph: Graph) -> bool:
    if not graph:
        return False
    root = next(iter(graph))
    visited = {root}
    stack = [root]
    while stack:
        vertex = stack.pop()
        new_vertices = graph[vertex] - visited
        visited.update(new_vertices)
        stack.extend(new_vertices)
    return len(visited) == len(graph)


def standardize_file(
    source: Path, destination: Path
) -> tuple[int, int, int, int, int]:
    graph = read_graph(source)
    original_vertices = len(graph)
    graph, component_count, component_removed = keep_largest_connected_component(graph)
    graph, twin_removed, rounds = remove_closed_twins(graph)
    graph = relabel_graph(graph)
    if has_closed_twins(graph):
        raise RuntimeError("internal error: closed twins remain after standardization")
    if not is_connected(graph):
        raise RuntimeError("internal error: standardized graph is not connected")
    write_graph(destination, graph)
    return (
        original_vertices,
        twin_removed,
        rounds,
        component_count,
        component_removed,
    )


def main() -> int:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=script_directory / "datasets")
    parser.add_argument(
        "--output", type=Path, default=script_directory / "standardized_dataset"
    )
    args = parser.parse_args()

    sources = sorted(
        (
            path
            for path in args.input.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )
    if not sources:
        parser.error(f"no supported graph files found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(sources)} graph(s) in {args.input}")
    for index, source in enumerate(sources, start=1):
        destination = args.output / source.name
        original, twin_removed, rounds, components, component_removed = standardize_file(
            source, destination
        )
        remaining = original - twin_removed - component_removed
        print(
            f"[{index:02d}/{len(sources):02d}] {source.name}: "
            f"{original} -> {remaining} vertices; "
            f"closed-twin removed={twin_removed} ({rounds} round(s)); "
            f"components={components}, outside-largest removed={component_removed}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
