#!/usr/bin/env python3
"""Check uniqueness of ORs of two closed-neighborhood rows.

A graph passes when, for every two distinct unordered vertex pairs
``{i, j} != {k, l}``, the following signatures are distinct::

    N[i] union N[j] != N[k] union N[l]

The closed-neighborhood rows are represented as Python integer bitsets, so a
dense adjacency matrix does not need to be materialized.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_standardize import Graph, SUPPORTED_SUFFIXES, read_graph  # noqa: E402


Pair = tuple[int, int]


@dataclass(frozen=True)
class CheckResult:
    passes: bool
    vertices: int
    pairs_checked: int
    first_pair: Pair | None = None
    second_pair: Pair | None = None


def closed_neighborhood_rows(graph: Graph) -> tuple[list[int], list[int]]:
    """Return sorted vertex labels and their closed-neighborhood bitsets."""
    vertices = sorted(graph)
    position = {vertex: index for index, vertex in enumerate(vertices)}
    rows: list[int] = []
    for vertex in vertices:
        row = 1 << position[vertex]
        for neighbor in graph[vertex]:
            row |= 1 << position[neighbor]
        rows.append(row)
    return vertices, rows


def check_graph(graph: Graph) -> CheckResult:
    """Return the first pair collision, or a successful exact result."""
    vertices, rows = closed_neighborhood_rows(graph)
    signature_to_pair: dict[int, Pair] = {}
    pairs_checked = 0

    for left, right in combinations(range(len(vertices)), 2):
        pairs_checked += 1
        signature = rows[left] | rows[right]
        pair = (vertices[left], vertices[right])
        previous = signature_to_pair.get(signature)
        if previous is not None:
            return CheckResult(
                False, len(vertices), pairs_checked, previous, pair
            )
        signature_to_pair[signature] = pair

    return CheckResult(True, len(vertices), pairs_checked)


def graph_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file()
        and candidate.suffix.lower() in SUPPORTED_SUFFIXES
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether all ORs of two closed-neighborhood rows are unique."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=REPOSITORY_ROOT / "standardized_dataset",
        help="a graph file or dataset directory (default: standardized_dataset)",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=5_000,
        help="skip larger graphs (default: 5000; use 0 for no limit)",
    )
    args = parser.parse_args()

    files = graph_files(args.path)
    if not files:
        parser.error(f"no supported graph files found at {args.path}")

    passing = 0
    skipped = 0
    for path in files:
        graph = read_graph(path)
        if args.max_vertices and len(graph) > args.max_vertices:
            skipped += 1
            print(
                f"SKIP\t{path.name}\tn={len(graph)}"
                f"\tlimit={args.max_vertices}"
            )
            continue
        result = check_graph(graph)
        if result.passes:
            passing += 1
            detail = "all pair signatures are distinct"
            status = "PASS"
        else:
            detail = (
                f"collision {result.first_pair} = {result.second_pair}"
            )
            status = "FAIL"
        print(
            f"{status}\t{path.name}\tn={result.vertices}"
            f"\tchecked={result.pairs_checked}\t{detail}"
        )

    failed = len(files) - passing - skipped
    print(
        f"SUMMARY\tpass={passing}\tfail={failed}"
        f"\tskip={skipped}\ttotal={len(files)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
