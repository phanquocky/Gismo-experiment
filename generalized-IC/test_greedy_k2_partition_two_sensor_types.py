"""Tests against an independent explicit K=2 Set Cover oracle."""

from __future__ import annotations

import csv
import importlib.util
import itertools
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().with_name("greedy_k2_partition_two_sensor_types.py")
spec = importlib.util.spec_from_file_location("two_sensor_types_test_target", SCRIPT)
assert spec is not None and spec.loader is not None
algorithm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(algorithm)


def explicit_oracle(adj):
    vertices = sorted(adj)
    states = [(vertex,) for vertex in vertices] + list(
        itertools.combinations(vertices, 2)
    )
    candidates = algorithm.build_candidates(adj)
    responses = {
        sensor: [bool(set(state) & detection) for state in states]
        for sensor, detection in candidates.items()
    }
    uncovered = {("dom", index) for index in range(len(states))}
    uncovered.update(
        ("sep", left, right) for left in range(len(states)) for right in range(left)
    )
    universe = set(uncovered)
    coverage = {}
    for sensor, bits in responses.items():
        coverage[sensor] = {("dom", index) for index, bit in enumerate(bits) if bit} | {
            ("sep", left, right)
            for left in range(len(states))
            for right in range(left)
            if bits[left] != bits[right]
        }

    bundled_coverage = {
        vertex: coverage[("N", vertex)] | coverage[("L", vertex)] for vertex in vertices
    }
    selected = []
    trace = []
    while uncovered:
        gains = {
            vertex: len(constraints & uncovered)
            for vertex, constraints in bundled_coverage.items()
            if vertex not in selected
        }
        vertex = max(gains, key=gains.get)
        gain = gains[vertex]
        if gain <= 0:
            raise AssertionError(
                "Explicit family should be feasible with all bundled sensors"
            )
        before = len(uncovered)
        uncovered.difference_update(bundled_coverage[vertex])
        selected.append(vertex)
        trace.append((vertex, gain, before, len(uncovered)))

    pruned = list(selected)
    for vertex in reversed(selected):
        trial = [kept_vertex for kept_vertex in pruned if kept_vertex != vertex]
        covered = set().union(*(bundled_coverage[item] for item in trial))
        if universe <= covered:
            pruned = trial
    return selected, pruned, trace


class AlgorithmTests(unittest.TestCase):
    def test_documented_example(self):
        graph = {
            1: {2, 5},
            2: {1, 3, 4},
            3: {2, 4},
            4: {2, 3, 5},
            5: {1, 4},
        }
        result = algorithm.solve_two_sensor_types(graph, trace=True)
        self.assertEqual(
            result["selected"],
            [
                ("N", 1),
                ("L", 1),
                ("N", 3),
                ("L", 3),
                ("N", 5),
                ("L", 5),
                ("N", 2),
                ("L", 2),
            ],
        )
        self.assertEqual(result["selected_vertices"], [1, 3, 5, 2])
        self.assertEqual(result["greedy_selected_vertices"], [1, 3, 5, 2])
        self.assertEqual(
            result["trace"],
            [
                (1, 83, 120, 37),
                (3, 27, 37, 10),
                (5, 8, 10, 2),
                (2, 2, 2, 0),
            ],
        )

    def test_all_graphs_up_to_five_vertices(self):
        for n in range(6):
            vertices = list(range(n))
            edges = list(itertools.combinations(vertices, 2))
            for mask in range(1 << len(edges)):
                graph = {vertex: set() for vertex in vertices}
                for index, (left, right) in enumerate(edges):
                    if mask >> index & 1:
                        graph[left].add(right)
                        graph[right].add(left)
                result = algorithm.solve_two_sensor_types(graph, trace=True)
                greedy_vertices, selected_vertices, trace = explicit_oracle(graph)
                expected_sensors = [
                    (kind, vertex)
                    for vertex in selected_vertices
                    for kind in ("N", "L")
                ]
                self.assertEqual(
                    result["greedy_selected_vertices"], greedy_vertices, graph
                )
                self.assertEqual(result["selected_vertices"], selected_vertices, graph)
                self.assertEqual(result["selected"], expected_sensors, graph)
                self.assertEqual(result["trace"], trace, graph)
                self.assertTrue(algorithm.validate_solution(graph, expected_sensors)[0])
                self.assertEqual(result["code_size"], len(selected_vertices))
                self.assertEqual(
                    result["total_selected_types"], 2 * result["code_size"]
                )
                self.assertEqual(result["n_sensor_count"], result["code_size"])
                self.assertEqual(result["l_sensor_count"], result["code_size"])
                self.assertEqual(result["dual_type_vertex_count"], result["code_size"])
                self.assertEqual(
                    result["pre_prune_code_size"], len(greedy_vertices), graph
                )
                self.assertLessEqual(result["code_size"], result["pre_prune_code_size"])

    def test_reverse_delete_removes_redundant_bundle(self):
        graph = {0: {1, 2}, 1: {0}, 2: {0}, 3: set(), 4: set()}
        result = algorithm.solve_two_sensor_types(graph, trace=True)
        self.assertEqual(result["greedy_selected_vertices"], [0, 3, 4, 1, 2])
        self.assertEqual(result["selected_vertices"], [3, 4, 1, 2])
        self.assertEqual(result["removed_vertices"], [0])
        self.assertTrue(algorithm.validate_solution(graph, result["selected"])[0])

    def test_two_types_at_one_vertex_count_once(self):
        graph = {0: {1}, 1: {0}, 2: set(), 3: set()}
        result = algorithm.solve_two_sensor_types(graph, trace=True)
        self.assertEqual(
            result["selected"],
            [
                ("N", 0),
                ("L", 0),
                ("N", 2),
                ("L", 2),
                ("N", 3),
                ("L", 3),
                ("N", 1),
                ("L", 1),
            ],
        )
        self.assertEqual(result["selected_vertices"], [0, 2, 3, 1])
        self.assertEqual(result["total_selected_types"], 8)
        self.assertEqual(result["dual_type_vertex_count"], 4)
        self.assertEqual(result["code_size"], 4)

    def test_degenerate_and_invalid_inputs(self):
        self.assertEqual(algorithm.solve_two_sensor_types({})["selected"], [])
        complete = {v: set(range(4)) - {v} for v in range(4)}
        result = algorithm.solve_two_sensor_types(complete)
        self.assertTrue(algorithm.validate_solution(complete, result["selected"])[0])
        with self.assertRaises(ValueError):
            algorithm.closed_neighborhoods({1: {2}, 2: set()})
        valid, _, _, _ = algorithm.validate_solution({1: set()}, [("X", 1)])
        self.assertFalse(valid)


@unittest.skipUnless(importlib.util.find_spec("psutil"), "psutil required")
class BatchTests(unittest.TestCase):
    def test_worker_csv_solutions_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="two-types-unit-") as directory:
            root = Path(directory)
            graph = root / "edge-and-isolates.txt"
            complete = root / "complete3.txt"
            graph.write_text("% 1 4 4\n1 2\n", encoding="utf-8")
            complete.write_text("% 3 3 3\n1 2\n1 3\n2 3\n", encoding="utf-8")
            output = root / "results.csv"
            solutions = root / "solutions"
            command = [
                sys.executable,
                str(SCRIPT),
                str(graph),
                str(complete),
                "--timeout",
                "30",
                "--ram-limit-gb",
                "1",
                "--output-csv",
                str(output),
                "--solution-dir",
                str(solutions),
            ]
            subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=45
            )
            with output.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["status"] for row in rows}, {"VALID"})
            for row in rows:
                self.assertEqual(row["K"], "2")
                self.assertEqual(
                    int(row["candidate_type_count"]),
                    2 * int(row["candidate_bundle_count"]),
                )
                self.assertEqual(row["validation_passed"], "True")
                self.assertEqual(row["remaining_constraints"], "0")
                self.assertLessEqual(
                    int(row["code_size"]), int(row["pre_prune_code_size"])
                )
                self.assertEqual(
                    int(row["total_selected_types"]), 2 * int(row["code_size"])
                )
                self.assertEqual(row["dual_type_vertex_count"], row["code_size"])
                sensors = Path(row["solution_file"]).read_text().splitlines()
                vertices = Path(row["vertex_solution_file"]).read_text().splitlines()
                self.assertEqual(len(sensors), int(row["total_selected_types"]))
                self.assertEqual(len(vertices), int(row["code_size"]))
            overlap = next(row for row in rows if row["dataset"] == graph.name)
            self.assertEqual(overlap["total_selected_types"], "8")
            self.assertEqual(overlap["dual_type_vertex_count"], "4")
            self.assertEqual(overlap["code_size"], "4")

            before = output.read_bytes()
            resumed = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=45
            )
            self.assertIn("con lai=0", resumed.stdout)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
