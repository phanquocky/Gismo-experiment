#!/usr/bin/env python3
"""Generalized Greedy Set Cover without Universe, co t0 va 1 <= |F| <= k.

Moi sensor ``i`` van tao hai cot lien tiep nhu mo hinh ban dau:

* ``x_i`` tai ``t=0``: bang 1 khi ``i`` nam trong fire set;
* ``y_i`` tai ``t1=t0+1``: bang 1 khi fire set giao voi ``N[i]``.

Khac voi ban cu, thuat toan khong materialize cac fire set. No chi giu
partition bitset cua cac dinh don theo signature ``x | y`` hien tai. So fire
set trong tung OR-signature va exact gain duoc khoi phuc bang cong thuc truc
tiep cho ``k=2`` hoac sparse DP cho ``k>2``, theo ``docs.md``.
"""

from __future__ import annotations

import argparse
import heapq
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


def _validate_k(number_of_vertices: int, k: int) -> None:
    if not 1 <= k <= number_of_vertices:
        raise ValueError(f"k phai nam trong [1, {number_of_vertices}], nhan duoc {k}")


def _observation_masks(
    graph: Graph, vertices: list[int]
) -> tuple[list[int], dict[int, int]]:
    """Ma hoa cot ``y_i`` thanh bitset cac dinh duoc sensor ``i`` thay.

    Dung inverse neighborhood thay vi ngam dinh graph doi xung. Nhu vay bit
    ``u`` cua mask sensor ``i`` dung khi va chi khi ``i in N[u]``.
    """
    neighborhoods = closed_neighborhoods(graph)
    vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
    masks = [0] * len(vertices)
    for observed_vertex, neighborhood in neighborhoods.items():
        observed_bit = 1 << vertex_index[observed_vertex]
        for sensor in neighborhood:
            masks[vertex_index[sensor]] |= observed_bit
    return masks, vertex_index


def _partition_items(
    partition: dict[int, int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Tra ve signature, member-bitset va size theo cung mot thu tu."""
    items = sorted(partition.items())
    signatures = tuple(signature for signature, _ in items)
    member_masks = tuple(members for _, members in items)
    sizes = tuple(members.bit_count() for members in member_masks)
    return signatures, member_masks, sizes


def _count_k2(signatures: tuple[int, ...], counts: tuple[int, ...]) -> dict[int, int]:
    """Dem OR-signature cua moi fire set co mot hoac hai dinh."""
    result: dict[int, int] = {}
    for index, (signature, count) in enumerate(zip(signatures, counts)):
        if count <= 0:
            continue
        same_class = count + count * (count - 1) // 2
        result[signature] = result.get(signature, 0) + same_class
        for other_index in range(index):
            other_count = counts[other_index]
            if other_count:
                joined = signature | signatures[other_index]
                result[joined] = result.get(joined, 0) + count * other_count
    return result


def _count_layers_dp(
    signatures: tuple[int, ...], counts: tuple[int, ...], k: int
) -> list[dict[int, int]]:
    """Sparse DP: layer ``s`` dem cach chon dung ``s`` dinh theo OR."""
    layers: list[dict[int, int]] = [{0: 1}] + [{} for _ in range(k)]
    processed_vertices = 0
    for signature, count in zip(signatures, counts):
        if count <= 0:
            continue
        new_layers = [dict(layer) for layer in layers]
        max_chosen = min(count, k)
        choose = [comb(count, amount) for amount in range(max_chosen + 1)]
        max_old_size = min(k, processed_vertices)
        for old_size in range(max_old_size + 1):
            if not layers[old_size]:
                continue
            max_added = min(max_chosen, k - old_size)
            for amount in range(1, max_added + 1):
                destination = new_layers[old_size + amount]
                multiplier = choose[amount]
                for old_signature, ways in layers[old_size].items():
                    joined = old_signature | signature
                    destination[joined] = destination.get(joined, 0) + ways * multiplier
        layers = new_layers
        processed_vertices += count
    return layers


def _count_signatures(
    signatures: tuple[int, ...], counts: tuple[int, ...], k: int
) -> dict[int, int]:
    """Dem fire set ``1 <= |F| <= k`` ma khong liet ke tung ``F``."""
    if k == 1:
        return {
            signature: count for signature, count in zip(signatures, counts) if count
        }
    if k == 2:
        return _count_k2(signatures, counts)

    result: dict[int, int] = {}
    for layer in _count_layers_dp(signatures, counts, k)[1:]:
        for signature, ways in layer.items():
            result[signature] = result.get(signature, 0) + ways
    return result


def _counts_containing_one_vertex(
    signatures: tuple[int, ...],
    sizes: tuple[int, ...],
    class_index: int,
    k: int,
) -> dict[int, int]:
    """Dem fire set chua mot candidate co dinh trong ``class_index``.

    Tat ca candidate trong cung singleton-signature class co cung ket qua,
    nen bang nay chi can tinh mot lan cho moi class trong mot greedy round.
    """
    candidate_signature = signatures[class_index]
    remaining = list(sizes)
    remaining[class_index] -= 1

    if k == 1:
        return {candidate_signature: 1}
    if k == 2:
        result = {candidate_signature: 1}
        for signature, count in zip(signatures, remaining):
            if count:
                joined = candidate_signature | signature
                result[joined] = result.get(joined, 0) + count
        return result

    result: dict[int, int] = {}
    layers = _count_layers_dp(signatures, tuple(remaining), k - 1)
    for layer in layers:
        for signature, ways in layer.items():
            joined = candidate_signature | signature
            result[joined] = result.get(joined, 0) + ways
    return result


def _remaining_constraints_from_counts(signature_counts: dict[int, int]) -> int:
    """So domination/separation constraints chua duoc cover."""
    return signature_counts.get(0, 0) + sum(
        count * (count - 1) // 2 for count in signature_counts.values()
    )


def _candidate_gain(
    total_counts: dict[int, int],
    zero_counts: dict[int, int],
    contains_candidate_counts: dict[int, int],
) -> int:
    """Exact gain khi them cap cot ``x_i(t0), y_i(t0+1)``.

    Trong moi OR-signature hien tai, candidate chia fire set thanh ba nhom:
    khong duoc thay ``00``, duoc thay nhung khong chua sensor ``01``, va co
    chua sensor ``11``. Gain separation la so cap nam o hai nhom khac nhau.
    """
    gain = total_counts.get(0, 0) - zero_counts.get(0, 0)
    for signature, total in total_counts.items():
        zeros = zero_counts.get(signature, 0)
        contains = contains_candidate_counts.get(signature, 0)
        observed_only = total - zeros - contains
        if observed_only < 0:
            raise AssertionError(
                "Partition candidate khong hop le: "
                f"T={total}, A00={zeros}, A11={contains}"
            )
        gain += zeros * observed_only + zeros * contains + observed_only * contains
    return gain


def _split_singleton_partition(
    partition: dict[int, int],
    observation_mask: int,
    candidate_index: int,
    selected_count: int,
) -> dict[int, int]:
    """Them hai bit t0/t1 va chia cac singleton class bang bitset."""
    y_bit = 1 << (2 * selected_count)
    x_bit = 1 << (2 * selected_count + 1)
    candidate_bit = 1 << candidate_index
    new_partition: dict[int, int] = {}

    for signature, members in partition.items():
        contains_candidate = members & candidate_bit
        observed_only = (members & observation_mask) & ~candidate_bit
        unobserved = members & ~observation_mask
        for appended_bits, part in (
            (0, unobserved),
            (y_bit, observed_only),
            (x_bit | y_bit, contains_candidate),
        ):
            if part:
                new_signature = signature | appended_bits
                new_partition[new_signature] = (
                    new_partition.get(new_signature, 0) | part
                )
    return new_partition


def fire_states(vertices: list[int], k: int) -> list[FireState]:
    """Liet ke cac dinh cua graph mo rong: moi F voi 1 <= |F| <= k."""
    number_of_vertices = len(vertices)
    _validate_k(number_of_vertices, k)

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
    """Tra ve thong ke graph mo rong ma khong tao fire state/phu canh.

    Doi thu tu dem: voi sensor ``x``, so state size ``s >= 2`` ke voi x la
    ``C(n,s) - C(n-|Y_x|,s)``, trong do ``Y_x`` la support cua cot y_x.
    """
    vertices = sorted(graph)
    number_of_vertices = len(graph)
    _validate_k(number_of_vertices, k)
    observation_masks, _ = _observation_masks(graph, vertices)
    expanded_vertices = sum(comb(number_of_vertices, size) for size in range(1, k + 1))
    original_edges = sum(
        1
        for vertex, neighbors in graph.items()
        for neighbor in neighbors
        if vertex < neighbor
    )
    auxiliary_edges = 0
    for mask in observation_masks:
        unobserved_vertices = number_of_vertices - mask.bit_count()
        for size in range(2, k + 1):
            observed_states = comb(number_of_vertices, size)
            if unobserved_vertices >= size:
                observed_states -= comb(unobserved_vertices, size)
            auxiliary_edges += observed_states
    expanded_edges = original_edges + auxiliary_edges
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
    x_columns: dict[int, set[FireState]] = {sensor: set() for sensor in vertices}
    y_columns: dict[int, set[FireState]] = {sensor: set() for sensor in vertices}

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
    unseparated_pairs = sum(len(group) * (len(group) - 1) // 2 for group in groups)
    return len(vector_0) + unseparated_pairs


def generalized_id_greedy_without_universe_with_t0(
    graph: Graph, k: int, *, verbose: bool = False
) -> set[int]:
    """Chay exact greedy nen bang singleton partition va OR-count.

    Candidate van la dinh goc va van them dong thoi hai cot ``x_i(t=0)`` va
    ``y_i(t1=t0+1)``. Lazy heap chi dung gain cu lam upper bound; vi vay
    sensor duoc chon giong full greedy (ke ca quy tac tie theo nhan nho).
    """
    vertices = sorted(graph)
    number_of_vertices = len(vertices)
    _validate_k(number_of_vertices, k)
    observation_masks, _ = _observation_masks(graph, vertices)

    partition = {0: (1 << number_of_vertices) - 1}
    singleton_signatures = [0] * number_of_vertices
    selected: set[int] = set()
    number_of_states = sum(comb(number_of_vertices, size) for size in range(1, k + 1))
    initial_constraints = (
        number_of_states + number_of_states * (number_of_states - 1) // 2
    )

    # Moi gain hien tai khong the vuot qua so constraint chua cover. Gain
    # exact cua round truoc la upper bound hop le cho moi round sau.
    lazy_heap = [
        (-initial_constraints, index, index) for index in range(number_of_vertices)
    ]
    heapq.heapify(lazy_heap)
    expected_remaining = initial_constraints
    step = 0

    while True:
        signatures, member_masks, sizes = _partition_items(partition)
        total_counts = _count_signatures(signatures, sizes, k)
        remaining_constraints = _remaining_constraints_from_counts(total_counts)
        if remaining_constraints != expected_remaining:
            raise AssertionError(
                "Gain khong khop voi so constraint con lai: "
                f"{expected_remaining} != {remaining_constraints}"
            )
        if remaining_constraints == 0:
            return selected
        if not lazy_heap:
            raise ValueError("Khong con sensor nao phu duoc constraint moi")

        class_by_signature = {
            signature: class_index for class_index, signature in enumerate(signatures)
        }
        zero_counts_cache: dict[tuple[int, ...], dict[int, int]] = {}
        contains_cache: dict[int, dict[int, int]] = {}
        gain_cache: dict[tuple[int, tuple[int, ...]], int] = {}
        exact_evaluations = 0
        cache_hits = 0

        while lazy_heap:
            _, candidate_rank, candidate_index = heapq.heappop(lazy_heap)
            observation_mask = observation_masks[candidate_index]
            unobserved_counts = tuple(
                size - (members & observation_mask).bit_count()
                for size, members in zip(sizes, member_masks)
            )
            candidate_class = class_by_signature[singleton_signatures[candidate_index]]
            equivalence_key = (candidate_class, unobserved_counts)

            if equivalence_key in gain_cache:
                gain = gain_cache[equivalence_key]
                cache_hits += 1
            else:
                zero_counts = zero_counts_cache.get(unobserved_counts)
                if zero_counts is None:
                    zero_counts = _count_signatures(signatures, unobserved_counts, k)
                    zero_counts_cache[unobserved_counts] = zero_counts

                contains_counts = contains_cache.get(candidate_class)
                if contains_counts is None:
                    contains_counts = _counts_containing_one_vertex(
                        signatures, sizes, candidate_class, k
                    )
                    contains_cache[candidate_class] = contains_counts

                gain = _candidate_gain(total_counts, zero_counts, contains_counts)
                gain_cache[equivalence_key] = gain
                exact_evaluations += 1

            # (-gain, rank) la best possible key cua candidate. Neu no da
            # tot hon upper-bound key dau heap, full greedy khong can cham
            # diem them candidate nao trong round nay.
            if not lazy_heap or (-gain, candidate_rank) <= lazy_heap[0][:2]:
                break
            heapq.heappush(lazy_heap, (-gain, candidate_rank, candidate_index))
        else:
            raise ValueError("Khong con sensor nao phu duoc constraint moi")

        if gain <= 0:
            raise ValueError("Khong con sensor nao phu duoc constraint moi")

        step += 1
        sensor = vertices[candidate_index]
        selected.add(sensor)
        partition = _split_singleton_partition(
            partition, observation_masks[candidate_index], candidate_index, step - 1
        )

        y_bit = 1 << (2 * (step - 1))
        x_bit = 1 << (2 * (step - 1) + 1)
        observed = observation_masks[candidate_index]
        while observed:
            least_significant_bit = observed & -observed
            observed_index = least_significant_bit.bit_length() - 1
            singleton_signatures[observed_index] |= y_bit
            observed ^= least_significant_bit
        singleton_signatures[candidate_index] |= x_bit
        expected_remaining = remaining_constraints - gain

        if verbose:
            print(
                f"Buoc {step}: chon v_{sensor} (gain={gain}); "
                f"constraint con lai={expected_remaining}; "
                f"exact gain={exact_evaluations}; cache hit={cache_hits}; "
                f"singleton classes={len(partition)}"
            )


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


def evaluate_output(graph: Graph, sensors: set[int], k: int) -> tuple[bool, str]:
    """Verify chinh xac bang OR-count, khong materialize fire states."""
    try:
        vertices = sorted(graph)
        _validate_k(len(vertices), k)
        observation_masks, vertex_index = _observation_masks(graph, vertices)
    except ValueError as exc:
        return False, str(exc)

    unknown_sensors = sensors - graph.keys()
    if unknown_sensors:
        return False, f"Sensor khong thuoc graph: {sorted(unknown_sensors)}"

    partition = {0: (1 << len(vertices)) - 1}
    for selected_count, sensor in enumerate(sorted(sensors)):
        sensor_index = vertex_index[sensor]
        partition = _split_singleton_partition(
            partition,
            observation_masks[sensor_index],
            sensor_index,
            selected_count,
        )

    signatures, _, sizes = _partition_items(partition)
    counts = _count_signatures(signatures, sizes, k)
    undominated = counts.get(0, 0)
    if undominated:
        return False, f"Con {undominated} fire state chua duoc dominate"

    largest_collision = max(counts.values(), default=0)
    if largest_collision > 1:
        collision_classes = sum(count > 1 for count in counts.values())
        return (
            False,
            f"Con {collision_classes} signature collision; "
            f"class lon nhat co {largest_collision} fire state",
        )

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
    expanded_vertices, expanded_edges, total_constraints = expanded_graph_statistics(
        SAMPLE_GRAPH, args.k
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
