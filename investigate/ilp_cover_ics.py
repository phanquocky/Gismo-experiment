"""
ILP-based minimum sensor set finder for the Cover-ICS problem.

Definitions (see web-gcnf/gismo_investigate/plan.md §2 and investigate/dl-separable/docs.md):

    Given Gamma = (V, E), an integer k (max simultaneous fires) and B >= k
    (precision budget), D subset V is a B-Cover-ICS if for every U subset V
    with |U| <= k:

        Soundness      U subset phi_D(s_U)
        Precision      |phi_D(s_U)| <= B
        Non-triviality s_U != empty whenever U != empty

    where s_U = N+(U) ∩ D is the signature and
    phi_D(s) = union of every W (|W| <= k) with s_W = s is the natural decoder.

Constraint 1 (Non-triviality) is exactly "D is a dominating set" (proved in
plan.md §2.5) -- linear in the sensor-selection variables z_v, no k/B needed.

Constraint 2 (Soundness) is automatic: U is always one of the W's unioned
into phi_D(s_U), so it never needs to be encoded.

Constraint 3 (Precision) is the interesting one. The obvious encoding --
"for every pair U1,U2 with |U1 ∪ U2| > B, force some sensor to separate
them" -- is only a NECESSARY condition, not sufficient: Precision bounds the
union of the whole same-signature equivalence class, and a class can have
every *pairwise* union <= B while its *total* union exceeds B.

Counterexample: star graph K_{1,m}, hub h, leaves l_1..l_m, k=1, B=2,
D={h}. Every single-leaf fire {l_i} has signature {h} -- identical
for all m leaves. Every pair {l_i},{l_j} has union size 2 (not > B=2), so a
pairwise-only ILP adds no separating constraint and wrongly reports D={h}
(size 1) as valid. The true decoded class is all m leaves at once, size m.
See solve_cover_ics_naive() below, kept only to demonstrate this gap, and
demo_star() in main() which reproduces it.

solve_cover_ics() fixes this with lazy constraint generation: solve with
domination constraints only, then use an oracle that groups *all* fire sets
by their actual signature under the candidate D (not just checking a fixed
pair), finds classes whose union exceeds B, and adds pairwise separating
cuts for every pair inside a violating class. Repeat until no class
violates Precision. Each round strictly shrinks some violating class (or
the loop terminates), so this converges to a genuinely optimal minimum
B-Cover-ICS, not just a feasible one.

Scalability note: enumerating all U with |U| <= k costs sum_{i=1..k} C(n,i),
same order of magnitude as the (d,l)-disjunct ILPs in ILP/ilp.md. Practical
directly for small n/k; for larger instances this would need the same kind
of lazy/lifted enumeration on the fire-set side too (not implemented here).

solve_cover_ics_oneshot() is the same exact solver with the loop removed:
every Precision cut is computed by minimal_violating_groups() and added to
the model before the single prob.solve() call. It answers the same ILP,
just built at a different time -- see that function's docstring for why
this trades the lazy version's per-round re-solve for a potentially much
larger upfront combinatorial enumeration.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Set, Tuple

import pulp

sys.path.insert(0, str(Path(__file__).parent.parent / "baseline" / "tools"))
from network_to_matrix import parse_network  # noqa: E402


Node = Any
FireSet = FrozenSet[Node]
Signature = FrozenSet[Node]


class CoverICSInfeasible(Exception):
    """No D can satisfy Precision for the given (k, B): some fire sets are
    structurally indistinguishable (true twins -- identical closed
    neighbourhoods) under this signature, and their combined union already
    exceeds B. See distinguishing_set()."""


# ── Graph / signature primitives ────────────────────────────────────────────

def closed_neighborhood(v: Node, adj: Dict[Node, Set[Node]]) -> Set[Node]:
    return {v} | adj.get(v, set())


def signature(U: FireSet, D: FrozenSet[Node], adj: Dict[Node, Set[Node]]) -> Signature:
    """s_U = N+_1(U) ∩ D."""
    n_plus_U: Set[Node] = set()
    for u in U:
        n_plus_U |= closed_neighborhood(u, adj)
    return frozenset(n_plus_U) & D


def enumerate_fire_sets(nodes: List[Node], k: int) -> List[FireSet]:
    """All non-empty U subset V with |U| <= k. Empty U is excluded: Non-triviality
    (== domination, enforced separately) already guarantees it can't be
    confused with any real fire, so it never contributes a Precision violation."""
    fire_sets: List[FireSet] = []
    for size in range(1, k + 1):
        fire_sets.extend(frozenset(c) for c in combinations(nodes, size))
    return fire_sets


def distinguishing_set(U1: FireSet, U2: FireSet, nodes: List[Node], adj: Dict[Node, Set[Node]]) -> Set[Node]:
    """Vertices v such that placing a sensor at v alone makes s_{U1} != s_{U2}:
        [U1 ∩ N+_1(v) != empty] xor [U2 ∩ N+_1(v) != empty]

    Can be EMPTY even when U1 != U2 -- e.g. U1={a}, U2={b} with a,b true
    twins (N[a] == N[b]) are then indistinguishable by any sensor placement
    at all. Callers must handle the empty case (see solve_cover_ics and
    solve_cover_ics_naive).
    """
    dist: Set[Node] = set()
    for v in nodes:
        nb = closed_neighborhood(v, adj)
        b1_1, b1_2 = bool(U1 & nb), bool(U2 & nb)
        if b1_1 != b1_2:
            dist.add(v)
    return dist


def find_violating_groups(
    D: FrozenSet[Node], fire_sets: List[FireSet], adj: Dict[Node, Set[Node]], B: int
) -> List[List[FireSet]]:
    """Group fire sets by their true signature under D; return the member
    lists of every group whose union exceeds the precision budget B."""
    groups: Dict[Signature, List[FireSet]] = defaultdict(list)
    for U in fire_sets:
        groups[signature(U, D, adj)].append(U)

    violations = []
    for members in groups.values():
        union = frozenset().union(*members)
        if len(union) > B:
            violations.append(members)
    return violations


# ── Exact solver (lazy constraint generation) ───────────────────────────────

def solve_cover_ics(
    nodes: List[Node],
    adj: Dict[Node, Set[Node]],
    k: int,
    B: int,
    max_rounds: int = 200,
    msg: bool = False,
) -> Dict[str, Any]:
    """Minimum B-Cover-ICS via ILP with lazily-generated Precision cuts.

    Correct by construction: a cut is only ever added after the oracle finds
    a *real* signature class (under the current candidate D) whose union
    exceeds B, and the loop only stops once no such class exists.
    Cut shape. For a violating group {m_1,...,m_t} (union > B), the true
    necessary condition on any valid D' is just "not all of them still share
    a signature under D'", i.e. OR_{i<j} (D' separates m_i, m_j). Separating
    a pair (i,j) is itself "some v in Dist(m_i,m_j) is in D'", so the full
    condition is an OR of ORs. Because every disjunct here has the form
    "sum over some set S >= 1" (binary indicators), OR_i(sum_{S_i} z >= 1)
    is *exactly* equivalent to sum_{union(S_i)} z >= 1 (a hit in any S_i is
    a hit in the union, and a hit in the union lies in some S_i). So one
    constraint per violating group -- summing z over the union of
    Dist(m_i,m_j) across all internal pairs -- is the correct, minimal cut.
    An earlier version of this function added a *separate* constraint per
    pair instead, which is strictly stronger than necessary: it forces
    every pair apart simultaneously (full singleton separation of the
    whole class) rather than just breaking the union below B, and
    overshoots the true optimum (verified on the K_{1,4} example: it found
    size 4 instead of the true optimum 3).
    """
    if B < k:
        raise ValueError(f"B must be >= k (got B={B}, k={k})")

    fire_sets = enumerate_fire_sets(nodes, k)

    prob = pulp.LpProblem("cover_ics", pulp.LpMinimize)
    z = {v: pulp.LpVariable(f"z_{v}", cat="Binary") for v in nodes}
    prob += pulp.lpSum(z.values())

    # Constraint 1 -- Non-triviality == domination (plan.md §2.5)
    for v in nodes:
        prob += pulp.lpSum(z[u] for u in closed_neighborhood(v, adj)) >= 1

    solver = pulp.PULP_CBC_CMD(msg=msg)
    added_cut_keys: Set[FrozenSet[Node]] = set()
    total_cuts = 0

    for round_idx in range(1, max_rounds + 1):
        prob.solve(solver)
        if pulp.LpStatus[prob.status] != "Optimal":
            raise RuntimeError(f"ILP not optimal at round {round_idx}: {pulp.LpStatus[prob.status]}")

        D = frozenset(v for v in nodes if z[v].varValue > 0.5)
        violations = find_violating_groups(D, fire_sets, adj, B)

        if not violations:
            return {
                "D": [v for v in nodes if v in D],
                "size": len(D),
                "rounds": round_idx,
                "cuts_added": total_cuts,
            }

        new_cuts = 0
        for members in violations:
            dist_group: Set[Node] = set()
            for U1, U2 in combinations(members, 2):
                dist_group |= distinguishing_set(U1, U2, nodes, adj)

            if not dist_group:
                # Every pair in this class is a true twin (distinguishing_set
                # is empty for all of them): no D, of any size, could ever
                # break them apart, and their union already exceeds B.
                raise CoverICSInfeasible(
                    f"k={k}, B={B}: fire sets {[sorted(m, key=str) for m in members]} "
                    f"are structurally indistinguishable (union size "
                    f"{len(frozenset().union(*members))} > B={B}); no sensor "
                    "placement can separate them."
                )

            key = frozenset(dist_group)
            if key in added_cut_keys:
                continue
            added_cut_keys.add(key)
            prob += pulp.lpSum(z[v] for v in dist_group) >= 1
            new_cuts += 1

        if new_cuts == 0:
            # A violating class survived with the identical cut already in
            # place -- contradicts how separation works (adding a cut makes
            # the old D infeasible, so D must change next round) and would
            # mean a solver-tolerance bug (e.g. a 0/1 value rounded wrong).
            raise RuntimeError(
                f"Round {round_idx}: violation persists with no new cuts to add "
                "(unexpected -- check solver tolerances)"
            )
        total_cuts += new_cuts

    raise RuntimeError(f"Did not converge within {max_rounds} rounds")


# ── Exact solver (single-shot: all cuts computed before the one solve()) ───

def minimal_violating_groups(
    fire_sets: List[FireSet], B: int, max_frontier: int = 20_000
) -> List[List[FireSet]]:
    """All inclusion-minimal subsets of fire_sets whose union exceeds B.

    Precision holds for a candidate D iff *every* subset W of fire_sets with
    |union(W)| > B contains a pair D separates. Proof sketch: the true
    signature classes under D partition fire_sets, and two members sit in
    different classes iff D separates them (that's exactly what "different
    signature" means). So an oversized W is either entirely inside one
    class -- making that class's own union oversized too, violating
    Precision directly -- or it spans >= 2 classes, which forces a
    separated pair somewhere inside W. Checking only inclusion-minimal such
    W is enough: any larger oversized W contains a minimal-violating
    subset, and that subset's separated pair is also a pair inside W.

    This is what lets solve_cover_ics_oneshot() below add every Precision
    cut *before* solving, instead of solve_cover_ics()'s solve-inspect-cut
    loop -- but finding all of them is itself a real search: "union(W) <= B"
    is downward closed (dropping members can't grow a union), the same
    monotonicity Apriori frequent-itemset mining relies on, so minimal
    violators can be found level by level -- a candidate of size m+1 is
    only tested once every one of its size-m subsets is already known
    non-violating (Apriori join+prune), which guarantees any violation
    found this way is minimal. Worst case this is still exponential in the
    number of fire_sets (same caveat as enumerate_fire_sets itself): e.g. a
    loose budget (B well above 2k) leaves nearly every pair non-violating,
    so almost the whole frontier survives to the next level and the
    level's O(frontier^2) join blows up well before any real graph would
    trouble solve_cover_ics(). max_frontier turns that into a fast,
    legible RuntimeError instead of a multi-minute hang -- raise it only
    once you've confirmed the run time is actually acceptable.
    """
    n = len(fire_sets)
    valid: Dict[FrozenSet[int], FrozenSet[Node]] = {frozenset({i}): fire_sets[i] for i in range(n)}
    minimal: List[List[FireSet]] = []

    while valid:
        if len(valid) > max_frontier:
            raise RuntimeError(
                f"minimal_violating_groups: frontier grew to {len(valid)} candidate "
                f"groups (> max_frontier={max_frontier}) before converging -- this "
                "(k, B) combination is a bad fit for the upfront/single-shot solver "
                "(likely because B is loose relative to k, so few pairs separate and "
                "the search stalls at a wide level); use solve_cover_ics() instead."
            )
        keys = list(valid.keys())
        target_size = len(keys[0]) + 1
        next_level: Dict[FrozenSet[int], FrozenSet[Node]] = {}
        seen: Set[FrozenSet[int]] = set()

        for a, b in combinations(range(len(keys)), 2):
            cand = keys[a] | keys[b]
            if len(cand) != target_size or cand in seen:
                continue
            if not all(frozenset(cand - {i}) in valid for i in cand):
                continue  # a smaller subset of cand already violates -- cand can't be minimal
            seen.add(cand)

            union = valid[keys[a]] | valid[keys[b]]
            if len(union) > B:
                minimal.append([fire_sets[i] for i in sorted(cand)])
            else:
                next_level[cand] = union

        valid = next_level

    return minimal


def solve_cover_ics_oneshot(
    nodes: List[Node],
    adj: Dict[Node, Set[Node]],
    k: int,
    B: int,
    msg: bool = False,
    max_frontier: int = 20_000,
) -> Dict[str, Any]:
    """Minimum B-Cover-ICS via a single ILP solve.

    Every Precision cut is computed from minimal_violating_groups() (see its
    docstring for why enumerating only those groups is correct and
    sufficient) and added to the model before the one and only prob.solve()
    call -- no solve-inspect-cut loop. Same cut shape as solve_cover_ics():
    for a violating group, the union of Dist(Ui,Uj) over every internal
    pair is the minimal correct cut (see solve_cover_ics()'s docstring for
    the OR-of-sums equivalence that makes this exact, not just necessary).

    Produces the same optimum as solve_cover_ics() -- both encode the exact
    same feasible region, just built at different times -- but the upfront
    enumeration can make the *model* much bigger on graphs where
    minimal_violating_groups() finds many groups, whereas solve_cover_ics()
    only ever adds cuts for classes an actual candidate D produced.
    """
    if B < k:
        raise ValueError(f"B must be >= k (got B={B}, k={k})")

    fire_sets = enumerate_fire_sets(nodes, k)

    prob = pulp.LpProblem("cover_ics_oneshot", pulp.LpMinimize)
    z = {v: pulp.LpVariable(f"z_{v}", cat="Binary") for v in nodes}
    prob += pulp.lpSum(z.values())

    # Constraint 1 -- Non-triviality == domination (plan.md §2.5)
    for v in nodes:
        prob += pulp.lpSum(z[u] for u in closed_neighborhood(v, adj)) >= 1

    # Constraint 3 -- Precision, every cut computed upfront
    cuts_added = 0
    for members in minimal_violating_groups(fire_sets, B, max_frontier=max_frontier):
        dist_group: Set[Node] = set()
        for U1, U2 in combinations(members, 2):
            dist_group |= distinguishing_set(U1, U2, nodes, adj)

        if not dist_group:
            raise CoverICSInfeasible(
                f"k={k}, B={B}: fire sets {[sorted(m, key=str) for m in members]} "
                f"are structurally indistinguishable (union size "
                f"{len(frozenset().union(*members))} > B={B}); no sensor "
                "placement can separate them."
            )
        prob += pulp.lpSum(z[v] for v in dist_group) >= 1
        cuts_added += 1

    prob.solve(pulp.PULP_CBC_CMD(msg=msg))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"ILP not optimal: {pulp.LpStatus[prob.status]}")

    D = frozenset(v for v in nodes if z[v].varValue > 0.5)
    return {"D": [v for v in nodes if v in D], "size": len(D), "cuts_added": cuts_added}


# ── Naive pairwise-only solver (kept to demonstrate the gap) ───────────────

def solve_cover_ics_naive(
    nodes: List[Node], adj: Dict[Node, Set[Node]], k: int, B: int, msg: bool = False
) -> Dict[str, Any]:
    """Pairwise-only ILP: separate every pair U1,U2 with |U1 ∪ U2| > B.

    NOT a correct Cover-ICS solver in general -- see the module docstring
    and the star-graph counterexample in demo_star(). Included only so the
    gap versus solve_cover_ics() can be shown side by side.
    """
    fire_sets = enumerate_fire_sets(nodes, k)

    prob = pulp.LpProblem("cover_ics_naive", pulp.LpMinimize)
    z = {v: pulp.LpVariable(f"z_{v}", cat="Binary") for v in nodes}
    prob += pulp.lpSum(z.values())

    for v in nodes:
        prob += pulp.lpSum(z[u] for u in closed_neighborhood(v, adj)) >= 1

    for U1, U2 in combinations(fire_sets, 2):
        if len(U1 | U2) > B:
            dist = distinguishing_set(U1, U2, nodes, adj)
            if not dist:
                raise CoverICSInfeasible(
                    f"k={k}, B={B}: fire sets {sorted(U1, key=str)} and "
                    f"{sorted(U2, key=str)} are structurally indistinguishable "
                    f"and their union already exceeds B={B}."
                )
            prob += pulp.lpSum(z[v] for v in dist) >= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=msg))
    D = frozenset(v for v in nodes if z[v].varValue > 0.5)
    return {"D": [v for v in nodes if v in D], "size": len(D)}


# ── Independent verifier (definition-level, works on any D from any source) ─

def verify_cover_ics(
    D: List[Node], nodes: List[Node], adj: Dict[Node, Set[Node]], k: int, B: int
) -> Dict[str, Any]:
    """Exhaustively check D against the formal Soundness/Precision/
    Non-triviality definition. Useful for validating output from any
    solver (this one, the naive one, or an external tool like GISMO-Cover)."""
    D_set = frozenset(D)
    fire_sets = enumerate_fire_sets(nodes, k)

    undominated = [v for v in nodes if not (closed_neighborhood(v, adj) & D_set)]

    groups: Dict[Signature, List[FireSet]] = defaultdict(list)
    for U in fire_sets:
        groups[signature(U, D_set, adj)].append(U)

    worst_size, worst_sig, worst_members = 0, None, []
    for s, members in groups.items():
        union = frozenset().union(*members)
        if len(union) > worst_size:
            worst_size, worst_sig, worst_members = len(union), s, members

    return {
        "non_trivial": len(undominated) == 0,
        "undominated_nodes": undominated,
        "precision_ok": worst_size <= B,
        "worst_class_size": worst_size,
        "worst_class_signature": worst_sig,
        "worst_class_members": [sorted(m, key=str) for m in worst_members],
        "valid_B_cover_ics": len(undominated) == 0 and worst_size <= B,
    }


# ── Demo graphs ──────────────────────────────────────────────────────────────

def hotel_graph() -> Tuple[List[Node], Dict[Node, Set[Node]]]:
    """5-room hotel from plan.md §1.2: G-O-P / B-R, plus (O,R) and (R,P)."""
    edges = [("G", "O"), ("G", "B"), ("O", "R"), ("O", "P"), ("B", "R"), ("R", "P")]
    adj: Dict[Node, Set[Node]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    nodes = ["B", "G", "O", "P", "R"]
    return nodes, dict(adj)


def star_graph(m: int = 4) -> Tuple[List[Node], Dict[Node, Set[Node]]]:
    """K_{1,m}: hub 'h' plus leaves 'l1'..'lm' -- the Precision counterexample."""
    adj: Dict[Node, Set[Node]] = defaultdict(set)
    leaves = [f"l{i}" for i in range(1, m + 1)]
    for leaf in leaves:
        adj["h"].add(leaf)
        adj[leaf].add("h")
    nodes = ["h"] + leaves
    return nodes, dict(adj)


# ── Reporting ────────────────────────────────────────────────────────────────

def _timed(fn, *args, **kwargs) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def _print_verification(v: Dict[str, Any], B: int) -> None:
    print(f"  Non-triviality (dominating set) : {'OK' if v['non_trivial'] else 'FAIL -- undominated: ' + str(v['undominated_nodes'])}")
    print(f"  Precision (worst class <= B={B})  : {'OK' if v['precision_ok'] else 'FAIL'} (worst class size = {v['worst_class_size']})")
    if not v["precision_ok"]:
        print(f"    worst class signature : {v['worst_class_signature']}")
        print(f"    worst class members   : {v['worst_class_members']}")
    print(f"  Valid B-Cover-ICS                : {v['valid_B_cover_ics']}")


def demo_hotel() -> None:
    print("=" * 70)
    print("Demo 1: 5-room hotel (plan.md example), k=1, B=2")
    print("=" * 70)
    nodes, adj = hotel_graph()
    k, B = 1, 2

    result, t_lazy = _timed(solve_cover_ics, nodes, adj, k, B)
    print(f"\nsolve_cover_ics (lazy cuts)   -> D = {result['D']}  (size {result['size']}, "
          f"{result['rounds']} round(s), {result['cuts_added']} precision cut(s), "
          f"{t_lazy * 1000:.2f} ms)")

    oneshot, t_oneshot = _timed(solve_cover_ics_oneshot, nodes, adj, k, B)
    print(f"solve_cover_ics_oneshot       -> D = {oneshot['D']}  (size {oneshot['size']}, "
          f"{oneshot['cuts_added']} precision cut(s), {t_oneshot * 1000:.2f} ms)")

    v = verify_cover_ics(result["D"], nodes, adj, k, B)
    _print_verification(v, B)
    print("\n(Under the full (x,y) signature, plan.md's GISMO-Cover run found the "
          "exact GICS D={O,R}, size 2. Here, with x_i dropped, minimum size is "
          "still 2 but D need not be an exact GICS anymore -- e.g. {B} and {G} "
          "now genuinely share a signature, using up the B=2 slack.)")


def demo_star(m: int = 4) -> None:
    print("\n" + "=" * 70)
    print(f"Demo 2: star graph K_1,{m}, k=1, B=2 -- naive vs. exact")
    print("=" * 70)
    nodes, adj = star_graph(m)
    k, B = 1, 2

    naive, t_naive = _timed(solve_cover_ics_naive, nodes, adj, k, B)
    print(f"\nsolve_cover_ics_naive (pairwise-only)  -> D = {naive['D']}  (size {naive['size']}, "
          f"{t_naive * 1000:.2f} ms)")
    v_naive = verify_cover_ics(naive["D"], nodes, adj, k, B)
    _print_verification(v_naive, B)

    exact, t_lazy = _timed(solve_cover_ics, nodes, adj, k, B)
    print(f"\nsolve_cover_ics (exact, lazy cuts)     -> D = {exact['D']}  (size {exact['size']}, "
          f"{exact['rounds']} round(s), {exact['cuts_added']} precision cut(s), "
          f"{t_lazy * 1000:.2f} ms)")
    v_exact = verify_cover_ics(exact["D"], nodes, adj, k, B)
    _print_verification(v_exact, B)

    oneshot, t_oneshot = _timed(solve_cover_ics_oneshot, nodes, adj, k, B)
    print(f"\nsolve_cover_ics_oneshot (exact, all cuts upfront) -> D = {oneshot['D']}  "
          f"(size {oneshot['size']}, {oneshot['cuts_added']} precision cut(s), "
          f"{t_oneshot * 1000:.2f} ms)")
    v_oneshot = verify_cover_ics(oneshot["D"], nodes, adj, k, B)
    _print_verification(v_oneshot, B)

    print(f"\n=> naive under-counts by {exact['size'] - naive['size']} sensor(s) while "
          f"silently violating Precision (class size {v_naive['worst_class_size']} > B={B}).")


def run_from_file(network_file: str, k: int, B: int, run_naive: bool, run_oneshot: bool) -> None:
    nodes, adj = parse_network(network_file)
    print("=" * 70)
    print(f"{network_file}  |  {len(nodes)} nodes  |  k={k}  B={B}")
    print("=" * 70)

    if run_naive:
        naive, t_naive = _timed(solve_cover_ics_naive, nodes, adj, k, B)
        v_naive = verify_cover_ics(naive["D"], nodes, adj, k, B)
        print(f"\nnaive (pairwise-only) -> size {naive['size']}, {t_naive * 1000:.2f} ms")
        _print_verification(v_naive, B)

    exact, t_lazy = _timed(solve_cover_ics, nodes, adj, k, B)
    v_exact = verify_cover_ics(exact["D"], nodes, adj, k, B)
    print(f"\nexact (lazy cuts) -> D = {exact['D']}")
    print(f"size {exact['size']}, {exact['rounds']} round(s), {exact['cuts_added']} precision "
          f"cut(s), {t_lazy * 1000:.2f} ms")
    _print_verification(v_exact, B)

    if run_oneshot:
        oneshot, t_oneshot = _timed(solve_cover_ics_oneshot, nodes, adj, k, B)
        v_oneshot = verify_cover_ics(oneshot["D"], nodes, adj, k, B)
        print(f"\nexact (all cuts upfront) -> D = {oneshot['D']}")
        print(f"size {oneshot['size']}, {oneshot['cuts_added']} precision cut(s), "
              f"{t_oneshot * 1000:.2f} ms")
        _print_verification(v_oneshot, B)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ILP solver for the Cover-ICS problem (Soundness / Precision / Non-triviality)."
    )
    parser.add_argument(
        "network", nargs="?", default=None,
        help="Path to a network file (.mtx or edge-list .txt). If omitted, runs the built-in hotel + star demos.",
    )
    parser.add_argument("-k", type=int, default=1, help="Max simultaneous fires (default: 1)")
    parser.add_argument("-B", type=int, default=None, help="Precision budget, B >= k (default: k+1)")
    parser.add_argument(
        "--naive", action="store_true",
        help="Also run the (incorrect) pairwise-only ILP for comparison",
    )
    parser.add_argument(
        "--no-oneshot", dest="oneshot", action="store_false",
        help="Skip the single-solve (all cuts upfront) ILP, e.g. if it enumerates too many cuts",
    )
    args = parser.parse_args()

    if args.network:
        B = args.B if args.B is not None else args.k + 1
        run_from_file(args.network, args.k, B, args.naive, args.oneshot)
    else:
        demo_hotel()
        demo_star()


if __name__ == "__main__":
    main()
