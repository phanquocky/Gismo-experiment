"""
Interactive HTML viewer for the network files under datasets/.

Reuses the same .mtx/.txt/.edges parser as the rest of the experiment
pipeline (baseline/tools/network_to_matrix.py), builds a networkx graph, and
renders it with pyvis as a single self-contained HTML file you can open in a
browser to pan/zoom/drag nodes and hover for degree info.

Some datasets have 1M+ nodes, which is unusable to render directly, so the
view is always capped at --max-nodes (top-degree nodes are kept if a
selection exceeds it). Use --component/--node to focus on a specific part of
a large graph instead of relying on the cap alone.

Usage:
    # whole graph (auto-capped to --max-nodes by degree if it's too big)
    python src/view/view_graph.py --network datasets/socfb-nips-ego.edges

    # just the largest connected component
    python src/view/view_graph.py --network datasets/road-minnesota.mtx --component largest

    # ego network: node 1 and everything within 2 hops
    python src/view/view_graph.py --network datasets/socfb-nips-ego.edges --node 1 --hops 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx
from pyvis.network import Network

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "baseline" / "tools"))

from network_to_matrix import parse_network  # noqa: E402

DEFAULT_MAX_NODES = 400
DEFAULT_OUT_DIR = Path(__file__).parent / "output"


def _connected_components(nodes: List[int], adj: Dict[int, Set[int]]) -> List[List[int]]:
    visited: Set[int] = set()
    components: List[List[int]] = []
    for start in nodes:
        if start in visited:
            continue
        comp: List[int] = []
        stack = [start]
        visited.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if w not in visited:
                    visited.add(w)
                    stack.append(w)
        components.append(comp)
    return components


def _ego_subset(adj: Dict[int, Set[int]], center: int, hops: int) -> Set[int]:
    visited = {center}
    frontier = {center}
    for _ in range(hops):
        nxt: Set[int] = set()
        for u in frontier:
            nxt |= adj[u] - visited
        if not nxt:
            break
        visited |= nxt
        frontier = nxt
    return visited


def _top_degree_sample(nodes: List[int], adj: Dict[int, Set[int]], k: int) -> Set[int]:
    ranked = sorted(nodes, key=lambda v: len(adj[v]), reverse=True)
    return set(ranked[:k])


def build_subgraph(
    nodes: List[int],
    adj: Dict[int, Set[int]],
    component: Optional[str],
    node: Optional[int],
    hops: int,
    max_nodes: int,
) -> nx.Graph:
    if node is not None:
        if node not in adj:
            raise SystemExit(f"--node {node} is not in this graph")
        keep = _ego_subset(adj, node, hops)
        mode = f"ego network of node {node} ({hops} hop(s), {len(keep)} nodes)"
    elif component is not None:
        components = sorted(_connected_components(nodes, adj), key=len, reverse=True)
        if component == "largest":
            keep = set(components[0])
            mode = f"largest connected component ({len(keep)} of {len(components)} components)"
        else:
            idx = int(component)
            if idx >= len(components):
                raise SystemExit(f"--component {idx} out of range (graph has {len(components)} components)")
            keep = set(components[idx])
            mode = f"component #{idx} ({len(keep)} nodes)"
    else:
        keep = set(nodes)
        mode = "full graph"

    if len(keep) > max_nodes:
        keep = _top_degree_sample(list(keep), adj, max_nodes)
        mode += f"  ->  capped to top {max_nodes} nodes by degree"

    G = nx.Graph()
    G.add_nodes_from(keep)
    for u in keep:
        for v in adj[u]:
            if v in keep and u < v:
                G.add_edge(u, v)

    print(f"Showing    : {mode}")
    print(f"Rendering  : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def render(G: nx.Graph, out_path: Path, height: str, highlight: Optional[int]) -> None:
    net = Network(
        height=height, width="100%", bgcolor="#ffffff", font_color="#222222",
        cdn_resources="in_line",  # single self-contained HTML file
    )
    net.force_atlas_2based()

    degrees = dict(G.degree())
    max_deg = max(degrees.values(), default=1)
    for v in G.nodes():
        deg = degrees[v]
        size = 8 + 22 * (deg / max_deg) ** 0.5
        color = "#e74c3c" if v == highlight else "#3f7fbf"
        net.add_node(v, label=str(v), title=f"node {v}  degree={deg}", size=size, color=color)
    for u, v in G.edges():
        net.add_edge(u, v)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_path), open_browser=False, notebook=False)
    print(f"Saved      -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", required=True, help="Path to a .mtx/.txt/.edges file (e.g. datasets/soc-karate.mtx)")
    parser.add_argument("--out", help="Output HTML path (default: src/view/output/<name>.html)")
    parser.add_argument("--component", help="'largest' or a 0-based component index to isolate")
    parser.add_argument("--node", type=int, help="Show the ego network around this node ID")
    parser.add_argument("--hops", type=int, default=1, help="Hop radius for --node (default: 1)")
    parser.add_argument(
        "--max-nodes", type=int, default=DEFAULT_MAX_NODES,
        help=f"Cap on nodes rendered (default: {DEFAULT_MAX_NODES}); oversized selections are cut down to the top-degree nodes",
    )
    parser.add_argument("--height", default="800px", help="Canvas height (default: 800px)")
    args = parser.parse_args()

    network_path = Path(args.network)
    nodes, adj = parse_network(str(network_path))
    n_edges = sum(len(v) for v in adj.values()) // 2
    print(f"Loaded     : {network_path.name}  ({len(nodes)} nodes, {n_edges} edges)")

    G = build_subgraph(nodes, adj, args.component, args.node, args.hops, args.max_nodes)

    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"{network_path.stem}.html"
    render(G, out_path, args.height, highlight=args.node)


if __name__ == "__main__":
    main()
