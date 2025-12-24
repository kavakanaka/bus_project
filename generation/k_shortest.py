import heapq
from typing import Dict, List, Tuple

from core.instance import Instance
from core.route import Route


def _build_adj(instance: Instance) -> Dict[int, List[Tuple[int, float]]]:
    adj: Dict[int, List[Tuple[int, float]]] = {}
    for e in instance.edges:
        adj.setdefault(e.u, []).append((e.v, e.travel_time))
    return adj


def _dijkstra(
    adj: Dict[int, List[Tuple[int, float]]],
    source: int,
    target: int,
    banned_edges: set[Tuple[int, int]] = set(),
    banned_nodes: set[int] = set(),
) -> List[int] | None:
    pq = [(0.0, source, [source])]
    best: Dict[int, float] = {}

    while pq:
        dist, u, path = heapq.heappop(pq)
        if u == target:
            return path

        if dist > best.get(u, float("inf")):
            continue

        for v, w in adj.get(u, []):
            if (u, v) in banned_edges:
                continue
            if v in banned_nodes:
                continue
            nd = dist + w
            if nd < best.get(v, float("inf")):
                best[v] = nd
                heapq.heappush(pq, (nd, v, path + [v]))

    return None


def yen_k_shortest_paths(
    instance: Instance,
    source: int,
    target: int,
    k: int,
) -> List[Route]:
    """
    Yen's algorithm for K shortest loopless paths.
    """
    adj = _build_adj(instance)

    A: List[List[int]] = []
    B: List[Tuple[float, List[int]]] = []

    first = _dijkstra(adj, source, target)
    if first is None:
        return []

    A.append(first)

    for k_i in range(1, k):
        for i in range(len(A[-1]) - 1):
            spur_node = A[-1][i]
            root_path = A[-1][: i + 1]

            banned_edges = set()
            banned_nodes = set(root_path[:-1])

            for p in A:
                if p[: i + 1] == root_path and i + 1 < len(p):
                    banned_edges.add((p[i], p[i + 1]))

            spur_path = _dijkstra(
                adj,
                spur_node,
                target,
                banned_edges=banned_edges,
                banned_nodes=banned_nodes,
            )

            if spur_path is None:
                continue

            total_path = root_path[:-1] + spur_path
            cost = 0.0
            for j in range(len(total_path) - 1):
                u = total_path[j]
                v = total_path[j + 1]
                for vv, w in adj[u]:
                    if vv == v:
                        cost += w
                        break

            heapq.heappush(B, (cost, total_path))

        if not B:
            break

        _, next_path = heapq.heappop(B)
        A.append(next_path)

    return [Route(p) for p in A]
