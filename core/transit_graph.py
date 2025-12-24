from typing import Dict, Tuple, List
import heapq

from core.instance import Instance
from core.route import RouteSet


Node = Tuple[int, int]  # (stop_id, route_index)


class TransitGraph:
    def __init__(
        self,
        instance: Instance,
        route_set: RouteSet,
        transfer_penalty: float = 5.0,
    ):
        self.instance = instance
        self.route_set = route_set
        self.transfer_penalty = transfer_penalty

        self.adj: Dict[Node, List[Tuple[Node, float]]] = {}
        self._build()

    def _build(self) -> None:
        """
        Build transit graph from routes.
        """
        # Initialize nodes
        for r_idx, route in enumerate(self.route_set.routes):
            for stop in route.stops:
                self.adj.setdefault((stop, r_idx), [])

        # Map (u, v) -> travel_time from instance
        edge_time: Dict[Tuple[int, int], float] = {}
        for e in self.instance.edges:
            edge_time[(e.u, e.v)] = e.travel_time

        # Add movement edges along routes
        for r_idx, route in enumerate(self.route_set.routes):
            for i in range(len(route.stops) - 1):
                u = route.stops[i]
                v = route.stops[i + 1]
                if (u, v) not in edge_time:
                    raise ValueError(
                        f"No edge ({u},{v}) in instance for route {r_idx}"
                    )
                w = edge_time[(u, v)]
                self.adj[(u, r_idx)].append(((v, r_idx), w))

        # Add transfer edges
        stop_to_routes: Dict[int, List[int]] = {}
        for r_idx, route in enumerate(self.route_set.routes):
            for stop in route.stops:
                stop_to_routes.setdefault(stop, []).append(r_idx)

        for stop, routes in stop_to_routes.items():
            for r1 in routes:
                for r2 in routes:
                    if r1 != r2:
                        self.adj[(stop, r1)].append(
                            ((stop, r2), self.transfer_penalty)
                        )

    def shortest_path(self, origin: int, destination: int) -> float:
        """
        Compute shortest travel time between two stops.
        """
        pq: List[Tuple[float, Node]] = []
        dist: Dict[Node, float] = {}

        # start from any route that contains origin
        for r_idx, route in enumerate(self.route_set.routes):
            if origin in route.stops:
                node = (origin, r_idx)
                dist[node] = 0.0
                heapq.heappush(pq, (0.0, node))

        best = float("inf")

        while pq:
            cur_dist, node = heapq.heappop(pq)
            if cur_dist > dist.get(node, float("inf")):
                continue

            stop, _ = node
            if stop == destination:
                best = min(best, cur_dist)
                continue

            for nxt, w in self.adj.get(node, []):
                nd = cur_dist + w
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt] = nd
                    heapq.heappush(pq, (nd, nxt))

        return best
