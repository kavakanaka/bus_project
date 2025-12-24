import numpy as np

from core.instance import Instance
from core.route import RouteSet
from core.transit_graph import TransitGraph


class Evaluator:
    def __init__(
        self,
        instance: Instance,
        transfer_penalty: float = 5.0,
        unreachable_penalty: float = 1e4,
    ):
        self.instance = instance
        self.transfer_penalty = transfer_penalty
        self.unreachable_penalty = unreachable_penalty

        # Map (u, v) -> travel_time
        self.edge_time = {
            (e.u, e.v): e.travel_time for e in instance.edges
        }

    def route_travel_time(self, route) -> float:
        """
        Total travel time of a single route.
        """
        t = 0.0
        for i in range(len(route.stops) - 1):
            u = route.stops[i]
            v = route.stops[i + 1]
            t += self.edge_time[(u, v)]
        return t

    def total_route_time(self, route_set: RouteSet) -> float:
        """
        TRT: sum of travel times of all routes.
        """
        return sum(self.route_travel_time(r) for r in route_set.routes)

    def average_travel_time(self, route_set: RouteSet) -> float:
        """
        ATT: demand-weighted average travel time.
        """
        tg = TransitGraph(
            self.instance,
            route_set,
            transfer_penalty=self.transfer_penalty,
        )

        total_time = 0.0
        total_demand = 0.0

        for o in range(1, self.instance.n_stops + 1):
            for d in range(1, self.instance.n_stops + 1):
                q = self.instance.demand[o - 1, d - 1]
                if q <= 0 or o == d:
                    continue

                t = tg.shortest_path(o, d)
                if t == float("inf"):
                    t = self.unreachable_penalty

                total_time += q * t
                total_demand += q

        if total_demand == 0:
            return float("inf")

        return total_time / total_demand
