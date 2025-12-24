from typing import List, Callable

from core.route import Route, RouteSet
from core.evaluator import Evaluator


class GreedyOptimizer:
    def __init__(
        self,
        evaluator: Evaluator,
        lambda_trt: float = 0.1,
        max_routes: int = 10,
    ):
        self.evaluator = evaluator
        self.lambda_trt = lambda_trt
        self.max_routes = max_routes

    def objective(self, route_set: RouteSet) -> float:
        """
        Scalar objective: ATT + lambda * TRT
        """
        att = self.evaluator.average_travel_time(route_set)
        trt = self.evaluator.total_route_time(route_set)
        return att + self.lambda_trt * trt

    def solve(self, candidates: List[Route]) -> RouteSet:
        """
        Greedy selection of routes from candidate pool.
        """
        selected: List[Route] = []
        remaining = list(candidates)

        best_value = float("inf")

        while remaining and len(selected) < self.max_routes:
            best_route = None
            best_new_value = best_value

            for r in remaining:
                trial = RouteSet(selected + [r])
                val = self.objective(trial)

                if val < best_new_value:
                    best_new_value = val
                    best_route = r

            if best_route is None:
                break

            selected.append(best_route)
            remaining.remove(best_route)
            best_value = best_new_value

        return RouteSet(selected)
