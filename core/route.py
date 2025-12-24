from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Route:
    """
    Public transport route represented as an ordered list of stops.
    """
    stops: List[int]

    def __post_init__(self):
        if len(self.stops) < 2:
            raise ValueError("Route must contain at least two stops")

    @property
    def length(self) -> int:
        """
        Number of stops in the route.
        """
        return len(self.stops)


@dataclass
class RouteSet:
    """
    A set of public transport routes.
    """
    routes: List[Route]

    def add(self, route: Route) -> None:
        self.routes.append(route)

    def remove(self, route: Route) -> None:
        self.routes.remove(route)

    def __len__(self) -> int:
        return len(self.routes)
