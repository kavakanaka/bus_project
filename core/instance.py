from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Edge:
    """
    Directed edge between two stops.
    """
    u: int              # from stop (1-based)
    v: int              # to stop (1-based)
    travel_time: float


@dataclass
class Instance:
    """
    Transport network instance:
    - stop graph
    - OD demand matrix
    """
    n_stops: int
    edges: list[Edge]
    demand: np.ndarray  # shape (n_stops, n_stops)

    def __post_init__(self):
        if self.demand.shape != (self.n_stops, self.n_stops):
            raise ValueError(
                "Demand matrix must have shape "
                f"({self.n_stops}, {self.n_stops})"
            )

    @property
    def n_edges(self) -> int:
        return len(self.edges)
