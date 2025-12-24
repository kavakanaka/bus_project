import random
from dataclasses import dataclass
from typing import List, Tuple

from core.route import Route, RouteSet
from core.evaluator import Evaluator


@dataclass
class Individual:
    routes: List[Route]
    f1_att: float | None = None
    f2_trt: float | None = None
    rank: int | None = None
    crowding: float = 0.0

    def as_routeset(self) -> RouteSet:
        return RouteSet(self.routes)


def dominates(a: Individual, b: Individual) -> bool:
    return (a.f1_att <= b.f1_att and a.f2_trt <= b.f2_trt) and (a.f1_att < b.f1_att or a.f2_trt < b.f2_trt)


def fast_nondominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    S = {id(p): [] for p in pop}
    n = {id(p): 0 for p in pop}
    fronts: List[List[Individual]] = [[]]

    for p in pop:
        for q in pop:
            if p is q:
                continue
            if dominates(p, q):
                S[id(p)].append(q)
            elif dominates(q, p):
                n[id(p)] += 1

        if n[id(p)] == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[id(p)]:
                n[id(q)] -= 1
                if n[id(q)] == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts


def crowding_distance(front: List[Individual]) -> None:
    if not front:
        return

    for p in front:
        p.crowding = 0.0

    front.sort(key=lambda x: x.f1_att)
    front[0].crowding = float("inf")
    front[-1].crowding = float("inf")
    fmin = front[0].f1_att
    fmax = front[-1].f1_att
    if fmax != fmin:
        for i in range(1, len(front) - 1):
            front[i].crowding += (front[i + 1].f1_att - front[i - 1].f1_att) / (fmax - fmin)

    front.sort(key=lambda x: x.f2_trt)
    front[0].crowding = float("inf")
    front[-1].crowding = float("inf")
    fmin = front[0].f2_trt
    fmax = front[-1].f2_trt
    if fmax != fmin:
        for i in range(1, len(front) - 1):
            front[i].crowding += (front[i + 1].f2_trt - front[i - 1].f2_trt) / (fmax - fmin)


def tournament_select(pop: List[Individual]) -> Individual:
    a = random.choice(pop)
    b = random.choice(pop)
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding > b.crowding else b


def repair_unique_routes(routes: List[Route]) -> List[Route]:
    uniq = {}
    for r in routes:
        uniq[tuple(r.stops)] = r
    return list(uniq.values())


def crossover(p1: Individual, p2: Individual, max_routes: int) -> Individual:
    a = p1.routes[:]
    b = p2.routes[:]
    random.shuffle(a)
    random.shuffle(b)

    cut_a = random.randint(0, min(len(a), max_routes))
    cut_b = random.randint(0, min(len(b), max_routes))

    child_routes = repair_unique_routes(a[:cut_a] + b[:cut_b])
    if len(child_routes) > max_routes:
        random.shuffle(child_routes)
        child_routes = child_routes[:max_routes]

    return Individual(routes=child_routes)


def mutate(ind: Individual, candidates: List[Route], max_routes: int, p_add=0.4, p_drop=0.3, p_swap=0.5) -> None:

    if ind.routes and random.random() < p_swap:
        idx = random.randrange(len(ind.routes))
        ind.routes[idx] = random.choice(candidates)

    if len(ind.routes) < max_routes and random.random() < p_add:
        ind.routes.append(random.choice(candidates))

    if ind.routes and random.random() < p_drop:
        idx = random.randrange(len(ind.routes))
        ind.routes.pop(idx)

    ind.routes = repair_unique_routes(ind.routes)
    if len(ind.routes) > max_routes:
        random.shuffle(ind.routes)
        ind.routes = ind.routes[:max_routes]


class NSGA2Optimizer:
    def __init__(
        self,
        evaluator: Evaluator,
        max_routes: int = 6,
        pop_size: int = 40,
        generations: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.4,
        seed: int = 42,
    ):
        self.evaluator = evaluator
        self.max_routes = max_routes
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        random.seed(seed)

    def evaluate(self, ind: Individual) -> None:
        rs = ind.as_routeset()
        ind.f1_att = self.evaluator.average_travel_time(rs)
        ind.f2_trt = self.evaluator.total_route_time(rs)

    def init_population(self, candidates: List[Route]) -> List[Individual]:
        pop = []
        for _ in range(self.pop_size):
            k = random.randint(1, self.max_routes)
            routes = random.sample(candidates, k=min(k, len(candidates)))
            pop.append(Individual(routes=repair_unique_routes(routes)))
        return pop

    def assign_rank_and_crowding(self, pop: List[Individual]) -> List[List[Individual]]:
        fronts = fast_nondominated_sort(pop)
        for f in fronts:
            crowding_distance(f)
        return fronts

    def make_offspring(self, pop: List[Individual], candidates: List[Route]) -> List[Individual]:
        offspring = []
        while len(offspring) < self.pop_size:
            p1 = tournament_select(pop)
            p2 = tournament_select(pop)

            if random.random() < self.crossover_rate:
                child = crossover(p1, p2, self.max_routes)
            else:
                child = Individual(routes=p1.routes[:])

            if random.random() < self.mutation_rate:
                mutate(child, candidates, self.max_routes)

            offspring.append(child)

        return offspring

    def select_next_generation(self, combined: List[Individual]) -> List[Individual]:
        fronts = self.assign_rank_and_crowding(combined)

        next_pop = []
        for f in fronts:
            if len(next_pop) + len(f) <= self.pop_size:
                next_pop.extend(f)
            else:
                f_sorted = sorted(f, key=lambda x: x.crowding, reverse=True)
                next_pop.extend(f_sorted[: self.pop_size - len(next_pop)])
                break

        return next_pop

    def solve(self, candidates: List[Route]) -> List[Individual]:
        pop = self.init_population(candidates)
        for ind in pop:
            self.evaluate(ind)

        pop = self.select_next_generation(pop)

        for _ in range(self.generations):
            offspring = self.make_offspring(pop, candidates)
            for ind in offspring:
                self.evaluate(ind)

            combined = pop + offspring
            pop = self.select_next_generation(combined)

        # return the final nondominated front (approx Pareto set)
        fronts = self.assign_rank_and_crowding(pop)
        pareto = fronts[0]
        pareto_sorted = sorted(pareto, key=lambda x: (x.f1_att, x.f2_trt))
        return pareto_sorted
