import csv
import numpy as np

from core.instance import Edge, Instance
from core.route import RouteSet
from core.evaluator import Evaluator
from generation.k_shortest import yen_k_shortest_paths
from optimization.greedy import GreedyOptimizer
from experiments.plots import plot_att_vs_routes, plot_att_vs_trt
from optimization.nsga2 import NSGA2Optimizer
from experiments.plots import plot_pareto_front
from experiments.plots import plot_att_trt_greedy_vs_nsga


def load_mandl_instance():
    edges = []
    max_node = 0

    # -------- links --------
    with open("data/raw/mandl_links.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row["from"])
            v = int(row["to"])
            t = float(row["travel_time"])
            edges.append(Edge(u, v, t))
            max_node = max(max_node, u, v)

    # -------- demand --------
    demand = np.zeros((max_node, max_node))
    with open("data/raw/mandl_demand.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            o = int(row["from"])
            d = int(row["to"])
            q = float(row["demand"])
            demand[o - 1, d - 1] += q

    return Instance(
        n_stops=max_node,
        edges=edges,
        demand=demand,
    )


def main():
    print("MAIN STARTED")

    print("Loading Mandl instance...")
    instance = load_mandl_instance()
    print("INSTANCE LOADED")

    print(f"Stops: {instance.n_stops}")
    print(f"Edges: {instance.n_edges}")


    print("Generating candidate routes...")

    pairs = []
    for o in range(1, instance.n_stops + 1):
        for d in range(1, instance.n_stops + 1):
            q = instance.demand[o - 1, d - 1]
            if q > 0 and o != d:
                pairs.append((q, o, d))

    pairs.sort(reverse=True)
    pairs = pairs[:30]

    candidates = []
    for q, o, d in pairs:
        print(f"  OD {o} -> {d} (demand={q})")
        routes = yen_k_shortest_paths(
            instance,
            source=o,
            target=d,
            k=3,
        )
        candidates.extend(routes)
 
    unique = {tuple(r.stops): r for r in candidates}
    candidates = list(unique.values())

    print(f"Candidate routes: {len(candidates)}")

    evaluator = Evaluator(instance, transfer_penalty=5.0)
    optimizer = GreedyOptimizer(
        evaluator,
        lambda_trt=0.1,
        max_routes=6,
    )

    # =========================
    # NSGA-II optimization
    # =========================
    print("\nRunning NSGA-II optimization...")

    nsga = NSGA2Optimizer(
        evaluator=evaluator,
        max_routes=10,
        pop_size=150,
        generations=200,
    )

    pareto = nsga.solve(candidates)

    plot_pareto_front(pareto)

    print("\n=== PARETO FRONT (approx) ===")
    for i, ind in enumerate(pareto[:10], 1):
        print(
            f"{i}: ATT={ind.f1_att:.3f}, "
            f"TRT={ind.f2_trt:.3f}, "
            f"#routes={len(ind.routes)}"
        )

    lambda_trt = 0.1
    best_nsga = min(
        pareto,
        key=lambda ind: ind.f1_att + lambda_trt * ind.f2_trt,
    )

    print("\n=== CHOSEN NSGA-II SOLUTION ===")
    print(
        f"ATT={best_nsga.f1_att:.3f}, "
        f"TRT={best_nsga.f2_trt:.3f}, "
        f"#routes={len(best_nsga.routes)}"
    )
    for j, r in enumerate(best_nsga.routes, 1):
        print(f"{j}: {r.stops}")

    


    print("Running greedy optimization...")
    solution = optimizer.solve(candidates)

    att = evaluator.average_travel_time(solution)
    trt = evaluator.total_route_time(solution)

    print("\n=== FINAL SOLUTION ===")
    print(f"Number of routes: {len(solution.routes)}")
    print(f"ATT: {att:.3f}")
    print(f"TRT: {trt:.3f}")
    print("Routes:")
    for i, r in enumerate(solution.routes, 1):
        print(f"{i}: {r.stops}")


    plot_att_vs_routes(instance, candidates)
    plot_att_vs_trt(instance, candidates)

    # =========================
    # Greedy: lambda sweep
    # =========================
    greedy_points = []

    lambdas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    for lam in lambdas:
        optimizer = GreedyOptimizer(
            evaluator,
            lambda_trt=lam,
            max_routes=6,
        )
        sol = optimizer.solve(candidates)

        att = evaluator.average_travel_time(sol)
        trt = evaluator.total_route_time(sol)

        greedy_points.append((lam, att, trt))
    plot_att_trt_greedy_vs_nsga(pareto, greedy_points)



if __name__ == "__main__":
    main()
