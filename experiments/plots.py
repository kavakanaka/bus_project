import matplotlib.pyplot as plt

from core.evaluator import Evaluator
from optimization.greedy import GreedyOptimizer


def plot_att_vs_routes(instance, candidates, lambda_trt=0.1, max_k=6):
    evaluator = Evaluator(instance, transfer_penalty=5.0)

    route_counts = []
    att_values = []

    for k in range(1, max_k + 1):
        optimizer = GreedyOptimizer(
            evaluator,
            lambda_trt=lambda_trt,
            max_routes=k,
        )
        solution = optimizer.solve(candidates)

        route_counts.append(k)
        att_values.append(evaluator.average_travel_time(solution))

    plt.figure()
    plt.scatter(route_counts, att_values)
    plt.xlabel("Number of routes")
    plt.ylabel("Average Travel Time (ATT)")
    plt.title("ATT vs Number of Routes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("att_vs_routes.png", dpi=200)
    plt.show()


def plot_att_vs_trt(instance, candidates, lambda_trt=0.1, max_k=6):
    evaluator = Evaluator(instance, transfer_penalty=5.0)

    atts = []
    trts = []

    for k in range(1, max_k + 1):
        optimizer = GreedyOptimizer(
            evaluator,
            lambda_trt=lambda_trt,
            max_routes=k,
        )
        solution = optimizer.solve(candidates)

        atts.append(evaluator.average_travel_time(solution))
        trts.append(evaluator.total_route_time(solution))

    plt.figure()
    plt.scatter(trts, atts)
    for i, k in enumerate(range(1, max_k + 1)):
        plt.annotate(
            f"k={k}",
            (trts[i], atts[i]),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            va="bottom",
        )


    plt.xlabel("Total Route Time (TRT)")
    plt.ylabel("Average Travel Time (ATT)")
    plt.title("ATT vs TRT trade-off")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("att_vs_trt.png", dpi=200)
    plt.show()

def plot_pareto_front(pareto, greedy_solution=None):
    """
    pareto: list[Individual] from NSGA-II
    greedy_solution: RouteSet or None
    """

    atts = [ind.f1_att for ind in pareto]
    trts = [ind.f2_trt for ind in pareto]

    plt.figure()
    plt.scatter(trts, atts, label="NSGA-II Pareto", s=40)

    for ind in pareto:
        plt.annotate(
            str(len(ind.routes)),
            (ind.f2_trt, ind.f1_att),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    if greedy_solution is not None:
        plt.scatter(
            greedy_solution.trt,
            greedy_solution.att,
            color="red",
            marker="x",
            s=80,
            label="Greedy",
        )

    plt.xlabel("Total Route Time (TRT)")
    plt.ylabel("Average Travel Time (ATT)")
    plt.title("Pareto front: NSGA-II")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

