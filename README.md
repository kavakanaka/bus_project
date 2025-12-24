# Transit Network Design Project (TNDP)

This repository contains a course project for the subject **Mathematical Modeling of Transport Flows**.  
The project studies the **Transit Network Design Problem (TNDP)** using heuristic and multi-objective optimization methods.

Author: **Nikita Khanzhin**

---

## Problem Description

The **Transit Network Design Problem (TNDP)** aims to design a set of public transport routes over a given transport network in order to balance:

- passenger service quality,
- operational cost of the transport system.

The problem is formulated as a **multi-objective optimization task** with the following objectives:

- **ATT (Average Travel Time)** — average passenger travel time,
- **TRT (Total Route Time)** — total length (or cost) of the route network.

The project follows approaches commonly used in the TNDP literature and reproduces qualitative results on a benchmark network.

---

## Methods Implemented

The solution pipeline consists of three main stages:

1. **Candidate route generation**
   - K-shortest paths using **Yen’s algorithm**
   - Applied to OD pairs with the highest demand

2. **Passenger assignment**
   - Construction of a transit graph
   - Shortest-path assignment using **Dijkstra’s algorithm**
   - Penalty for uncovered OD demand

3. **Route set optimization**
   - **Greedy heuristic** (weighted-sum scalarization)
   - **NSGA-II** multi-objective genetic algorithm
   - Approximation of the Pareto front (ATT vs TRT)

---

## Project Structure

```text
bus_project/
│
├── core/
│   ├── instance.py        # Transport network and demand representation
│   ├── route.py           # Route and route set abstractions
│   └── evaluator.py       # ATT / TRT evaluation and passenger assignment
│
├── generation/
│   └── k_shortest.py      # Yen’s K-shortest paths algorithm
│
├── optimization/
│   ├── greedy.py          # Greedy route selection heuristic
│   └── nsga2.py           # NSGA-II multi-objective optimizer
│
├── experiments/
│   ├── mandl_experiment.py  # Main experiment script
│   └── plots.py             # Visualization utilities
│
├── data/
│   └── raw/
│       ├── mandl_nodes.csv
│       ├── mandl_links.csv
│       └── mandl_demand.csv
│
├── requirements.txt
└── README.md
