## Imports 

import json
import csv
import time
import math
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Import solver functions from solve.py

from solve import (
    greedy_solve,
    beam_solve,
    simulated_annealing_solve,
    local_search_solve,
    hybrid_solve,
    path_length as compute_length
)

# Constraint validation

def validate_path(path, start_points, end_points, goal_points):
    """Check if a path satisfies all constraints."""
    if not path:
        return False
    # Start and end pointvalidity
    if tuple(path[0]) not in start_points: 
        return False
    if tuple(path[-1]) not in end_points: 
        return False
    # Point limit
    if len(path) > 100:
        return False
    # Length constraint
    if compute_length(path) >= 2000:
        return False
    # Coordinate bounds
    for x, y in path:
        if not (-1000 <= x <= 1000 and -1000 <= y <= 1000):
            return False
    # No repeats
    if len(path) != len(set(path)):
        return False
    return True

# Parallelise solver

def run_instance(name, data, method):
    """Run one instance with one method and return metrics."""
    start_pts = [tuple(p) for p in data["start_points"]]
    end_pts   = [tuple(p) for p in data["end_points"]]
    goal_pts  = [tuple(p) for p in data["goal_points"]]

    if method == "greedy":
        solver = greedy_solve
    elif method == "beam":
        solver = beam_solve
    elif method == "annealing":
        solver = simulated_annealing_solve
    elif method == "local":
        solver = local_search_solve
    elif method == "hybrid":
        solver = hybrid_solve
    else:
        raise ValueError(f"Unknown method {method}")

    start_time = time.time()
    try:
        path = solver(start_pts, end_pts, goal_pts)
    except Exception as e:
        path = []
    runtime = time.time() - start_time

    # Metrics
    
    goal_set = set(goal_pts)
    goals_visited = sum(1 for p in path if p in goal_set)
    length = compute_length(path) if path else 0.0
    num_points = len(path)
    valid = validate_path(path, start_pts, end_pts, goal_pts)

    return {
        "instance": name,
        "method": method,
        "goals_visited": goals_visited,
        "path_length": round(length, 3),
        "num_points": num_points,
        "runtime": round(runtime, 4),
        "valid": int(valid)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate route construction methods.")
    parser.add_argument("--input", required=True, help="Path to instances.json")
    parser.add_argument("--output", required=True, help="Path to results.csv")
    parser.add_argument("--methods", nargs="+", default=["greedy","beam","annealing","local","hybrid"],
                        help="List of methods to run")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Timeout per instance (seconds)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # Load input instances
    with open(args.input, "r") as f:
        instances = json.load(f)

    methods = args.methods
    results = []

    workers = cpu_count()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for name, data in instances.items():
            for method in methods:
                futures[executor.submit(run_instance, name, data, method)] = (name, method)

        for future in as_completed(futures):
            name, method = futures[future]
            try:
                res = future.result(timeout=args.timeout)
            except Exception as e:
                if args.verbose:
                    print(f"[{name}-{method}] timed out or failed: {e}")
                res = {
                    "instance": name,
                    "method": method,
                    "goals_visited": 0,
                    "path_length": 0.0,
                    "num_points": 0,
                    "runtime": args.timeout if args.timeout else 0.0,
                    "valid": 0
                }
            results.append(res)
            if args.verbose:
                print(f"{name}-{method}: goals={res['goals_visited']}, "
                      f"length={res['path_length']}, points={res['num_points']}, "
                      f"time={res['runtime']}s, valid={res['valid']}")

    # Write results to CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance","method","goals_visited","path_length","num_points","runtime","valid"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()
