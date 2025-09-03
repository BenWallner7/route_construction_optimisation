import json
import math
import argparse
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Solve Route Construction Problem using various heuristics"
    )
    parser.add_argument('--input', required=True, help="Path to input JSON file with instances")
    parser.add_argument('--output', required=True, help="Path to output JSON file for solutions")
    parser.add_argument('--method', required=True,
                        choices=['greedy', 'beam', 'annealing', 'local', 'hybrid'],
                        help="Solving method (greedy, beam, annealing, local, hybrid)")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose logging per instance")
    parser.add_argument('--timeout', type=float, default=None,
                        help="Timeout (seconds) per instance")
    return parser.parse_args()

def euclidean(p1, p2):
    """Euclidean distance between points p1 and p2."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def path_length(path):
    """Total length of a path (sum of Euclidean segments)."""
    length = 0.0
    for i in range(len(path) - 1):
        length += euclidean(path[i], path[i+1])
    return length

def greedy_solve(start_points, end_points, goal_points):
    """
    Enhanced greedy: start at a chosen start point and always go to the nearest unvisited goal
    that does not violate length/end constraints. Then finish at the nearest end point.
    """
    current = start_points[0]  # pick the first start point
    path = [current]
    visited = set()
    total_len = 0.0

    # Iteratively pick nearest goal
    while True:
        next_goal = None
        min_dist = float('inf')
        for g in goal_points:
            if g in visited:
                continue
            d = euclidean(current, g)
            # Check if adding this goal still allows reaching an end within limits
            nearest_end_dist = min(euclidean(g, e) for e in end_points)
            if total_len + d + nearest_end_dist < 2000 and d < min_dist:
                min_dist = d
                next_goal = g
        if next_goal is None:
            break
        # Add this goal
        path.append(next_goal)
        visited.add(next_goal)
        total_len += min_dist
        current = next_goal
        if len(path) >= 100:
            break  # respect point limit

    # Finally append the nearest end point
    nearest_end = min(end_points, key=lambda e: euclidean(current, e))
    path.append(nearest_end)
    return path

def beam_solve(start_points, end_points, goal_points, beam_width=3):
    """
    Beam search: expand a fixed number of best partial paths at each step
    Each beam state is (path, visited_goals, length). We keep the top W beams sorted
    by (goals_visited desc, length asc).
    """
    # Initialize beams from all start points
    beams = [([s], set(), 0.0) for s in start_points]
    best_path = None

    # Expand in levels up to number of goals
    for _ in range(len(goal_points)):
        candidates = []
        for path, visited, length in beams:
            last = path[-1]
            # Try adding each unvisited goal
            for g in goal_points:
                if g in visited:
                    continue
                d = euclidean(last, g)
                nearest_end_dist = min(euclidean(g, e) for e in end_points)
                if length + d + nearest_end_dist < 2000:
                    new_path = path + [g]
                    new_visited = visited.union({g})
                    new_length = length + d
                    candidates.append((new_path, new_visited, new_length))
        if not candidates:
            break
        # Sort by (goals count desc, length asc)
        candidates.sort(key=lambda x: (-len(x[1]), x[2]))
        beams = candidates[:beam_width]
        # Track best by goals, then by shortest length
        for path, visited, length in beams:
            if (best_path is None) or (len(visited) > len(best_path[1])) \
               or (len(visited) == len(best_path[1]) and length < best_path[2]):
                best_path = (path, visited, length)

    # If no expansions, fallback to greedy
    if not beams:
        return greedy_solve(start_points, end_points, goal_points)
    # Choose the best beam and append closest end
    path, visited, length = best_path if best_path else beams[0]
    end_choice = min(end_points, key=lambda e: euclidean(path[-1], e))
    path.append(end_choice)
    return path

def simulated_annealing_solve(start_points, end_points, goal_points,
                              max_iter=500, init_temp=1000.0, cooling_rate=0.995):
    """
    Simulated annealing: start from a greedy solution and perform random local changes,
    accepting worse solutions with decreasing probability.
    We encode objectives so that higher goal count and shorter length are preferred.
    """
    # Initial solution (strip end to operate on goals only)
    full_path = greedy_solve(start_points, end_points, goal_points)
    end_pt = full_path[-1]
    path = full_path[:-1]  # drop final end
    current_score = len(path)
    best_path = list(path)
    best_score = current_score
    length_curr = path_length(path + [end_pt])
    temperature = init_temp

    for _ in range(max_iter):
        # Cool down
        temperature *= cooling_rate
        if temperature < 1e-3:
            break
        # Generate a neighbor: either 2-opt swap or add a random new goal
        new_path = list(path)
        if len(new_path) > 1 and random.random() < 0.5:
            i, j = sorted(random.sample(range(1, len(new_path)), 2))
            new_path[i:j] = reversed(new_path[i:j])
        else:
            # try inserting a new goal
            remaining = [g for g in goal_points if g not in new_path]
            if remaining:
                if random.random() < 0.5 and new_path:
                    idx = random.randrange(len(new_path))
                    new_goal = random.choice(remaining)
                    new_path.insert(idx, new_goal)
                else:
                    new_goal = random.choice(remaining)
                    new_path.append(new_goal)
        # Compute new score
        new_score = len(new_path)
        new_length = path_length(new_path + [end_pt]) if new_path else float('inf')
        # Weighted difference: prioritize goals (×1000) then length
        delta = (new_score - current_score) * 1000 + (length_curr - new_length)
        # Accept condition
        if (delta > 0) or (math.exp(delta / temperature) > random.random()):
            path = new_path
            current_score = new_score
            length_curr = new_length
            if (new_score > best_score) or (new_score == best_score and new_length < path_length(best_path + [end_pt])):
                best_path = list(new_path)
                best_score = new_score

    final = best_path + [end_pt]
    return final

def local_search_solve(start_points, end_points, goal_points):
    """
    Local 2-opt search: start from greedy solution and iteratively apply 2-opt swaps
    to shorten the path:contentReference[oaicite:11]{index=11}. Repeat until no improvement.
    """
    path = greedy_solve(start_points, end_points, goal_points)
    best = list(path)
    improved = True
    while improved:
        improved = False
        n = len(best)
        # Try all pairs of edges (i, j)
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                new_path = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if path_length(new_path) < path_length(best):
                    best = new_path
                    improved = True
                    # Once improved, restart search
        # continue until no more improvements
    return best

def hybrid_solve(start_points, end_points, goal_points):
    """
    Hybrid: run greedy then improve by local 2-opt search.
    """
    path = greedy_solve(start_points, end_points, goal_points)
    best = list(path)
    improved = True
    while improved:
        improved = False
        n = len(best)
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                new_path = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if path_length(new_path) < path_length(best):
                    best = new_path
                    improved = True
    return best

def solve_instance(name, data, method, verbose=False):
    """
    Solve one instance using the selected method. Returns (path, time).
    """
    start_pts = [tuple(pt) for pt in data['start_points']]
    end_pts = [tuple(pt) for pt in data['end_points']]
    goal_pts = [tuple(pt) for pt in data['goal_points']]

    # Select solver function
    if method == 'greedy':
        solver = greedy_solve
    elif method == 'beam':
        solver = beam_solve
    elif method == 'annealing':
        solver = simulated_annealing_solve
    elif method == 'local':
        solver = local_search_solve
    elif method == 'hybrid':
        solver = hybrid_solve
    else:
        raise ValueError(f"Unknown method: {method}")

    start_time = time.time()
    try:
        path = solver(start_pts, end_pts, goal_pts)
    except Exception:
        # In case of error, fallback to greedy
        path = greedy_solve(start_pts, end_pts, goal_pts)
    elapsed = time.time() - start_time

    if verbose:
        print(f"[{name}] method={method}, time={elapsed:.2f}s")
    return path, elapsed

def main():
    args = parse_arguments()
    # Load input JSON
    with open(args.input, 'r') as infile:
        instances = json.load(infile)

    solutions = {}
    total_goals = 0
    total_instances = len(instances)
    total_time = 0.0

    workers = cpu_count()
    if workers < 1:
        workers = 1

    start_all = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(solve_instance, name, inst, args.method, args.verbose): name
            for name, inst in instances.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                path, used_time = future.result(timeout=args.timeout)
            except Exception:
                if args.verbose:
                    print(f"[{name}] Timed out or error (timeout={args.timeout}s). Using fallback.")
                inst = instances[name]
                start_pt = tuple(inst['start_points'][0])
                end_pt = tuple(inst['end_points'][0])
                path = [start_pt, end_pt]
                used_time = args.timeout if args.timeout else 0.0

            # Save solution
            sol_list = [list(pt) for pt in path]
            solutions[name] = sol_list

            # Metrics
            goal_set = {tuple(pt) for pt in instances[name]['goal_points']}
            goals_visited = sum(1 for pt in path if pt in goal_set)
            total_goals += goals_visited
            total_time += used_time

            length = path_length(path)
            num_points = len(path)
            print(f"{name}: goals={goals_visited}, length={length:.2f}, "
                  f"points={num_points}, time={used_time:.2f}s")

    # Write solutions to json file
    with open(args.output, 'w') as outfile:
        json.dump(solutions, outfile, indent=2)

    # Overall summary
    total_runtime = time.time() - start_all
    print(f"\nSolution Summary:")
    print(f"Total goal points visited: {total_goals}")
    print(f"Total instances solved: {total_instances}")
    print(f"Average goals per instance: {total_goals/total_instances:.2f}")
    print(f"Total runtime (wall time): {total_runtime:.2f}s")
    print(f"Average time per instance: {total_time/total_instances:.3f}s")

if __name__ == "__main__":
    main()
