import json
import math
import argparse
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from itertools import combinations

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
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def path_length(path):
    return sum(euclidean(path[i], path[i+1]) for i in range(len(path)-1))

def segments_intersect(seg1, seg2):
    (p1, p2), (p3, p4) = seg1, seg2
    if p1 in seg2 or p2 in seg2:
        return False
    def ccw(A, B, C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def has_self_intersection(path):
    if len(path) < 4: return False
    segs = [(path[i], path[i+1]) for i in range(len(path)-1)]
    for i, j in combinations(range(len(segs)), 2):
        if j == i+1: continue
        if segments_intersect(segs[i], segs[j]): return True
    return False

def is_valid(path, goal_points, max_length=2000.0, max_points=100):
    if len(path) > max_points: return False
    if path_length(path) >= max_length: return False
    if len(path) != len(set(path)): return False
    if has_self_intersection(path): return False
    return True

def greedy_insertion_solve(start_points, end_points, goal_points):
    start = start_points[0]
    end = min(end_points, key=lambda e: euclidean(start, e))
    path = [start]
    visited, total_len = set(), 0.0

    while len(path) < 100 and len(visited) < len(goal_points):
        best_gain, best_g, best_pos = float('inf'), None, None
        for g in goal_points:
            if g in visited: continue
            for i in range(len(path)):
                a = path[i]
                b = path[i+1] if i+1 < len(path) else end
                gain = euclidean(a, g) + euclidean(g, b) - euclidean(a, b)
                new_len = total_len + gain + euclidean(g, end)
                if new_len < 2000 and gain < best_gain:
                    candidate = path[:i+1] + [g] + path[i+1:]
                    if not has_self_intersection(candidate):
                        best_gain, best_g, best_pos = gain, g, i+1
        if not best_g: break
        path.insert(best_pos, best_g)
        visited.add(best_g)
        total_len += best_gain

    path.append(end)
    return path if is_valid(path, goal_points) else [start, end]

def greedy_solve(start_points, end_points, goal_points):
    current = start_points[0]
    path = [current]
    visited = set()
    total_len = 0.0

    while len(path) < 100:
        next_goal, min_d = None, float('inf')
        for g in goal_points:
            if g in visited: continue
            d = euclidean(current, g)
            nearest_end = min(euclidean(g, e) for e in end_points)
            if total_len + d + nearest_end < 2000 and d < min_d:
                candidate_path = path + [g]
                if not has_self_intersection(candidate_path):
                    next_goal, min_d = g, d
        if not next_goal: break
        path.append(next_goal)
        visited.add(next_goal)
        total_len += min_d
        current = next_goal

    end = min(end_points, key=lambda e: euclidean(current, e))
    candidate = path + [end]
    return candidate if is_valid(candidate, goal_points) else [start_points[0], end]


def beam_with_distance_heuristic(start_points, end_points, goal_points, beam_width=100, alpha=0.001):
    beams = [([s], set(), 0.0) for s in start_points]
    best = None

    for _ in range(len(goal_points)):
        candidates = []
        for path, visited, length in beams:
            last = path[-1]
            for g in goal_points:
                if g in visited: continue
                d = euclidean(last, g)
                nearest_end = min(euclidean(g, e) for e in end_points)
                if length + d + nearest_end < 2000:
                    new = path + [g]
                    if has_self_intersection(new): continue
                    candidates.append((new, visited|{g}, length + d))

        if not candidates:
            break

        candidates.sort(key=lambda x: (
            -len(x[1]) - alpha * min(euclidean(x[0][-1], e) for e in end_points),
            x[2]
        ))
        beams = candidates[:beam_width]
        best = beams[0]

    path, visited, _ = best
    end = min(end_points, key=lambda e: euclidean(path[-1], e))
    candidate = path + [end]
    return candidate if is_valid(candidate, goal_points) else greedy_insertion_solve(start_points, end_points, goal_points)


# def beam_solve(start_points, end_points, goal_points, beam_width=3):
#     beams = [([s], set(), 0.0) for s in start_points]
#     best = None

#     for _ in range(len(goal_points)):
#         candidates = []
#         for path, visited, length in beams:
#             last = path[-1]
#             for g in goal_points:
#                 if g in visited: continue
#                 d = euclidean(last, g)
#                 nearest_end = min(euclidean(g, e) for e in end_points)
#                 if length + d + nearest_end < 2000:
#                     new = path + [g]
#                     if has_self_intersection(new): continue
#                     candidates.append((new, visited|{g}, length + d))
#         if not candidates: break
#         candidates.sort(key=lambda x: (-len(x[1]), x[2]))
#         beams = candidates[:beam_width]
#         best = max(beams, key=lambda x: (len(x[1]), -x[2])) if beams else best

#     if best:
#         path, visited, length = best
#         end = min(end_points, key=lambda e: euclidean(path[-1], e))
#         candidate = path + [end]
#         return candidate if is_valid(candidate, goal_points) else greedy_solve(start_points, end_points, goal_points)
#     return greedy_solve(start_points, end_points, goal_points)

def generate_neighbor_expanded(path, end, goal_points, attempts=20):
    for _ in range(attempts):
        new = path[:]
        r = random.random()
        if r < 0.3 and len(new) > 1:
            i, j = sorted(random.sample(range(1, len(new)), 2))
            new[i:j] = reversed(new[i:j])
        elif r < 0.6 and len(new) > 2:
            i, j, k = sorted(random.sample(range(1, len(new)), 3))
            new = new[:i] + new[j:k] + new[i:j] + new[k:]
        elif r < 0.9:
            candidates = [g for g in goal_points if g not in new]
            if not candidates: continue
            g = random.choice(candidates)
            best_gain, best_pos = float('inf'), None
            for idx in range(len(new)):
                a = new[idx]
                b = new[idx+1] if idx + 1 < len(new) else end
                gain = euclidean(a, g) + euclidean(g, b) - euclidean(a, b)
                if gain < best_gain:
                    best_gain, best_pos = gain, idx+1
            if best_pos is not None:
                new.insert(best_pos, g)
        else:
            if new:
                new.remove(random.choice(new))

        if is_valid(new + [end], goal_points):
            return new
    return None

def simulated_annealing_advanced(start_points, end_points, goal_points,
                                 max_iter=2000, init_temp=1000.0, rate=0.995):
    path = greedy_insertion_solve(start_points, end_points, goal_points)[:-1]
    end = greedy_insertion_solve(start_points, end_points, goal_points)[-1]
    current, best = path[:], path[:]
    current_score = best_score = len(path)
    current_len = best_len = path_length(path + [end])

    temp = init_temp
    for _ in range(max_iter):
        temp *= rate
        if temp < 1e-3: break
        neighbor = generate_neighbor_expanded(current, end, goal_points)
        if not neighbor: continue
        new_score = len(neighbor)
        new_len = path_length(neighbor + [end])
        delta = 1000*(new_score - current_score) + (current_len - new_len)
        if delta > 0 or math.exp(delta / temp) > random.random():
            current, current_score, current_len = neighbor, new_score, new_len
            if new_score > best_score or (new_score == best_score and new_len < best_len):
                best, best_score, best_len = neighbor[:], new_score, new_len

    return best + [end]

# def simulated_annealing_solve(start_points, end_points, goal_points,
#                               max_iter=1000, temp0=1000.0, rate=0.995):
#     path = greedy_solve(start_points, end_points, goal_points)
#     end = path[-1]
#     path = path[:-1]
#     if not is_valid(path + [end], goal_points):
#         path = []

#     best = current = path[:]
#     best_score = current_score = len(current)
#     best_len = current_len = path_length(current + [end])

#     temp = temp0
#     for _ in range(max_iter):
#         temp *= rate
#         if temp < 1e-3: break
#         neighbor = generate_neighbor_expanded(current, end, goal_points)
#         if not neighbor: continue
#         new_score = len(neighbor)
#         new_len = path_length(neighbor + [end])
#         delta = 1000*(new_score - current_score) + (current_len - new_len)
#         if delta > 0 or math.exp(delta / temp) > random.random():
#             current, current_score, current_len = neighbor, new_score, new_len
#             if new_score > best_score or (new_score == best_score and new_len < best_len):
#                 best, best_score, best_len = neighbor[:], new_score, new_len

    return best + [end]

## Local with 3-Opt swap

def three_opt_swap(path, i, j, k):
    """Perform a 3-opt swap by reversing segments between i, j, and k."""
    new_path = path[:i+1] + path[j:k+1][::-1] + path[k+1:]
    return new_path

def local_search_solve(start_points, end_points, goal_points):
    path = greedy_solve(start_points, end_points, goal_points)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path)-3):
            for j in range(i+1, len(path)-2):
                for k in range(j+1, len(path)-1):
                    new_path = three_opt_swap(path, i, j, k)
                    if is_valid(new_path, goal_points) and path_length(new_path) < path_length(path):
                        path = new_path
                        improved = True
                        break
                if improved: break
            if improved: break
    return path


# def local_search_solve(start_points, end_points, goal_points):
#     path = greedy_solve(start_points, end_points, goal_points)
#     improved = True
#     while improved:
#         improved = False
#         for i in range(1, len(path)-2):
#             for j in range(i+1, len(path)-1):
#                 new = path[:i] + path[i:j+1][::-1] + path[j+1:]
#                 if is_valid(new, goal_points) and path_length(new) < path_length(path):
#                     path = new
#                     improved = True
#                     break
#             if improved: break
#     return path

def hybrid_solve(start_points, end_points, goal_points):
    greedy_path = greedy_insertion_solve(start_points, end_points, goal_points)
    local_path = local_search_solve(start_points, end_points, goal_points)
    sa_path = simulated_annealing_advanced(start_points, end_points, goal_points)
    
    # Select the best path based on length
    best_path = min([greedy_path, local_path, sa_path], key=lambda p: path_length(p))
    
    return best_path


def solve_instance(name, data, method, verbose=False):
    sp = [tuple(pt) for pt in data['start_points']]
    ep = [tuple(pt) for pt in data['end_points']]
    gp = [tuple(pt) for pt in data['goal_points']]

    solver = {
        'greedy': greedy_insertion_solve,
        'beam': beam_with_distance_heuristic,
        'annealing': simulated_annealing_advanced,
        'local': local_search_solve,
        'hybrid': hybrid_solve
    }[method]

    t0 = time.time()
    path = solver(sp, ep, gp)
    elapsed = time.time() - t0

    if verbose:
        print(f"[{name}] method={method}, time={elapsed:.2f}s")

    if not is_valid(path, gp):
        path = greedy_solve(sp, ep, gp)

    return path, elapsed

def main():
    args = parse_arguments()
    with open(args.input) as f:
        instances = json.load(f)

    solutions, total_goals, total_time = {}, 0, 0.0
    workers = max(1, cpu_count())

    start_all = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(solve_instance, name, inst, args.method, args.verbose): name
            for name, inst in instances.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                path, used = future.result(timeout=args.timeout)
            except Exception:
                inst = instances[name]
                sp = tuple(inst['start_points'][0])
                ep = tuple(inst['end_points'][0])
                path = [sp, ep]
                used = args.timeout or 0.0
                print(f"[{name}] fallback solution used.")

            solutions[name] = [list(pt) for pt in path]
            gp_set = {tuple(p) for p in instances[name]['goal_points']}
            goals = sum(pt in gp_set for pt in path)
            total_goals += goals
            total_time += used
            print(f"{name}: goals={goals}, length={path_length(path):.2f}, points={len(path)}, time={used:.2f}s")

    with open(args.output, 'w') as out:
        json.dump(solutions, out, indent=2)

    total_runtime = time.time() - start_all
    print("\nSummary:")
    print(f" Total goals: {total_goals}")
    print(f" Instances: {len(instances)}")
    print(f" Avg goals/instance: {total_goals/len(instances):.2f}")
    print(f" Total wall time: {total_runtime:.2f}s")
    print(f" Avg per-instance time: {total_time/len(instances):.3f}s")

if __name__ == "__main__":
    main()