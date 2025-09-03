##Imports 

import json
import argparse
import math
import time
from typing import List, Tuple, Dict, Set, Optional
from itertools import combinations


class RouteConstructionSolver:
    """
    Solver for the route construction optimization problem.
    
    Uses a greedy nearest-neighbor approach with path optimization
    and constraint validation.
    """
    
    def __init__(self, max_length: float = 2000.0, max_points: int = 100):
        self.max_length = max_length
        self.max_points = max_points
        
    def distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def path_length(self, path: List[Tuple[int, int]]) -> float:
        """Calculate total length of a path."""
        if len(path) < 2:
            return 0.0
        return sum(self.distance(path[i], path[i+1]) for i in range(len(path)-1))
    
    def segments_intersect(self, seg1: Tuple[Tuple[int, int], Tuple[int, int]], 
                          seg2: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """
        Check if two line segments intersect at interior points.
        Returns False if segments are adjacent (share an endpoint).
        """
        (p1, p2), (p3, p4) = seg1, seg2
        
        # Skip if segments share an endpoint (adjacent segments)
        if p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4:
            return False
            
        # Check if segments intersect using cross product method
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def has_self_intersection(self, path: List[Tuple[int, int]]) -> bool:
        """Check if path has any self-intersections."""
        if len(path) < 4:
            return False
            
        segments = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        # Check all non-adjacent segment pairs
        for i in range(len(segments)):
            for j in range(i+2, len(segments)):
                if self.segments_intersect(segments[i], segments[j]):
                    return True
        return False
    
    def is_valid_path(self, path: List[Tuple[int, int]], 
                     start_points: List[Tuple[int, int]], 
                     end_points: List[Tuple[int, int]]) -> bool:
        """Validate that path satisfies all constraints."""
        if len(path) == 0:
            return True
            
        # Check point limit
        if len(path) > self.max_points:
            return False
            
        # Check length constraint
        if self.path_length(path) >= self.max_length:
            return False
            
        # Check valid start and end points
        if len(path) >= 1 and tuple(path[0]) not in start_points:
            return False
        if len(path) >= 2 and tuple(path[-1]) not in end_points:
            return False
            
        # Check for self-intersection
        if self.has_self_intersection(path):
            return False
            
        # Check coordinate bounds
        for x, y in path:
            if not (-1000 <= x <= 1000 and -1000 <= y <= 1000):
                return False
                
        # Check for repeated coordinates
        if len(set(path)) != len(path):
            return False
            
        return True
    
    def greedy_solve(self, start_points: List[List[int]], 
                    end_points: List[List[int]], 
                    goal_points: List[List[int]]) -> List[List[int]]:
        """
        Solve using greedy nearest-neighbor approach.
        
        Args:
            start_points: List of valid starting coordinates
            end_points: List of valid ending coordinates  
            goal_points: List of goal coordinates to visit
            
        Returns:
            List of coordinates forming the optimal path
        """
        # Convert to tuples for easier handling
        start_pts = [tuple(p) for p in start_points]
        end_pts = [tuple(p) for p in end_points]
        goal_pts = [tuple(p) for p in goal_points]
        
        best_path = []
        best_goals = 0
        
        # Try each start point
        for start in start_pts:
            for end in end_pts:
                # Skip if start and end are the same
                if start == end:
                    continue
                    
                # Try to build path from start to end visiting goals
                path = self.build_path(start, end, goal_pts)
                
                # Check constraints, is path valid
                if self.is_valid_path(path, start_pts, end_pts):
                    # Count goal points visited
                    goals_visited = sum(1 for p in path if p in goal_pts)
                    
                    if goals_visited > best_goals:
                        best_goals = goals_visited
                        best_path = path
        
        # Convert back to list format
        return [list(p) for p in best_path]
    
    def build_path(self, start: Tuple[int, int], end: Tuple[int, int], 
                  goal_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Build path from start to end visiting as many goals as possible."""
        path = [start]
        visited_goals = set()
        remaining_goals = [g for g in goal_points if g != start and g != end]
        
        current = start
        
        while len(path) < self.max_points - 1:
            # Find nearest unvisited goal that doesn't violate constraints
            best_goal = None
            best_distance = float('inf')
            
            for goal in remaining_goals:
                if goal in visited_goals:
                    continue
                    
                # Check if adding this goal would exceed length limit
                dist_to_goal = self.distance(current, goal)
                dist_goal_to_end = self.distance(goal, end)
                
                if (self.path_length(path) + dist_to_goal + dist_goal_to_end 
                    < self.max_length):
                    
                    # Test path with this goal added
                    test_path = path + [goal]
                    if not self.has_self_intersection(test_path):
                        if dist_to_goal < best_distance:
                            best_distance = dist_to_goal
                            best_goal = goal
            
            if best_goal is None:
                break
                
            path.append(best_goal)
            visited_goals.add(best_goal)
            current = best_goal
        
        # Add end point if we can reach it
        if current != end:
            final_dist = self.distance(current, end)
            if (self.path_length(path) + final_dist < self.max_length and
                len(path) < self.max_points):
                test_path = path + [end]
                if not self.has_self_intersection(test_path):
                    path.append(end)
        
        return path
    
    def solve_instance(self, instance_data: Dict) -> List[List[int]]:
        """Solve a single problem instance."""
        start_points = instance_data['start_points']
        end_points = instance_data['end_points']
        goal_points = instance_data['goal_points']
        
        return self.greedy_solve(start_points, end_points, goal_points)


def main():
    """Main function to run the route construction solver."""
    
    # Arguments setup
    
    parser = argparse.ArgumentParser(description='Route Construction Problem Solver')
    parser.add_argument('--input', required=True, 
                       help='Path to input JSON file containing problem instances')
    parser.add_argument('--output', required=True,
                       help='Path where solution JSON file should be written')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load input data
    try:
        with open(args.input, 'r') as f:
            instances = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading input file: {e}")
        return 1
    
    # Initialize solver
    solver = RouteConstructionSolver()
    solutions = {}
    
    total_goals = 0
    total_instances = len(instances)
    start_time = time.time()
    
    # Solve each instance
    for i, (instance_name, instance_data) in enumerate(instances.items(), 1):
        if args.verbose:
            print(f"Solving {instance_name} ({i}/{total_instances})...")
            
        instance_start = time.time()
        solution = solver.solve_instance(instance_data)
        instance_time = time.time() - instance_start
        
        solutions[instance_name] = solution
        
        # Count goals visited
        goal_points_set = set(tuple(p) for p in instance_data['goal_points'])
        goals_visited = sum(1 for p in solution if tuple(p) in goal_points_set)
        total_goals += goals_visited
        
        if args.verbose:
            print(f"  Goals visited: {goals_visited}/{len(instance_data['goal_points'])}")
            print(f"  Path length: {len(solution)} points")
            if solution:
                path_dist = solver.path_length([tuple(p) for p in solution])
                print(f"  Total distance: {path_dist:.2f}")
            print(f"  Time: {instance_time:.3f}s")
    
    total_time = time.time() - start_time
    
    # Save solutions to json file
    try:
        with open(args.output, 'w') as f:
            json.dump(solutions, f, indent=2)
    # Exception handling
    except IOError as e:
        print(f"Error writing output file: {e}")
        return 1
    
    # Display summary of optimsation algorithm
    print(f"\nSolution Summary:")
    print(f"Total goal points visited: {total_goals}")
    print(f"Total instances solved: {total_instances}")
    print(f"Average goals per instance: {total_goals/total_instances:.2f}")
    print(f"Total runtime: {total_time:.2f}s")
    print(f"Average time per instance: {total_time/total_instances:.3f}s")
    
    return 0


if __name__ == "__main__":
    exit(main())