import json
import random
import math

def random_point(bound=1000):
    return [random.randint(-bound, bound), random.randint(-bound, bound)]

def clustered_points(center, radius, count):
    cx, cy = center
    pts = []
    for _ in range(count):
        angle = random.random() * 2 * math.pi
        r = random.random() * radius
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        x = max(-1000, min(1000, x))
        y = max(-1000, min(1000, y))
        pts.append([x, y])
    return pts

def generate_instance(type_kind):
    if type_kind == "challenging_arc":
        start = random_point()
        end = random_point()
        cx = (start[0] + end[0]) / 2
        cy = (start[1] + end[1]) / 2
        radius = random.randint(300, 900)
        goals = []
        for i in range(21):
            angle = math.pi * i / 20
            x = int(cx + radius * math.cos(angle) + random.randint(-50, 50))
            y = int(cy + radius * math.sin(angle) + random.randint(-50, 50))
            goals.append([max(-1000, min(1000, x)), max(-1000, min(1000, y))])
        return {"start_points": [start], "end_points": [end], "goal_points": goals}
    elif type_kind == "dense_cluster":
        start = random_point()
        end = random_point()
        center = random_point(bound=500)
        goals = clustered_points(center, radius=200, count=80)
        return {"start_points": [start], "end_points": [end], "goal_points": goals}
    elif type_kind == "sparse_spread":
        start = random_point()
        end = random_point()
        goals = [random_point() for _ in range(50)]
        return {"start_points": [start], "end_points": [end], "goal_points": goals}
    elif type_kind == "grid_pattern":
        start = random_point(bound=500)
        end = random_point(bound=500)
        goals = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * 80 + random.randint(-20, 20)
                y = j * 80 + random.randint(-20, 20)
                goals.append([max(-1000, min(1000, x)), max(-1000, min(1000, y))])
        random.shuffle(goals)
        return {"start_points": [start], "end_points": [end], "goal_points": goals}
    elif type_kind == "line_pattern":
        start = random_point()
        end = random_point()
        goals = []
        for t in [i / 20 for i in range(1, 20)]:
            x = int(start[0] + t * (end[0] - start[0]) + random.randint(-50, 50))
            y = int(start[1] + t * (end[1] - start[1]) + random.randint(-50, 50))
            x = max(-1000, min(1000, x))
            y = max(-1000, min(1000, y))
            goals.append([x, y])
        return {"start_points": [start], "end_points": [end], "goal_points": goals}
    else:
        start = random_point()
        end = random_point()
        goals = [random_point() for _ in range(random.randint(20, 70))]
        return {"start_points": [start], "end_points": [end], "goal_points": goals}

def main():
    random.seed(12345)
    instances = {}
    types = ["challenging_arc", "dense_cluster", "grid_pattern", "line_pattern", "sparse_spread"]
    # 20 challenging
    for i in range(1, 51):
        name = f"instance_challenging_{i:03d}"
        t = random.choice(types)
        instances[name] = generate_instance(t)
    # 80 mixed
    all_types = types + ["default_random"]
    for i in range(21, 151):
        name = f"instance_mix_{i:03d}"
        t = random.choice(all_types)
        instances[name] = generate_instance(t)
    with open("./generate_test_instances/instances_harder_100.json", "w") as f:
        json.dump(instances, f, indent=2)

if __name__ == "__main__":
    main()
