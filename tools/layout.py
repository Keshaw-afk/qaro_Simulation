"""Random arena layouts: obstacles, robot starts and goals, with a no-softlock guarantee.

Used by both the Gazebo world generator and the 2D training environment, so training and testing see
the same distribution of worlds. Arena is centred on the origin: x in [-W/2, W/2], y in [-H/2, H/2].

No-softlock guarantee: obstacles and walls are inflated by the robot radius plus a margin; every start
and every goal must lie in one connected free region of that inflated map. So each robot has a
collision-free path to its goal, and no start/goal sits in a pocket or a gap too narrow to pass.
"""
import math
import os
from dataclasses import dataclass, field

import numpy as np
import yaml
from scipy import ndimage

# SNR_ARENA=big selects the large arena (8 x 6 m, 4x the area, more obstacles, higher camera); default is the 4 x 3 m training arena.
BIG = os.environ.get("SNR_ARENA") == "big"
ARENA_W, ARENA_H = (8.0, 6.0) if BIG else (4.0, 3.0)
ROBOT_RADIUS = 0.13          # chassis radius 0.095 plus the lidar offset, rounded up
CLEARANCE_MARGIN = 0.05
GRID_RES = 0.02
N_ROBOTS = 4
OBSTACLE_SIDE = (0.15, 0.60)  # random side lengths, metres
OBSTACLE_COUNT = (12, 32) if BIG else (3, 8)  # same obstacle density in both arenas
MIN_START_SEP, MIN_GOAL_SEP = 0.5, 0.5
MIN_START_GOAL = 2.0 if BIG else 1.0  # longer trips in the big arena
WALL_MARGIN = 0.25            # starts/goals keep this far from the walls
CAM_Z, OBSTACLE_H, TAG_HALF = (5.0 if BIG else 2.5), 0.4, 0.1  # overhead camera height (higher in the big arena), obstacle height, goal-tag half size


@dataclass
class Layout:
    seed: int
    obstacles: list = field(default_factory=list)  # (x, y, sx, sy, yaw)
    starts: list = field(default_factory=list)     # (x, y, yaw)
    goals: list = field(default_factory=list)      # (x, y)
    arena: tuple = (ARENA_W, ARENA_H)


def _box_dist(px, py, ob):
    """Distance from points to the (rotated) rectangle ob; 0 inside it."""
    x, y, sx, sy, yaw = ob
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = px - x, py - y
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return np.hypot(np.maximum(np.abs(lx) - sx / 2, 0), np.maximum(np.abs(ly) - sy / 2, 0))


def _grid_points(arena):
    w, h = arena
    xs = np.arange(-w / 2 + GRID_RES / 2, w / 2, GRID_RES)
    ys = np.arange(-h / 2 + GRID_RES / 2, h / 2, GRID_RES)
    return np.meshgrid(xs, ys, indexing="xy")  # arrays shaped (len(ys), len(xs))


def blocked_map(layout, inflate):
    """Boolean map: True where a disc of radius `inflate` centred there would touch an obstacle or wall."""
    gx, gy = _grid_points(layout.arena)
    w, h = layout.arena
    blocked = (np.abs(gx) > w / 2 - inflate) | (np.abs(gy) > h / 2 - inflate)
    for ob in layout.obstacles:
        blocked |= _box_dist(gx, gy, ob) <= inflate
    return blocked


def _cell(layout, x, y):
    w, h = layout.arena
    return int((y + h / 2) / GRID_RES), int((x + w / 2) / GRID_RES)


def validate(layout):
    """True if every start and goal is free (with clearance) and all lie in one connected region."""
    blocked = blocked_map(layout, ROBOT_RADIUS + CLEARANCE_MARGIN)
    labels, _ = ndimage.label(~blocked)
    pts = [(x, y) for x, y, _ in layout.starts] + list(layout.goals)
    comps = set()
    for x, y in pts:
        r, c = _cell(layout, x, y)
        if blocked[r, c]:
            return False
        comps.add(labels[r, c])
    return len(comps) == 1


def goal_visible(lay, g):
    """True if the overhead camera (above the arena centre) has a clear line of sight to the goal tag at floor level:
    tall obstacles seen at an angle hide a strip of floor behind them."""
    for frac in np.linspace(1 - OBSTACLE_H / CAM_Z, 0.999, 25):  # points on the camera-to-goal ray that are below obstacle-top height
        if any(_box_dist(np.array(frac * g[0]), np.array(frac * g[1]), ob) <= 0.0 for ob in lay.obstacles):
            return False
    # tag footprint at floor level must not lie inside an obstacle's shadow either: test the tag's four corner rays
    for dx, dy in ((TAG_HALF, TAG_HALF), (-TAG_HALF, TAG_HALF), (TAG_HALF, -TAG_HALF), (-TAG_HALF, -TAG_HALF)):
        for frac in np.linspace(1 - OBSTACLE_H / CAM_Z, 0.999, 20):
            px, py = frac * (g[0] + dx), frac * (g[1] + dy)
            if any(_box_dist(np.array(px), np.array(py), ob) <= 0.0 for ob in lay.obstacles):
                return False
    return True


def _sample_points(rng, layout, n, min_sep, avoid=()):
    w, h = layout.arena
    blocked = blocked_map(layout, ROBOT_RADIUS + CLEARANCE_MARGIN)
    pts = []
    for _ in range(2000):
        x = rng.uniform(-w / 2 + WALL_MARGIN, w / 2 - WALL_MARGIN)
        y = rng.uniform(-h / 2 + WALL_MARGIN, h / 2 - WALL_MARGIN)
        r, c = _cell(layout, x, y)
        if blocked[r, c]:
            continue
        if any(math.hypot(x - px, y - py) < min_sep for px, py in pts):
            continue
        if any(math.hypot(x - px, y - py) < d for (px, py), d in avoid):
            continue
        pts.append((x, y))
        if len(pts) == n:
            return pts
    return None


def generate_layout(seed, n_obstacles=None):
    """Return a validated random Layout. Deterministic for a given seed (and n_obstacles)."""
    rng = np.random.default_rng(seed)
    for _ in range(500):
        k = n_obstacles if n_obstacles is not None else int(rng.integers(OBSTACLE_COUNT[0], OBSTACLE_COUNT[1] + 1))
        lay = Layout(seed=seed)
        for _ in range(k):
            lay.obstacles.append((
                float(rng.uniform(-ARENA_W / 2 + 0.3, ARENA_W / 2 - 0.3)),
                float(rng.uniform(-ARENA_H / 2 + 0.3, ARENA_H / 2 - 0.3)),
                float(rng.uniform(*OBSTACLE_SIDE)), float(rng.uniform(*OBSTACLE_SIDE)),
                float(rng.uniform(0, math.pi))))
        starts = _sample_points(rng, lay, N_ROBOTS, MIN_START_SEP)
        if starts is None:
            continue
        goals = []
        ok = True
        for i in range(N_ROBOTS):
            avoid = [(starts[i], MIN_START_GOAL)] + [(s, 0.3) for j, s in enumerate(starts) if j != i]
            avoid += [(g, MIN_GOAL_SEP) for g in goals]
            g = _sample_points(rng, lay, 1, 0.0, avoid)
            if g is None:
                ok = False
                break
            goals.append(g[0])
        if not ok:
            continue
        lay.starts = [(x, y, float(rng.uniform(-math.pi, math.pi))) for x, y in starts]
        lay.goals = goals
        if validate(lay) and all(goal_visible(lay, g) for g in goals):
            return lay
    raise RuntimeError(f"could not generate a valid layout for seed {seed}")


def robot_radius_from_config():
    cfg = yaml.safe_load(open(__file__.replace("tools/layout.py", "config/robot.yaml")))
    return cfg["chassis_radius"] + abs(cfg["lidar_x"])


if __name__ == "__main__":
    print(f"ROBOT_RADIUS used {ROBOT_RADIUS}, from config {robot_radius_from_config():.3f}")
