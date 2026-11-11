"""Hand-designed stress-test layouts (standard 4 x 3 m arena) in which the robots MUST cross each other's paths:
head-on swaps in corridors, a four-way intersection, a single door, a roundabout and a pillar forest. They test collision
avoidance between moving robots (dynamic obstacles) far harder than the random layouts. Documented in
docs/DESIGNED_LAYOUTS.md.

IDs 9001-9005 sit outside every random-seed range (evaluation 0-199, training/validation 1000-2999), and
tools/layout.py get_layout(seed) returns them in place of a random layout, so every seed-based tool works unchanged:
    bash tools/make_demo.sh 9003 se live                      # watch one (Gazebo)
    SNR_FIRST=9001 python3 rl/eval_gazebo.py weights/final_policy.npz 9006 0   # all five in Gazebo
    .venv/bin/python benchmark/eval_designed.py                # all benchmark controllers in 2D
Each design uses at most 8 boxes (the 2D pool's limit, rl/pool.py K_MAX) and passes the same checks as random layouts
(layout.validate: every start and goal connected with clearance; layout.goal_visible: goal tags seen by the camera).

    python3 tools/designed_layouts.py [OUT_DIR]   # check all designs, print path conflicts, draw top-down maps (PNG)"""
import math
import sys
from collections import deque

import numpy as np

import layout as L

PI = math.pi


def _wall_y(x, y0, y1, t=0.2):
    """Box covering x +- t/2, y0..y1."""
    return (x, (y0 + y1) / 2, t, y1 - y0, 0.0)


# id: (short name, what it tests, obstacles (x, y, sx, sy, yaw), starts (x, y, yaw), goals (x, y))
DESIGNS = {
    9001: ("crossroads",
           "Four big corner blocks leave a plus-shaped pair of corridors (0.9 m and 0.8 m wide). West and east robots swap "
           "ends head-on in the same corridor, north and south do the same, and all four reach the intersection together.",
           [(1.175, 0.95, 1.55, 1.0, 0), (-1.175, 0.95, 1.55, 1.0, 0), (1.175, -0.95, 1.55, 1.0, 0), (-1.175, -0.95, 1.55, 1.0, 0)],
           [(-1.7, 0.0, 0.0), (1.7, 0.0, PI), (0.0, 1.2, -PI / 2), (0.0, -1.2, PI / 2)],
           [(1.2, 0.0), (-1.2, 0.0), (0.0, -0.75), (0.0, 0.75)]),
    9002: ("roundabout",
           "Robots start in the four corners and go to the opposite side; every straight path runs through a central "
           "diamond pillar, so all four must go round it at once. Four side posts narrow the ring.",
           [(0.0, 0.0, 0.6, 0.6, PI / 4), (0.0, 1.05, 0.4, 0.3, 0), (0.0, -1.05, 0.4, 0.3, 0), (1.45, 0.0, 0.3, 0.5, 0),
            (-1.45, 0.0, 0.3, 0.5, 0)],
           [(-1.65, 1.15, -0.6), (1.65, 1.15, -PI + 0.6), (1.65, -1.15, PI - 0.6), (-1.65, -1.15, 0.6)],
           [(1.05, -0.7), (-1.05, -0.7), (-1.05, 0.7), (1.05, 0.7)]),
    9003: ("single door",
           "A wall splits the arena with one 0.6 m door in the middle. Two robots on each side must all change sides "
           "diagonally through the same door; two posts in front of the door force a curved approach.",
           [_wall_y(0.0, -1.45, -0.3), _wall_y(0.0, 0.3, 1.45), (-0.9, 0.0, 0.3, 0.3, PI / 4), (0.9, 0.0, 0.3, 0.3, PI / 4)],
           [(-1.6, 0.9, 0.0), (-1.6, -0.9, 0.0), (1.6, 0.9, PI), (1.6, -0.9, PI)],
           [(1.0, -0.9), (1.0, 0.9), (-1.0, -0.9), (-1.0, 0.9)]),
    9004: ("pillar forest",
           "Eight pillars in three staggered columns fill the whole arena height, so there is no way round. Two robots "
           "on each side swap sides diagonally, so the two streams cross each other inside the forest, in 0.5-0.65 m gaps.",
           [(-0.85, -1.05, 0.4, 0.4, 0), (-0.85, 0.0, 0.4, 0.4, 0), (-0.85, 1.05, 0.4, 0.4, 0),
            (0.0, -0.6, 0.3, 0.3, 0), (0.0, 0.6, 0.3, 0.3, 0),
            (0.85, -1.05, 0.4, 0.4, 0), (0.85, 0.0, 0.4, 0.4, 0), (0.85, 1.05, 0.4, 0.4, 0)],
           [(-1.7, 0.5, 0.0), (-1.7, -0.5, 0.0), (1.7, 0.5, PI), (1.7, -0.5, PI)],
           [(1.4, -0.6), (1.4, 0.6), (-1.4, -0.6), (-1.4, 0.6)]),
    9005: ("corridor swap",
           "Two long blocks leave one straight corridor, 0.8 m wide and 2 m long, across the middle. Two robots at each "
           "end must all go through it to the far side, so they pass each other head-on inside the corridor.",
           [(0.0, 0.925, 2.0, 1.05, 0), (0.0, -0.925, 2.0, 1.05, 0)],
           [(-1.7, 0.6, 0.0), (-1.7, -0.6, 0.0), (1.7, 0.6, PI), (1.7, -0.6, PI)],
           [(1.35, -0.6), (1.35, 0.6), (-1.35, -0.6), (-1.35, 0.6)]),
}
IDS = tuple(DESIGNS)


def get(seed):
    """Layout for a designed id, else None."""
    if seed not in DESIGNS:
        return None
    _, _, obst, starts, goals = DESIGNS[seed]
    return L.Layout(seed=seed, obstacles=[tuple(map(float, o)) for o in obst], starts=[tuple(map(float, s)) for s in starts],
                    goals=[tuple(map(float, g)) for g in goals], arena=(4.0, 3.0))


def shortest_path(lay, start, goal):
    """Grid shortest path (8-neighbour BFS on the inflated free map) as a list of (x, y) points, or None."""
    free = ~L.blocked_map(lay, L.ROBOT_RADIUS + L.CLEARANCE_MARGIN)
    rows, cols = free.shape
    s, g = L._cell(lay, *start), L._cell(lay, *goal)
    prev = {s: None}
    q = deque([s])
    while q:
        c = q.popleft()
        if c == g:
            break
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (c[0] + dr, c[1] + dc)
                if (dr or dc) and 0 <= n[0] < rows and 0 <= n[1] < cols and free[n] and n not in prev:
                    prev[n] = c
                    q.append(n)
    if g not in prev:
        return None
    path, c = [], g
    w, h = lay.arena
    while c is not None:
        path.append(((c[1] + 0.5) * L.GRID_RES - w / 2, (c[0] + 0.5) * L.GRID_RES - h / 2))
        c = prev[c]
    return path[::-1]


def conflicts(lay, speed=0.15, near=0.35, window=4.0):
    """Robot pairs whose shortest paths come within `near` m of each other at roughly the same time (both driving at
    `speed` m/s from t=0, arrival within `window` s): the encounters the design forces. Returns [(i, j, x, y, t_i, t_j)]."""
    paths = [np.array(shortest_path(lay, s[:2], g)) for s, g in zip(lay.starts, lay.goals)]
    times = [np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))]) / speed for p in paths]
    out = []
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.hypot(paths[i][:, None, 0] - paths[j][None, :, 0], paths[i][:, None, 1] - paths[j][None, :, 1])
            ok = (d < near) & (np.abs(times[i][:, None] - times[j][None, :]) < window)
            if ok.any():
                a, b = np.argwhere(ok)[0]
                out.append((i, j, *paths[i][a], times[i][a], times[j][b]))
    return out, paths


def check(lay):
    """Same acceptance rules as tools/layout.generate_layout, plus the 8-box limit of the 2D pool."""
    msgs = []
    if len(lay.obstacles) > L.OBSTACLE_COUNT[1]:
        msgs.append(f"{len(lay.obstacles)} boxes > 2D pool limit {L.OBSTACLE_COUNT[1]}")
    if not L.validate(lay):
        msgs.append("a start or goal is blocked or not connected")
    for i, g in enumerate(lay.goals):
        if not L.goal_visible(lay, g):
            msgs.append(f"goal {i} hidden from the camera")
    w, h = lay.arena
    for i, (x, y) in enumerate([s[:2] for s in lay.starts] + list(lay.goals)):
        if abs(x) > w / 2 - L.WALL_MARGIN + 1e-9 or abs(y) > h / 2 - L.WALL_MARGIN + 1e-9:
            msgs.append(f"point {i} closer than {L.WALL_MARGIN} m to a wall")
    for i, (s, g) in enumerate(zip(lay.starts, lay.goals)):
        if math.hypot(s[0] - g[0], s[1] - g[1]) < L.MIN_START_GOAL:
            msgs.append(f"robot {i}: start-goal < {L.MIN_START_GOAL} m")
        for j, s2 in enumerate(lay.starts):
            if j != i and math.hypot(s2[0] - g[0], s2[1] - g[1]) < 0.3:
                msgs.append(f"goal {i} within 0.3 m of start {j}")
    return msgs


def draw(lay, paths, conf, path_png, title):
    """Top-down map: obstacles, starts (arrow = heading), goals (ring), shortest paths, forced encounters (black X)."""
    import cv2
    S, M = 160, 30  # pixels per metre, margin
    w, h = lay.arena
    img = np.full((int(h * S) + 2 * M + 40, int(w * S) + 2 * M, 3), 255, np.uint8)
    px = lambda x, y: (int(M + (x + w / 2) * S), int(40 + M + (h / 2 - y) * S))
    cv2.rectangle(img, px(-w / 2, h / 2), px(w / 2, -h / 2), (60, 110, 60), -1)
    cv2.rectangle(img, px(-w / 2, h / 2), px(w / 2, -h / 2), (130, 130, 130), 6)
    for x, y, sx, sy, yaw in lay.obstacles:
        c, s = math.cos(yaw), math.sin(yaw)
        pts = [px(x + c * a - s * b, y + s * a + c * b) for a, b in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))]
        cv2.fillPoly(img, [np.array(pts, np.int32)], (180, 105, 255))
    bgr = [(40, 40, 220), (0, 140, 255), (220, 90, 30), (0, 210, 240)]  # red, orange, blue, yellow (gen_world.ROBOT_COLORS)
    for i, p in enumerate(paths):
        cv2.polylines(img, [np.array([px(*q) for q in p[::3]], np.int32)], False, bgr[i], 2, cv2.LINE_AA)
    for i, ((x, y, yaw), g) in enumerate(zip(lay.starts, lay.goals)):
        cv2.circle(img, px(x, y), int(0.095 * S), bgr[i], -1, cv2.LINE_AA)
        cv2.arrowedLine(img, px(x, y), px(x + 0.2 * math.cos(yaw), y + 0.2 * math.sin(yaw)), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)
        cv2.circle(img, px(*g), int(0.12 * S), bgr[i], 3, cv2.LINE_AA)
        cv2.putText(img, f"R{i + 1}", (px(x, y)[0] - 12, px(x, y)[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for _, _, x, y, _, _ in conf:
        c = px(x, y)
        cv2.line(img, (c[0] - 7, c[1] - 7), (c[0] + 7, c[1] + 7), (0, 0, 0), 3)
        cv2.line(img, (c[0] - 7, c[1] + 7), (c[0] + 7, c[1] - 7), (0, 0, 0), 3)
    cv2.putText(img, title, (M, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(path_png), img)


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    bad = False
    for k, (name, desc, *_ ) in DESIGNS.items():
        lay = get(k)
        msgs = check(lay)
        bad |= bool(msgs)
        conf, paths = conflicts(lay)
        lens = [float(np.sum(np.hypot(*np.diff(np.array(p), axis=0).T))) for p in paths]
        print(f"{k} {name:14s} {'OK' if not msgs else 'FAIL: ' + '; '.join(msgs)}  boxes {len(lay.obstacles)}  "
              f"shortest paths {', '.join(f'{x:.1f}' for x in lens)} m  forced encounters {len(conf)} "
              f"{[(f'R{i + 1}-R{j + 1}') for i, j, *_ in conf]}")
        if out:
            out.mkdir(parents=True, exist_ok=True)
            draw(lay, paths, conf, out / f"layout_{k}_{name.replace(' ', '_')}.png", f"{k}  {name}")
    sys.exit(1 if bad else 0)
