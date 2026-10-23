"""Rank evaluation layouts for a demo: no start or goal is hidden behind an obstacle from the angled demo camera
(gen_world.demo_camera_pose), and (unless --all) all four robots succeeded in the recorded Gazebo sweeps. Standard arena only.

    python3 tools/demo_layouts.py [se|sw] [--hard] [--all]

Default order: most obstacles, then shortest run. --hard orders by difficulty = 2 x (obstacles in the way of the robots'
straight start-goal lines, summed over robots) + (pairs of robots whose straight paths cross). --all also lists evaluation
seeds 0-199 never run in Gazebo (marked "untested": run them headless first, they may fail)."""
import ast, itertools, pathlib, re, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import gen_world, layout as L

PATH_MARGIN = 0.12  # an obstacle within this distance of a straight start-goal line is "in the way" (robot radius 0.095 + margin)


def sweep_results():
    """{seed: (all four succeeded, policy steps)} from docs/results/*standard*.log."""
    res = {}
    for f in sorted((ROOT / "docs/results").glob("gazebo_standard_arena_seeds_*.log")):
        for line in f.read_text().splitlines():
            m = re.match(r"seed\s+(\d+): (\[.*?\])\s+\((\d+) policy", line)
            if m:
                res[int(m[1])] = (all(s == "success" for s in ast.literal_eval(m[2])), int(m[3]))
    return res


def hidden(lay, cam, p):
    """Fraction of a robot body at p (heights 0.05, 0.12, 0.2 m) whose line of sight from cam crosses an obstacle."""
    bad = 0
    for z in (0.05, 0.12, 0.2):
        pts = cam[None] + np.linspace(0.0, 0.97, 300)[:, None] * (np.array([p[0], p[1], z]) - cam)[None]
        low = pts[pts[:, 2] < gen_world.OBSTACLE_H]
        bad += any((L._box_dist(low[:, 0], low[:, 1], ob) < 1e-9).any() for ob in lay.obstacles)  # _box_dist is 0 inside a box
    return bad / 3


def difficulty(lay):
    """(obstacles in the way per robot, number of crossing robot-path pairs) for straight start-goal lines."""
    s = [np.array(p[:2]) for p in lay.starts]
    g = [np.array(p) for p in lay.goals]
    t = np.linspace(0.0, 1.0, 200)[:, None]
    blocked = []
    for a, b in zip(s, g):
        pts = a + t * (b - a)
        blocked.append(sum(bool((L._box_dist(pts[:, 0], pts[:, 1], ob) < PATH_MARGIN).any()) for ob in lay.obstacles))
    side = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    cross = sum(side(s[i], g[i], s[j]) * side(s[i], g[i], g[j]) < 0 and side(s[j], g[j], s[i]) * side(s[j], g[j], g[i]) < 0
                for i, j in itertools.combinations(range(4), 2))
    return blocked, cross


if __name__ == "__main__":
    side = next((a for a in sys.argv[1:] if a in ("se", "sw")), "se")
    hard, everything = "--hard" in sys.argv, "--all" in sys.argv
    sweep = sweep_results()
    rows = []
    for seed in range(200) if everything else sorted(sweep):
        ok, steps = sweep.get(seed, (None, None))
        if ok is False:
            continue
        lay = L.generate_layout(seed)
        cam = np.array(gen_world.demo_camera_pose(*lay.arena, side)[:3])
        if any(hidden(lay, cam, p[:2]) for p in list(lay.starts) + list(lay.goals)):
            continue
        blocked, cross = difficulty(lay)
        score = 2 * sum(blocked) + cross
        key = (-score, -len(lay.obstacles)) if hard else (-len(lay.obstacles), steps or 9999)
        rows.append((key, seed, len(lay.obstacles), blocked, cross, score, steps))
    print(f"layouts for a demo from the {side} camera (every start and goal visible), {'hardest' if hard else 'most obstacles'} first")
    for _, seed, n, blocked, cross, score, steps in sorted(rows):
        run = f"{steps / 10:.1f} s in the sweep" if steps else "untested"
        print(f"seed {seed:3d}: {n} obstacles, in the way {blocked}, {cross} path crossings, difficulty {score:2d}, {run}")
