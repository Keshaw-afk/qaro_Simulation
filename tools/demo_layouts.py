"""Rank evaluation layouts for a demo: all four robots succeeded in the recorded Gazebo sweeps, and no start or goal is hidden
behind an obstacle from the angled demo camera (gen_world.demo_camera_pose). Standard arena only.

    python3 tools/demo_layouts.py [se|sw]

Prints one line per layout, best first (most obstacles, then shortest run)."""
import ast, pathlib, re, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import gen_world, layout as L


def all_success_layouts():
    """{seed: policy steps} for layouts where all four robots succeeded in docs/results/*standard*.log."""
    ok = {}
    for f in sorted((ROOT / "docs/results").glob("gazebo_standard_arena_seeds_*.log")):
        for line in f.read_text().splitlines():
            m = re.match(r"seed\s+(\d+): (\[.*?\])\s+\((\d+) policy", line)
            if m and all(s == "success" for s in ast.literal_eval(m[2])):
                ok[int(m[1])] = int(m[3])
    return ok


def hidden(lay, cam, p):
    """Fraction of a robot body at p (heights 0.05, 0.12, 0.2 m) whose line of sight from cam crosses an obstacle."""
    bad = 0
    for z in (0.05, 0.12, 0.2):
        pts = cam[None] + np.linspace(0.0, 0.97, 300)[:, None] * (np.array([p[0], p[1], z]) - cam)[None]
        low = pts[pts[:, 2] < gen_world.OBSTACLE_H]
        bad += any((L._box_dist(low[:, 0], low[:, 1], ob) < 1e-9).any() for ob in lay.obstacles)  # _box_dist is 0 inside a box
    return bad / 3


if __name__ == "__main__":
    side = sys.argv[1] if len(sys.argv) > 1 else "se"
    rows = []
    for seed, steps in all_success_layouts().items():
        lay = L.generate_layout(seed)
        cam = np.array(gen_world.demo_camera_pose(*lay.arena, side)[:3])
        if all(hidden(lay, cam, p[:2]) == 0 for p in list(lay.starts) + list(lay.goals)):
            rows.append((-len(lay.obstacles), steps, seed))
    print(f"layouts for a demo from the {side} camera: all 4 succeeded in the sweep, every start and goal visible")
    for neg_n, steps, seed in sorted(rows):
        print(f"seed {seed:3d}: {-neg_n} obstacles, {steps / 10:.1f} s in the sweep")
