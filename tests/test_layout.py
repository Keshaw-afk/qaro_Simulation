"""Run: python3 tests/test_layout.py"""
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "tools"))
import layout as L

t0 = time.time()
counts, deterministic = [], True
for seed in range(300):
    lay = L.generate_layout(seed)
    assert L.validate(lay), seed
    assert L.N_ROBOTS == len(lay.starts) == len(lay.goals)
    assert L.OBSTACLE_COUNT[0] <= len(lay.obstacles) <= L.OBSTACLE_COUNT[1]
    for i, (sx, sy, _) in enumerate(lay.starts):
        assert math.hypot(sx - lay.goals[i][0], sy - lay.goals[i][1]) >= L.MIN_START_GOAL - 1e-9
        for j in range(i):
            assert math.hypot(sx - lay.starts[j][0], sy - lay.starts[j][1]) >= L.MIN_START_SEP - 1e-9
    counts.append(len(lay.obstacles))
    if seed < 20:
        deterministic &= L.generate_layout(seed) == lay
print(f"300 layouts valid; obstacles per layout {min(counts)}-{max(counts)}; deterministic={deterministic}; {time.time()-t0:.1f}s")
assert deterministic

# negative test: box in robot 0's start with four obstacles -> validate must reject
lay = L.generate_layout(7)
x, y, _ = lay.starts[0]
lay.obstacles = lay.obstacles[:2] + [(x + 0.25, y, 0.1, 0.6, 0), (x - 0.25, y, 0.1, 0.6, 0),
                                     (x, y + 0.25, 0.6, 0.1, 0), (x, y - 0.25, 0.6, 0.1, 0)]
assert not L.validate(lay), "boxed-in start was accepted"
# negative test: a start inside an obstacle
lay2 = L.generate_layout(8)
lay2.obstacles.append((lay2.goals[1][0], lay2.goals[1][1], 0.3, 0.3, 0.0))
assert not L.validate(lay2), "goal inside an obstacle was accepted"
lay3 = L.generate_layout(9)
gx, gy = lay3.goals[0]; far = math.hypot(gx, gy)
lay3.obstacles = [(gx + 0.25 * gx / far, gy + 0.25 * gy / far, 0.3, 0.3, 0.0)]   # tall box just beyond the goal, away from the camera
assert not L.goal_visible(lay3, lay3.goals[0]), "occluded goal reported visible"
assert all(L.goal_visible(L.generate_layout(s), g) for s in range(20) for g in L.generate_layout(s).goals)
print("goal-visibility tests passed")
print("negative tests passed (boxed-in start and goal-in-obstacle both rejected)")
