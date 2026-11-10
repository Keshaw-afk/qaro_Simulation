"""Deterministic reactive controller (no learning) that uses ONLY the policy's 59-number observation: a dynamic-window
search over the same 45 commands, with obstacles taken from the 36 lidar sectors and other robots from the camera slots.

For every command, the robot's arc over HORIZON s is predicted (robot frame, start at the origin). Obstacles = one point per
lidar sector at its measured range (sectors reading >= 2 m are free; the 0.15 m blind zone counts as 0.15 m); other robots
move at their camera-measured velocity. Commands whose predicted chassis gap falls below MARGIN are vetoed; the rest are
scored by progress toward the goal, final heading to the goal, clearance and speed. If nothing is safe, the command with
the largest gap is taken. Parameters are tuned on validation layouts only (`tune`).

    .venv/bin/python benchmark/rule.py tune            # grid over PARAM_GRID on validation (seeds 2900-2999)"""
import itertools
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

import common as C
from common import E

CV, CW = E.CV.astype(np.float64), E.CW.astype(np.float64)
SECTOR_ANG = np.radians(-180 + 10 * np.arange(E.N_POOL) + 5)
PARAM_GRID = dict(horizon=[1.0, 1.5, 2.0], margin=[0.02, 0.05], w_clear=[0.5, 1.0, 2.0], w_head=[0.3, 1.0])
DEFAULT = dict(horizon=1.5, margin=0.03, w_clear=1.0, w_head=0.3, w_prog=2.0, w_speed=0.2, steps=8)


def make(params=None):
    p = {**DEFAULT, **(params or {})}
    T = (np.arange(p["steps"]) + 1) * p["horizon"] / p["steps"]
    w_eff = np.where(np.abs(CW) < 1e-3, 1e-3, CW)[:, None]
    th = w_eff * T                                                       # (45, S) heading along the arc
    ax, ay = CV[:, None] / w_eff * np.sin(th), -CV[:, None] / w_eff * (np.cos(th) - 1)  # arc points, robot frame

    def ctrl(obs, env=None):
        o = np.asarray(obs, np.float64)
        n = len(o)
        r = np.maximum(o[:, :E.N_POOL] * E.RANGE_MAX, E.RANGE_MIN)              # (n, 36) metres
        valid = o[:, :E.N_POOL] < 0.999
        px = E.LIDAR_X + r * np.cos(SECTOR_ANG)                                # obstacle points in the robot frame (lidar sits behind the axle)
        py = r * np.sin(SECTOR_ANG)
        dx = ax[None, :, :, None] - px[:, None, None, :]
        dy = ay[None, :, :, None] - py[:, None, None, :]
        d_obs = np.where(valid[:, None, None, :], np.hypot(dx, dy), np.inf).min(-1) - E.R_ROBOT   # (n, 45, S)
        rel = o[:, 41:53].reshape(n, 3, 4)
        rp, rv = rel[..., :2] * 3.0, rel[..., 2:] * 0.4
        ov = rv + np.stack([o[:, 39] * E.VMAX, np.zeros(n)], -1)[:, None, :]  # other robot's own velocity, robot frame
        far = (np.abs(rel[..., :2]) >= 0.999).any(-1)                         # clipped slot: robot >= 3 m away, ignore
        ox = rp[:, None, None, :, 0] + ov[:, None, None, :, 0] * T[None, None, :, None]
        oy = rp[:, None, None, :, 1] + ov[:, None, None, :, 1] * T[None, None, :, None]
        d_rob = np.where(far[:, None, None, :], np.inf, np.hypot(ax[None, :, :, None] - ox, ay[None, :, :, None] - oy)).min(-1) - 2 * E.R_ROBOT
        clear = np.minimum(d_obs, d_rob).min(-1)                               # (n, 45)
        gd = o[:, 36] * 4.0
        gx, gy = gd * o[:, 37], gd * o[:, 38]
        d_end = np.hypot(gx[:, None] - ax[None, :, -1], gy[:, None] - ay[None, :, -1])
        head = np.cos(np.arctan2(gy[:, None] - ay[None, :, -1], gx[:, None] - ax[None, :, -1]) - th[None, :, -1])
        score = (p["w_prog"] * (gd[:, None] - d_end) / (E.VMAX * p["horizon"]) + p["w_head"] * head
                 + p["w_clear"] * np.minimum(clear, 0.4) / 0.4 + p["w_speed"] * CV[None] / E.VMAX)
        score = np.where(clear > p["margin"], score, -1e6 + clear)
        return score.argmax(1)
    return ctrl


def _val(params):
    tp = C.train_pool()
    sm = C.summarize(C.run_protocol(tp, make(params), C.VAL_IDX), n_boot=200)
    return params, {k: sm[k] for k in ("success", "collision", "timeout", "coll_obstacle", "coll_wall", "coll_robot", "time_s")}


def tune(procs=4):
    from meter import Meter
    grid = [dict(zip(PARAM_GRID, v)) for v in itertools.product(*PARAM_GRID.values())]
    with Meter("rule_tune", kind="tune", method="rule", n_sets=len(grid)) as m:
        with Pool(procs) as pool:
            res = pool.map(_val, grid)
        best = max(res, key=lambda x: C.score(x[1]))
        m.add(best=best[0])
    for p_, s in sorted(res, key=lambda x: -C.score(x[1])):
        print(json.dumps(p_), f"val success {s['success']:.3f} collision {s['collision']:.3f} timeout {s['timeout']:.3f}", flush=True)
    C.RUNS.mkdir(parents=True, exist_ok=True)
    (C.RUNS / "rule_tuning.json").write_text(json.dumps([{"params": a, **b} for a, b in res], indent=1))
    C.save("rule_best_params", {"params": {**DEFAULT, **best[0]}, "validation": best[1], "grid": PARAM_GRID, "n_sets": len(grid)})
    print("BEST", json.dumps({**DEFAULT, **best[0]}), best[1])


# Extension (PROTOCOL amendment 3): the 36-set grid's best sat on its lower edge for horizon, margin and w_head, so the
# grid is extended past that edge, within the pre-registered budget of 50 sets (36 + 14), still on validation only.
PARAM_GRID_EXT = dict(horizon=[0.7, 1.0], margin=[0.01, 0.02], w_clear=[0.5, 2.0], w_head=[0.1, 0.3])


def tune_ext(procs=6):
    from meter import Meter
    prev = json.loads((C.RUNS / "rule_tuning.json").read_text())
    done = [{k: r["params"][k] for k in PARAM_GRID} for r in prev]
    grid = [g for g in (dict(zip(PARAM_GRID_EXT, v)) for v in itertools.product(*PARAM_GRID_EXT.values())) if g not in done]
    assert len(done) + len(grid) <= 50, "tuning budget (PROTOCOL.md) exceeded"
    with Meter("rule_tune_ext", kind="tune", method="rule", n_sets=len(grid)) as m:
        with Pool(procs) as pool:
            new = pool.map(_val, grid)
        res = [(r["params"], {k: v for k, v in r.items() if k != "params"}) for r in prev] + new
        best = max(res, key=lambda x: C.score(x[1]))
        m.add(best=best[0])
    for p_, s in sorted(res, key=lambda x: -C.score(x[1]))[:10]:
        print(json.dumps(p_), f"val success {s['success']:.3f} collision {s['collision']:.3f} timeout {s['timeout']:.3f}", flush=True)
    (C.RUNS / "rule_tuning.json").write_text(json.dumps([{"params": a, **b} for a, b in res], indent=1))
    C.save("rule_best_params", {"params": {**DEFAULT, **best[0]}, "validation": best[1], "grid": PARAM_GRID,
                                "grid_ext": PARAM_GRID_EXT, "n_sets": len(res)})
    print("BEST", json.dumps({**DEFAULT, **best[0]}), best[1])


def best_ctrl():
    return make(json.loads((C.OUT / "rule_best_params.json").read_text())["params"])


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif sys.argv[1] == "tune_ext":
        tune_ext(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
