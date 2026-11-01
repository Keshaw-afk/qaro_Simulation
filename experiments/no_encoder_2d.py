"""Hardware experiment (2D, no Gazebo): how much does the policy need wheel-encoder input, and does it work with one robot?

The policy observation has exactly two encoder-derived entries: own v and w (columns OBS_VW = 39, 40, normalised by VMAX, WMAX).
This script runs the standard 2D evaluation protocol but overwrites those two columns with a substitute before the policy sees them.
It does NOT modify the shared observation code (rl/arena_env.py, rl/deploy.py); the substitution happens on the returned array.

Substitutes for (v, w):
  encoder  - true v, w (what the policy was trained with; baseline)
  cmdmodel - first-order lag model of the commanded (v, w): est += (cmd - est) * 0.75   (no sensor at all)
  camera   - finite difference of camera pose: v = (dpos . heading)/DT, w = wrap(dyaw)/DT, with Gaussian pose noise
  zeros    - v = w = 0 (input removed)

Actuator model (the real robot has no encoders, so no closed-loop wheel speed control):
  nominal  - motors do exactly what is commanded (default eval)
  train    - the randomisation used during training: speed x U(0.9,1.1), lag U(0.5,1.0), 30% one-step delay
  harsh    - stress test, wider than training: speed x U(0.7,1.3), lag U(0.3,1.0), 50% one-step delay

Single-robot mode: only robot k drives; the other three are inactive and parked (see park= below), so their observation
slots read fixed positions with relative velocity = minus own velocity, as the env computes it. Each of k = 0..3 is run
on every layout with its own start, goal and ID.

Usage (from repo root, needs .venv):  .venv/bin/python experiments/no_encoder_2d.py [N_LAYOUTS=100] [four,single]

Absent robots (single-robot mode, park=): "outside" (default, deployable: three fixed virtual robots 0.3 m outside the walls),
"start" (left inactive at their start poses), "far" (50 m away; slot values saturate, out of training distribution).
"""
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
import arena_env as E  # noqa: E402
import deploy  # noqa: E402
import pool as PL  # noqa: E402
from actions import CV, CW  # noqa: E402

OBS_VW = E.N_POOL + 3  # 39: own v, 40: own w
assert E.OBS_DIM == 59


def set_actuators(env, mode, rng):
    shape = env.speed_scale.shape
    if mode == "nominal":
        return
    lo, hi, lag_lo, p_delay = {"train": (0.9, 1.1, 0.5, 0.3), "harsh": (0.7, 1.3, 0.3, 0.5)}[mode]
    env.speed_scale[:] = rng.uniform(lo, hi, shape)
    env.lag[:] = rng.uniform(lag_lo, 1.0, shape)
    env.delay[:] = rng.random(shape) < p_delay


OUTSIDE = [(-E.ARENA_W / 2 - 0.3, 0.0), (E.ARENA_W / 2 + 0.3, 0.0), (0.0, E.ARENA_H / 2 + 0.3)]  # fixed virtual-robot positions


def run(policy, pool, n, subst="encoder", act="nominal", single=None, park="outside", cam_sigma=0.01, cam_yaw_sigma=0.02, seed=0, max_steps=900):
    rng = np.random.default_rng(seed)
    env = E.ArenaVecEnv(pool, mode="eval", eval_idx=np.arange(n), max_steps=max_steps, seed=seed, randomize=False, shield=True)
    obs = env.reset()
    set_actuators(env, act, rng)
    if single is not None:
        slots = iter(OUTSIDE)
        for a in range(E.N_AG):
            if a != single:
                env.active[:, a] = False
                if park == "far":
                    env.pos[:, a] = (50.0 + 5 * a, 50.0)  # far away: slots saturate at (+-1, +-1), never seen in training
                elif park == "outside":
                    env.pos[:, a] = next(slots)  # virtual robots just outside the walls: in training range, wall blocks lidar
                # park == "start": stay inactive at the layout's start pose (physically present, like a finished robot)
        obs = env._all_obs()
    W = env.W
    est = np.zeros((W, E.N_AG, 2), np.float32)
    cam_pos = env.pos + rng.normal(0, cam_sigma, env.pos.shape)
    cam_yaw = env.yaw + rng.normal(0, cam_yaw_sigma, env.yaw.shape)
    vw = np.zeros((W, E.N_AG, 2), np.float32)

    for _ in range(max_steps):
        o = obs.reshape(W, E.N_AG, -1).copy()
        if subst != "encoder":
            o[..., OBS_VW] = vw[..., 0] / E.VMAX
            o[..., OBS_VW + 1] = vw[..., 1] / E.WMAX
        k = policy.index(o.reshape(W * E.N_AG, -1))
        env.step_async(k)
        idx = env.actions  # after the safety layer: what was actually commanded
        obs, _, _, _ = env.step_wait()
        # update substitute for the next observation
        if subst == "cmdmodel":
            cmd = np.stack([CV[idx], CW[idx]], -1)
            est += (cmd - est) * 0.75
            est *= env.active[..., None]
            vw = est
        elif subst == "camera":
            p = env.pos + rng.normal(0, cam_sigma, env.pos.shape)
            y = env.yaw + rng.normal(0, cam_yaw_sigma, env.yaw.shape)
            dp = p - cam_pos
            v = (dp[..., 0] * np.cos(y) + dp[..., 1] * np.sin(y)) / E.DT
            dy = (y - cam_yaw + math.pi) % (2 * math.pi) - math.pi
            vw = np.stack([v, dy / E.DT], -1).astype(np.float32)
            cam_pos, cam_yaw = p, y
        elif subst == "zeros":
            vw = np.zeros_like(vw)
        if not env.active.any():
            break

    o = env.outcome if single is None else env.outcome[:, [single]]
    ft = env.finish_t if single is None else env.finish_t[:, [single]]
    cause = env.cause if single is None else env.cause[:, [single]]
    ok = o == 1
    cz = cause[o == 2]
    return {"success": float(ok.mean()), "collision": float((o == 2).mean()), "timeout": float((o == 3).mean()),
            "coll_obstacle": float((cz == 1).sum() / o.size), "coll_wall": float((cz == 2).sum() / o.size), "coll_robot": float((cz == 3).sum() / o.size),
            "mean_time_s": float(ft[ok].mean() * E.DT) if ok.any() else float("nan"), "n_robot_runs": int(o.size)}


def run_single(policy, pool, n, **kw):
    rs = [run(policy, pool, n, single=k, **kw) for k in range(E.N_AG)]
    tot = sum(r["n_robot_runs"] for r in rs)
    out = {key: sum(r[key] * r["n_robot_runs"] for r in rs) / tot for key in rs[0] if key not in ("n_robot_runs", "mean_time_s")}
    out["mean_time_s"] = float(np.nanmean([r["mean_time_s"] for r in rs]))
    out["n_robot_runs"] = tot
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    which = sys.argv[2].split(",") if len(sys.argv) > 2 else ["four", "single"]
    pool = PL.build(PL.EVAL_SEEDS, ROOT / "rl" / "pool_eval.npz")
    pol = deploy.Policy(str(ROOT / "weights" / "final_policy.npz"))
    results = []
    for robots in which:
        for act in ("nominal", "train", "harsh"):
            for subst in ("encoder", "cmdmodel", "camera", "zeros"):
                f = run_single if robots == "single" else run
                r = f(pol, pool, n, subst=subst, act=act)
                r.update(robots=robots, actuators=act, vw_source=subst)
                results.append(r)
                print(f"{robots:6s} {act:7s} {subst:8s}  success {r['success']:.3f}  collision {r['collision']:.3f} "
                      f"(obst {r['coll_obstacle']:.3f} wall {r['coll_wall']:.3f} robot {r['coll_robot']:.3f})  timeout {r['timeout']:.3f}  "
                      f"time {r['mean_time_s']:.1f}s  n={r['n_robot_runs']}", flush=True)
    out = ROOT / "runs" / f"no_encoder_2d_{'_'.join(which)}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}")
