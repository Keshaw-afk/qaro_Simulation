"""Shared benchmark harness (see benchmark/PROTOCOL.md): data split, protocol evaluation, bootstrap statistics.

A controller is a function `ctrl(obs, env) -> actions`, obs = (N, 59) raw observations (not normalised), actions =
(N,) indices into rl/actions.py or (N, 2) normalised (v, w) pairs. `env` is the ArenaVecEnv (only privileged
controllers such as DWA may read its true state)."""
import json
import pathlib
import time
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "rl"), str(ROOT / "benchmark")]
import arena_env as E  # noqa: E402
import pool as PL  # noqa: E402

TEST = np.arange(100, 200)        # evaluation layouts never used for training, tuning or checkpoint selection
PPO_SELECTION = np.arange(0, 100)  # where PPO's checkpoint was chosen: reference only
N_TRAIN_POOL = 2000                # pool_train.npz holds seeds 1000-2999 in order
VAL_IDX = np.arange(1900, 2000)    # seeds 2900-2999: validation for the new methods
TRAIN_IDX = np.arange(0, 1900)     # seeds 1000-2899: training for the new methods
OUT = ROOT / "benchmark" / "results"
RUNS = ROOT / "runs" / "benchmark"


def eval_pool():
    p = PL.build(PL.EVAL_SEEDS, ROOT / "rl" / "pool_eval.npz")
    assert (p["seeds"] == np.arange(200)).all()
    return p


def train_pool():
    p = PL.build(PL.TRAIN_SEEDS, ROOT / "rl" / "pool_train.npz")
    assert len(p["seeds"]) == N_TRAIN_POOL and p["seeds"][0] == 1000
    return p


def subset(pool, idx):
    return {k: v[idx] for k, v in pool.items()}


def run_protocol(pool, ctrl, layouts, shield=True, max_steps=900):
    """Evaluation protocol of rl/evaluate.py on chosen layouts; returns raw per-robot arrays (layouts x 4)."""
    env = E.ArenaVecEnv(pool, mode="eval", eval_idx=np.asarray(layouts), max_steps=max_steps, seed=0, randomize=False, shield=shield)
    obs = env.reset()
    prop = over = calls = 0
    t_ctrl = t_all = 0.0
    t_start = time.perf_counter()
    for _ in range(max_steps):
        t0 = time.perf_counter()
        a = ctrl(obs, env)
        t_ctrl += time.perf_counter() - t0
        calls += 1
        active = env.active.reshape(-1).copy()
        env.step_async(a)
        if shield:  # proposed vs executed action (env.actions holds the executed one after the safety layer)
            a2 = np.asarray(a)
            if a2.ndim == 2:  # same nearest-entry snapping as ArenaVecEnv.step_async
                v, w = (np.clip(a2[:, 0], -1, 1) + 1) / 2 * E.VMAX, np.clip(a2[:, 1], -1, 1) * E.WMAX
                a2 = ((v[:, None] - E.CV[None]) ** 2 / E.VMAX ** 2 + (w[:, None] - E.CW[None]) ** 2 / E.WMAX ** 2).argmin(1)
            over += int((env.actions.reshape(-1) != a2)[active].sum())
            prop += int(active.sum())
        obs, _, _, _ = env.step_wait()
        if not env.active.any():
            break
    t_all = time.perf_counter() - t_start
    return {"layouts": [int(x) for x in layouts], "outcome": env.outcome.tolist(), "cause": env.cause.tolist(),
            "finish_t": env.finish_t.tolist(), "override_rate": over / max(prop, 1), "shield": shield,
            "decision_calls": calls, "decision_s_total": t_ctrl, "eval_wall_s": t_all,
            "decision_us_per_robot": t_ctrl / max(calls, 1) / env.num_envs * 1e6}  # batched over all robots of all layouts


def rates(outcome, cause, finish_t):
    o, c, f = np.asarray(outcome), np.asarray(cause), np.asarray(finish_t)
    ok = o == 1
    return {"success": ok.mean(), "collision": (o == 2).mean(), "timeout": (o == 3).mean(),
            "coll_obstacle": ((o == 2) & (c == 1)).mean(), "coll_wall": ((o == 2) & (c == 2)).mean(),
            "coll_robot": ((o == 2) & (c == 3)).mean(), "all_four": ok.all(1).mean(),
            "time_s": float(f[ok].mean() * E.DT) if ok.any() else float("nan")}


def summarize(raw, ref=None, n_boot=10000, seed=0):
    """Point estimates + 95% layout-bootstrap CIs; with ref (raw result of the reference method on the same layouts),
    also the paired bootstrap CI of (this - ref) for success and collision."""
    o, c, f = (np.asarray(raw[k]) for k in ("outcome", "cause", "finish_t"))
    rng = np.random.default_rng(seed)
    L = len(o)
    idx = rng.integers(0, L, (n_boot, L))
    out = {k: float(v) for k, v in rates(o, c, f).items()}
    for k in ("success", "collision", "timeout", "all_four"):
        x = {"success": (o == 1).mean(1), "collision": (o == 2).mean(1), "timeout": (o == 3).mean(1), "all_four": (o == 1).all(1) * 1.0}[k]
        bs = x[idx].mean(1)
        out[k + "_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    if ref is not None:
        ro = np.asarray(ref["outcome"])
        assert list(ref["layouts"]) == list(raw["layouts"]), "paired comparison needs the same layouts"
        for k, code in (("success", 1), ("collision", 2)):
            d = (o == code).mean(1) - (ro == code).mean(1)
            bs = d[idx].mean(1)
            out[f"d_{k}_vs_ref"] = float(d.mean())
            out[f"d_{k}_vs_ref_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out["override_rate"] = raw.get("override_rate", float("nan"))
    out["decision_us_per_robot"] = raw.get("decision_us_per_robot", float("nan"))
    out["n_robot_runs"] = int(o.size)
    return out


def fmt(name, s):
    ci = lambda k: f"[{s[k + '_ci'][0] * 100:4.1f}-{s[k + '_ci'][1] * 100:4.1f}]"
    line = (f"{name:14s} success {s['success'] * 100:5.1f}% {ci('success')}  collision {s['collision'] * 100:5.1f}% {ci('collision')} "
            f"(obst {s['coll_obstacle'] * 100:.1f} wall {s['coll_wall'] * 100:.1f} robot {s['coll_robot'] * 100:.1f})  "
            f"timeout {s['timeout'] * 100:5.1f}%  all4 {s['all_four'] * 100:5.1f}%  t {s['time_s']:.1f}s  overrides {s['override_rate'] * 100:.1f}%")
    if "d_success_vs_ref" in s:
        line += f"  dSucc vs PPO {s['d_success_vs_ref'] * 100:+.1f} [{s['d_success_vs_ref_ci'][0] * 100:+.1f},{s['d_success_vs_ref_ci'][1] * 100:+.1f}]"
    return line


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))


def score(s):
    """Checkpoint/tuning selection score, the same rule PPO's training used (rl/train.py)."""
    return s["success"] - 0.5 * s["collision"]


# ---------------------------------------------------------------- simple controllers
def npz_ctrl(path):
    import deploy
    pol = deploy.Policy(str(path))
    return lambda obs, env: pol.index(obs)


def goto_ctrl(obs, env):
    import evaluate as EV
    return EV.goto_goal(obs)


def dwa_ctrl(obs, env):
    import expert
    return expert.dwa_actions(env)
