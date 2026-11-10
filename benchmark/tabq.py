"""Tabular Q-learning ("classic" RL) on a discretised version of the same 59-number observation.

    .venv/bin/python benchmark/tabq.py train CONFIG SEED STEPS NAME     # writes runs/benchmark/tabq/NAME/{best,latest}.npz, log.csv

State (6,480 states): goal bearing (8 bins) x goal distance (3: <0.4, <1.2, >=1.2 m) x lidar minimum in front (-30..30 deg),
front-left (30..90) and front-right (-90..-30), 3 levels each (<0.35, <0.8, >=0.8 m) x nearest other robot within 1 m
(none / ahead / left / right / behind) x own speed (2: below / above half of max). Actions: the same 45 commands.
Learning: one shared table for all 256 robots of 64 training worlds (seeds 1000-2899), env reward unchanged, safety layer
on (part of the environment, as for PPO). Duplicate (state, action) updates within one step are averaged.
Every 1M steps: protocol evaluation on validation layouts (seeds 2900-2999); the best (success - 0.5 collision) is kept.

Representation variant "fine" (protocol amendment 2026-11-10, after the coarse table plateaued at ~60% on validation and
before any test-set evaluation of tabq): 157,464 states = goal bearing 12 bins x distance 3 x five lidar regions
(front -20..20, front-left 20..60, front-right -60..-20, left 60..110, right -110..-60 deg; 3 levels each) x nearest
robot within 1 m (none, or ahead/left/right/behind x closing/not) x own speed 2. Configs prefixed "f" use it."""
import json
import math
import sys
import time

import numpy as np

import common as C
from common import E
from meter import Meter

N_STATE = 8 * 3 * 27 * 5 * 2
FRONT, FL, FR = np.arange(15, 21), np.arange(21, 27), np.arange(9, 15)  # lidar sector j covers angles -180+10j .. -170+10j deg

CONFIGS = {  # tuning grid (benchmark/PROTOCOL.md: up to 6 configurations x 2M steps, judged on validation)
    "a": dict(alpha="visit", alpha_min=0.02, gamma=0.99, eps_frac=0.5),
    "b": dict(alpha="const", alpha_min=0.10, gamma=0.99, eps_frac=0.5),
    "c": dict(alpha="visit", alpha_min=0.02, gamma=0.95, eps_frac=0.5),
    "d": dict(alpha="visit", alpha_min=0.02, gamma=0.99, eps_frac=0.2),
    "e": dict(alpha="const", alpha_min=0.05, gamma=0.95, eps_frac=0.3),
    "f": dict(alpha="visit", alpha_min=0.05, gamma=0.97, eps_frac=0.3),
}


def _lvl(m):  # lidar values are range / 2 m
    return np.where(m < 0.175, 0, np.where(m < 0.4, 1, 2))


def discretize(obs):
    o = np.asarray(obs)
    bear = np.arctan2(o[:, 38], o[:, 37])
    b = ((bear + math.pi) / (2 * math.pi) * 8).astype(int) % 8
    d = o[:, 36] * 4.0
    db = np.where(d < 0.4, 0, np.where(d < 1.2, 1, 2))
    lid = _lvl(o[:, FRONT].min(1)) * 9 + _lvl(o[:, FL].min(1)) * 3 + _lvl(o[:, FR].min(1))
    rel = o[:, 41:53].reshape(-1, 3, 4)[..., :2] * 3.0                    # other robots, robot frame, metres
    dist = np.hypot(rel[..., 0], rel[..., 1])
    k = dist.argmin(1)
    near = rel[np.arange(len(o)), k]
    ang = np.degrees(np.arctan2(near[:, 1], near[:, 0]))
    rb = np.where(dist.min(1) >= 1.0, 0, np.where(np.abs(ang) <= 45, 1, np.where((ang > 45) & (ang <= 135), 2, np.where((ang < -45) & (ang >= -135), 3, 4))))
    vb = (o[:, 39] > 0.5).astype(int)
    return ((((b * 3 + db) * 27 + lid) * 5 + rb) * 2 + vb).astype(np.int64)


F5 = [np.arange(16, 20), np.arange(20, 24), np.arange(12, 16), np.arange(24, 29), np.arange(7, 12)]  # F, FL, FR, L, R sectors
N_FINE = 12 * 3 * 243 * 9 * 2


def discretize_fine(obs):
    o = np.asarray(obs)
    bear = np.arctan2(o[:, 38], o[:, 37])
    b = ((bear + math.pi) / (2 * math.pi) * 12).astype(int) % 12
    d = o[:, 36] * 4.0
    db = np.where(d < 0.4, 0, np.where(d < 1.2, 1, 2))
    lid = np.zeros(len(o), np.int64)
    for sec in F5:
        lid = lid * 3 + _lvl(o[:, sec].min(1))
    rel = o[:, 41:53].reshape(-1, 3, 4)
    rp, rv = rel[..., :2] * 3.0, rel[..., 2:] * 0.4
    dist = np.hypot(rp[..., 0], rp[..., 1])
    k = dist.argmin(1)
    i = np.arange(len(o))
    near, nv = rp[i, k], rv[i, k]
    ang = np.degrees(np.arctan2(near[:, 1], near[:, 0]))
    dirb = np.where(np.abs(ang) <= 45, 0, np.where((ang > 45) & (ang <= 135), 1, np.where((ang < -45) & (ang >= -135), 2, 3)))
    closing = ((near * nv).sum(1) < 0).astype(int)
    rb = np.where(dist.min(1) >= 1.0, 0, 1 + dirb * 2 + closing)
    vb = (o[:, 39] > 0.5).astype(int)
    return ((((b * 3 + db) * 243 + lid) * 9 + rb) * 2 + vb).astype(np.int64)


REPRS = {"coarse": (discretize, N_STATE), "fine": (discretize_fine, N_FINE)}
CONFIGS.update({"fa": dict(CONFIGS["a"], repr="fine"), "fd": dict(CONFIGS["d"], repr="fine"),
                "fg": dict(alpha="visit", alpha_min=0.05, gamma=0.99, eps_frac=0.5, repr="fine")})


def ctrl_from(path):
    d = np.load(path)
    disc = REPRS[str(d["repr"]) if "repr" in d.files else "coarse"][0]
    Q = d["Q"]
    return lambda obs, env: Q[disc(obs)].argmax(1)


def train(cfg_name, seed, steps, name):
    cfg = CONFIGS[cfg_name]
    rname = cfg.get("repr", "coarse")
    discretize, N_STATE = REPRS[rname]
    out = C.RUNS / "tabq" / name
    out.mkdir(parents=True, exist_ok=True)
    tp = C.train_pool()
    env = E.ArenaVecEnv(C.subset(tp, C.TRAIN_IDX), n_worlds=64, seed=seed)
    rng = np.random.default_rng(seed)
    N = env.num_envs
    Q = np.zeros((N_STATE, E.N_ACT), np.float32)
    n_visit = np.zeros((N_STATE, E.N_ACT), np.int64)
    obs = env.reset()
    s = discretize(obs)
    t, next_eval, best, t0 = 0, 1_000_000, -1e9, time.time()
    ep_s = ep_c = ep_n = 0
    # fine-grained training curve every 100k steps (episodes ended in the interval, all robots)
    curve = open(out / "train_curve.csv", "w")
    curve.write("steps,wall_s,steps_per_s,eps,episodes,success,collision,truncated,mean_return,td_abs_mean,q_mean,q_max,visited_states,visited_pairs\n")
    next_curve, c_t0, c_steps = 100_000, time.time(), 0
    cv = dict(n=0, s=0, c=0, tr=0, ret=0.0, td=0.0, tdn=0)
    log = open(out / "log.csv", "w")
    log.write("steps,wall_s,eps,val_success,val_collision,val_timeout,val_score,visited_states,train_episodes,train_success_rate,train_collision_rate\n")
    with Meter(f"tabq_train_{name}", kind="train", method="tabq", config=cfg_name, seed=seed, repr=rname) as m:
        while t < steps:
            eps = max(0.05, 1.0 - t / (cfg["eps_frac"] * steps) * 0.95)
            greedy = Q[s].argmax(1)
            a = np.where(rng.random(N) < eps, rng.integers(0, E.N_ACT, N), greedy)
            env.step_async(a)
            obs2, r, done, infos = env.step_wait()
            s_next = discretize(obs2)        # state the robot is in now (after an episode end: the respawned / new-goal state)
            s_boot = s_next.copy()           # state to bootstrap the target from (after an episode end: the final state)
            terminal = np.zeros(N, bool)
            for i in np.nonzero(done)[0]:
                info = infos[i]
                if "terminal_observation" in info:
                    s_boot[i] = discretize(info["terminal_observation"][None])[0]
                    terminal[i] = info.get("is_success", False) or info.get("collision", False)
                    if "episode" in info and not info.get("TimeLimit.truncated", False):
                        ep_n += 1; ep_s += info.get("is_success", False); ep_c += info.get("collision", False)
                    if "episode" in info:
                        cv["n"] += 1; cv["s"] += info.get("is_success", False); cv["c"] += info.get("collision", False)
                        cv["tr"] += info.get("TimeLimit.truncated", False); cv["ret"] += info["episode"]["r"]
            target = r + cfg["gamma"] * Q[s_boot].max(1) * (~terminal)
            td = target - Q[s, a]
            cv["td"] += float(np.abs(td).sum()); cv["tdn"] += N
            flat = s * E.N_ACT + a
            hit, inv = np.unique(flat, return_inverse=True)          # distinct (state, action) pairs updated this step
            sums = np.bincount(inv, weights=td)                         # duplicates within the step are averaged
            cnt = np.bincount(inv)
            n_visit.ravel()[hit] += cnt
            alpha = np.full(len(hit), cfg["alpha_min"]) if cfg["alpha"] == "const" else np.maximum(cfg["alpha_min"], 1.0 / n_visit.ravel()[hit] ** 0.6)
            Q.ravel()[hit] += (alpha * sums / cnt).astype(np.float32)
            s = s_next
            t += N
            if t >= next_curve:
                next_curve += 100_000
                n = max(cv["n"], 1)
                curve.write(f"{t},{time.time() - t0:.1f},{(t - c_steps) / max(time.time() - c_t0, 1e-9):.0f},{eps:.4f},{cv['n']},{cv['s'] / n:.4f},{cv['c'] / n:.4f},"
                            f"{cv['tr'] / n:.4f},{cv['ret'] / n:.3f},{cv['td'] / max(cv['tdn'], 1):.4f},{float(Q.mean()):.4f},{float(Q.max()):.4f},"
                            f"{int((n_visit.sum(1) > 0).sum())},{int((n_visit > 0).sum())}\n")
                curve.flush()
                c_t0, c_steps = time.time(), t
                cv = dict(n=0, s=0, c=0, tr=0, ret=0.0, td=0.0, tdn=0)
            if t >= next_eval or t >= steps:
                next_eval += 1_000_000
                np.savez_compressed(out / "latest.npz", Q=Q, n_visit=n_visit, repr=rname)
                raw = C.run_protocol(tp, ctrl_from(out / "latest.npz"), C.VAL_IDX)
                sm = C.summarize(raw, n_boot=200)
                visited = int((n_visit.sum(1) > 0).sum())
                log.write(f"{t},{time.time() - t0:.0f},{eps:.3f},{sm['success']:.4f},{sm['collision']:.4f},{sm['timeout']:.4f},{C.score(sm):.4f},{visited},{ep_n},{ep_s / max(ep_n, 1):.4f},{ep_c / max(ep_n, 1):.4f}\n")
                log.flush()
                print(f"[tabq {name}] steps {t:>9d} eps {eps:.2f} val success {sm['success']:.3f} collision {sm['collision']:.3f} "
                      f"timeout {sm['timeout']:.3f} visited {visited}/{N_STATE} train-episodes {ep_n} ({time.time() - t0:.0f}s)", flush=True)
                ep_s = ep_c = ep_n = 0
                if C.score(sm) > best:
                    best = C.score(sm)
                    np.savez_compressed(out / "best.npz", Q=Q, n_visit=n_visit, repr=rname)
                    (out / "best.json").write_text(json.dumps({"steps": t, **{k: sm[k] for k in ("success", "collision", "timeout")}}))
        m.add(env_steps=t, states=N_STATE, visited_states=int((n_visit.sum(1) > 0).sum()), visited_pairs=int((n_visit > 0).sum()),
              table_bytes=int(Q.nbytes), best_val_score=best, config_params=cfg)
    print(f"[tabq {name}] done: {m.record['wall_s']:.0f}s, cpu package energy {m.record.get('cpu_package_J')} J", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], int(sys.argv[3]), int(float(sys.argv[4])), sys.argv[5])
