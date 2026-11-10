"""Tabular Q-learning ("classic" RL) on a discretised version of the same 59-number observation.

    .venv/bin/python benchmark/tabq.py train CONFIG SEED STEPS NAME     # writes runs/benchmark/tabq/NAME/{best,latest}.npz, log.csv

State (6,480 states): goal bearing (8 bins) x goal distance (3: <0.4, <1.2, >=1.2 m) x lidar minimum in front (-30..30 deg),
front-left (30..90) and front-right (-90..-30), 3 levels each (<0.35, <0.8, >=0.8 m) x nearest other robot within 1 m
(none / ahead / left / right / behind) x own speed (2: below / above half of max). Actions: the same 45 commands.
Learning: one shared table for all 256 robots of 64 training worlds (seeds 1000-2899), env reward unchanged, safety layer
on (part of the environment, as for PPO). Duplicate (state, action) updates within one step are averaged.
Every 1M steps: protocol evaluation on validation layouts (seeds 2900-2999); the best (success - 0.5 collision) is kept."""
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


def ctrl_from(path):
    Q = np.load(path)["Q"]
    return lambda obs, env: Q[discretize(obs)].argmax(1)


def train(cfg_name, seed, steps, name):
    cfg = CONFIGS[cfg_name]
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
    log = open(out / "log.csv", "w")
    log.write("steps,wall_s,eps,val_success,val_collision,val_timeout,val_score,visited_states,train_episodes,train_success_rate,train_collision_rate\n")
    with Meter(f"tabq_train_{name}", kind="train", method="tabq", config=cfg_name, seed=seed) as m:
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
            target = r + cfg["gamma"] * Q[s_boot].max(1) * (~terminal)
            td = target - Q[s, a]
            flat = s * E.N_ACT + a
            sums = np.bincount(flat, weights=td, minlength=Q.size)
            cnt = np.bincount(flat, minlength=Q.size)
            hit = np.nonzero(cnt)[0]
            n_visit.ravel()[hit] += cnt[hit]
            alpha = np.full(len(hit), cfg["alpha_min"]) if cfg["alpha"] == "const" else np.maximum(cfg["alpha_min"], 1.0 / n_visit.ravel()[hit] ** 0.6)
            Q.ravel()[hit] += (alpha * sums[hit] / cnt[hit]).astype(np.float32)
            s = s_next
            t += N
            if t >= next_eval or t >= steps:
                next_eval += 1_000_000
                np.savez_compressed(out / "latest.npz", Q=Q, n_visit=n_visit)
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
                    np.savez_compressed(out / "best.npz", Q=Q, n_visit=n_visit)
                    (out / "best.json").write_text(json.dumps({"steps": t, **{k: sm[k] for k in ("success", "collision", "timeout")}}))
        m.add(env_steps=t, states=N_STATE, visited_states=int((n_visit.sum(1) > 0).sum()), best_val_score=best)
    print(f"[tabq {name}] done: {m.record['wall_s']:.0f}s, cpu package energy {m.record.get('cpu_package_J')} J", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], int(sys.argv[3]), int(float(sys.argv[4])), sys.argv[5])
