"""Protocol evaluation in the 2D env: fixed layouts/starts/goals, robots stop at their goal or on collision."""
import numpy as np

import arena_env as E


def run_eval(pool, act_fn, n_layouts=100, seed=0, max_steps=900):
    """act_fn(obs[N,49]) -> actions[N,2]. Returns per-robot rates over n_layouts x 4 robots."""
    env = E.ArenaVecEnv(pool, mode="eval", eval_idx=np.arange(n_layouts), max_steps=max_steps, seed=seed, randomize=False)
    obs = env.reset()
    for _ in range(max_steps):
        env.step_async(act_fn(obs))
        obs, _, dones, _ = env.step_wait()
        if not env.active.any():
            break
    o = env.outcome
    ok = o == 1
    cz = env.cause[o == 2]
    return {"coll_obstacle": float((cz == 1).sum() / o.size), "coll_wall": float((cz == 2).sum() / o.size), "coll_robot": float((cz == 3).sum() / o.size), "success": float(ok.mean()), "collision": float((o == 2).mean()), "timeout": float((o == 3).mean()),
            "mean_time_s": float(env.finish_t[ok].mean() * E.DT) if ok.any() else float("nan"),
            "all_four_success": float(ok.all(1).mean())}


def goto_goal(obs):
    """Naive baseline: turn toward the goal, drive when roughly aligned, ignore everything else."""
    bearing = np.arctan2(obs[:, E.N_POOL + 2], obs[:, E.N_POOL + 1])
    a = np.zeros((len(obs), 2), np.float32)
    a[:, 1] = np.clip(bearing * 2, -1, 1)
    a[:, 0] = np.where(np.abs(bearing) < 0.6, 1.0, -1.0)
    return a
