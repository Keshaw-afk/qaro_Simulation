"""Dynamic-window-style reference controller using the simulator's true state. Diagnostic yardstick (is the env solvable?)
and potential demonstrator; NOT the deployed policy."""

import numpy as np

import arena_env as E

CV, CW = E.CV.astype(np.float64), E.CW.astype(np.float64)   # the same discrete motions the policy chooses from


def dwa_actions(env, horizon=2.0, steps=10, margin=0.03):
    W, N = env.W, env.W * E.N_AG
    widx, aidx = np.repeat(np.arange(W), E.N_AG), np.tile(np.arange(E.N_AG), W)
    pos, yaw, goal = env.pos.reshape(N, 2), env.yaw.reshape(N), env.goal.reshape(N, 2)
    t = (np.arange(steps) + 1) * horizon / steps
    w_eff = np.where(np.abs(CW) < 1e-3, 1e-3, CW)[None, :, None]   # near-zero turn rate stand-in; must drive the heading too
    th = yaw[:, None, None] + w_eff * t
    x = pos[:, 0, None, None] + CV[None, :, None] / w_eff * (np.sin(th) - np.sin(yaw)[:, None, None])
    y = pos[:, 1, None, None] - CV[None, :, None] / w_eff * (np.cos(th) - np.cos(yaw)[:, None, None])
    d_ob = E.obb_dist(x[..., None], y[..., None], env.obst[widx][:, None, None, :, :]).min(-1) - E.R_ROBOT
    d_wall = np.minimum(E.ARENA_W / 2 - np.abs(x), E.ARENA_H / 2 - np.abs(y)) - E.R_ROBOT
    oth = E.OTHERS[aidx]
    opos, oyaw, ov = env.pos[widx[:, None], oth], env.yaw[widx[:, None], oth], env.v[widx[:, None], oth]
    ox = opos[:, None, None, :, 0] + (ov * np.cos(oyaw))[:, None, None, :] * t[None, None, :, None]
    oy = opos[:, None, None, :, 1] + (ov * np.sin(oyaw))[:, None, None, :] * t[None, None, :, None]
    d_rob = np.hypot(x[..., None] - ox, y[..., None] - oy).min(-1) - 2 * E.R_ROBOT
    clear = np.minimum(np.minimum(d_ob, d_wall), d_rob).min(-1)                    # (N, C) worst clearance over the horizon
    d0 = np.hypot(*(goal - pos).T)[:, None]
    d_end = np.hypot(goal[:, 0, None] - x[..., -1], goal[:, 1, None] - y[..., -1])
    head = np.cos(np.arctan2(goal[:, 1, None] - y[..., -1], goal[:, 0, None] - x[..., -1]) - th[..., -1])
    score = 2.0 * (d0 - d_end) / (0.2 * horizon) + 0.3 * head + np.minimum(clear, 0.4) / 0.4 + 0.2 * CV / 0.2 - 0.05 * CW / 2.0
    score = np.where(clear > margin, score, -1e6 + clear)
    best = score.argmax(1)
    return best  # index into the shared (v, w) table


def run_expert_eval(pool, n_layouts=100, max_steps=900):
    env = E.ArenaVecEnv(pool, mode="eval", eval_idx=np.arange(n_layouts), max_steps=max_steps, seed=0, randomize=False)
    env.reset()
    for _ in range(max_steps):
        env.step_async(dwa_actions(env)); env.step_wait()
        if not env.active.any():
            break
    o, cz = env.outcome, env.cause[env.outcome == 2]
    return {"success": float((o == 1).mean()), "collision": float((o == 2).mean()), "timeout": float((o == 3).mean()),
            "coll_obstacle": float((cz == 1).sum() / o.size), "coll_wall": float((cz == 2).sum() / o.size), "coll_robot": float((cz == 3).sum() / o.size),
            "all_four_success": float((o == 1).all(1).mean()), "mean_time_s": float(env.finish_t[o == 1].mean() * E.DT)}
