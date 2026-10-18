"""Inter-robot safety layer (rule-based, camera-level information only).

For the motion the policy picked, predict the next 2 s of own motion (constant command) against the other robots (constant
velocity, as measured by the overhead camera). If the gap between chassis would drop below MARGIN, swap in the nearest safe
motion (closest in speed/turn-rate); if none is safe, take the one with the largest predicted gap.
The policy (RL) still owns goal-seeking and all obstacle avoidance; this layer only vetoes robot-robot collision courses."""
import numpy as np

from actions import CV, CW, N_ACT, VMAX, WMAX

R_ROBOT, HORIZON, STEPS, MARGIN = 0.095, 2.0, 10, 0.06
_T = (np.arange(STEPS) + 1) * HORIZON / STEPS
_W = np.where(np.abs(CW) < 1e-3, 1e-3, CW)


def clearance(pos, yaw, opos, ovel):
    """pos (N,2), yaw (N,), opos/ovel (N,K,2) -> (N, N_ACT): worst predicted gap between chassis surfaces over the horizon."""
    th = yaw[:, None, None] + _W[None, :, None] * _T
    x = pos[:, 0, None, None] + CV[None, :, None] / _W[None, :, None] * (np.sin(th) - np.sin(yaw)[:, None, None])
    y = pos[:, 1, None, None] - CV[None, :, None] / _W[None, :, None] * (np.cos(th) - np.cos(yaw)[:, None, None])
    ox = opos[:, None, None, :, 0] + ovel[:, None, None, :, 0] * _T[None, None, :, None]
    oy = opos[:, None, None, :, 1] + ovel[:, None, None, :, 1] * _T[None, None, :, None]
    return (np.hypot(x[..., None] - ox, y[..., None] - oy).min(-1) - 2 * R_ROBOT).min(-1)


def apply(pos, yaw, opos, ovel, idx):
    """Return the (possibly replaced) action indices, shape (N,)."""
    cl = clearance(pos, yaw, opos, ovel)
    safe = cl > MARGIN
    keep = safe[np.arange(len(idx)), idx]
    dist = (CV[None, :] - CV[idx][:, None]) ** 2 / VMAX ** 2 + (CW[None, :] - CW[idx][:, None]) ** 2 / WMAX ** 2
    dist = np.where(safe, dist, np.inf)
    alt = np.where(np.isfinite(dist.min(1)), dist.argmin(1), cl.argmax(1))
    return np.where(keep, idx, alt)
