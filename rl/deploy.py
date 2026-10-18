"""Deployment side: numpy policy + observation built from *sensor-level* inputs (raw 360-beam lidar, camera poses,
odometry). Must stay in lock-step with ArenaVecEnv._obs; tests/test_obs_parity.py enforces that."""
import math

import numpy as np

DT, VMAX, WMAX = 0.1, 0.20, 2.5
RANGE_MIN, RANGE_MAX, N_POOL = 0.15, 2.0, 36
BEAM_SUBSET = (1, 3, 5, 7, 9)  # must equal arena_env.BEAM_SUBSET: the policy was trained on exactly these beams


class Policy:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.layers = [(d[f"weight{i}"], d[f"bias{i}"]) for i in range(3)]

    def __call__(self, obs):
        x = np.atleast_2d(obs).astype(np.float32)
        for k, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if k < 2:
                x = np.tanh(x)
        return np.clip(x, -1, 1)


def lidar_features(scan360):
    """360 raw ranges (inf / -inf / nan allowed) -> 36 values in [0,1]: min over the training beam subset of each group."""
    r = np.asarray(scan360, np.float64)
    n = np.where(np.isnan(r) | (r > RANGE_MAX), 1.0, np.where(r < RANGE_MIN, 0.0, r / RANGE_MAX))  # +inf / out of range -> 1, -inf / too close -> 0
    return n.reshape(N_POOL, 10)[:, list(BEAM_SUBSET)].min(-1)


def obs_from_sensors(scan360, own_xy, own_yaw, goal_xy, others_xy, own_vel_xy, others_vel_xy, v, w, prev_act):
    """own_xy/own_yaw/goal_xy/others_xy come from the overhead camera, *_vel_xy are world-frame velocities from
    differencing consecutive camera frames; v, w from wheel odometry."""
    c, s = math.cos(own_yaw), math.sin(own_yaw)
    rot = lambda q: np.stack([c * q[:, 0] + s * q[:, 1], -s * q[:, 0] + c * q[:, 1]], -1)
    g = np.asarray(goal_xy) - np.asarray(own_xy)
    d = math.hypot(*g)
    goal = [min(d, 4.0) / 4.0, (c * g[0] + s * g[1]) / max(d, 1e-6), (-s * g[0] + c * g[1]) / max(d, 1e-6)]
    relr = rot(np.asarray(others_xy) - np.asarray(own_xy))
    relvr = rot(np.asarray(others_vel_xy) - np.asarray(own_vel_xy))
    feat = np.concatenate([np.clip(relr / 3.0, -1, 1), np.clip(relvr / 0.4, -1, 1)], -1)[np.argsort(np.hypot(relr[:, 0], relr[:, 1]))]
    return np.concatenate([lidar_features(scan360), goal, [v / VMAX, w / WMAX], feat.ravel(), prev_act]).astype(np.float32)


def action_to_cmd(a):
    return (a[0] + 1) / 2 * VMAX, a[1] * WMAX
