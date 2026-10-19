"""Vectorised 2D training environment: many arenas x 4 robots, one shared policy (Alternative 1).

Every robot is one sub-environment from the policy's point of view; all robots in all worlds are stepped
together in numpy. It mirrors the Gazebo robot: unicycle kinematics with the real speed limits, a 360-beam
lidar (0.15-2.0 m, mounted 3.3 cm behind the axle, pooled to 36 beams), other robots visible to the lidar only
as their 7 cm lidar head, and camera-derived goal / other-robot positions.

mode="train": continuing task. Reaching the goal draws a new goal in place; a collision respawns the robot at a
              random free spot; layouts are redrawn every `layout_life` steps.
mode="eval":  the real task protocol. Fixed layouts, starts and goals; a robot stops when it reaches its goal
              (and stays as an obstacle) or collides; the episode ends when all four are done or on timeout.
"""
import math

import gymnasium as gym
import numpy as np
import shield as SH
from stable_baselines3.common.vec_env import VecEnv

DT, VMAX, WMAX = 0.1, 0.20, 2.5
R_ROBOT, HEAD_R, LIDAR_X = 0.095, 0.035, -0.03275
ARENA_W, ARENA_H = 4.0, 3.0
N_AG, N_OTH, N_POOL = 4, 3, 36
RANGE_MIN, RANGE_MAX = 0.15, 2.0
GOAL_TOL, GOAL_MIN_DIST = 0.12, 1.0
OBS_DIM = N_POOL + 3 + 2 + 4 * N_OTH + 2 + N_AG   # lidar, goal, own v/w, others (rel pos + rel vel, fixed ID order), previous action, own-ID one-hot
OTHERS = np.array([[b for b in range(N_AG) if b != a] for a in range(N_AG)])
_STEP = 2 * math.pi / 359  # Gazebo scan: 360 beams over [-pi, pi]
BEAM_SUBSET = (1, 3, 5, 7, 9)  # beams used inside each 10-beam group; deploy.py reads exactly the same ones
RAY_IDX = np.array([[10 * j + k for k in BEAM_SUBSET] for j in range(N_POOL)])
RAY_ANG = (-math.pi + RAY_IDX * _STEP).reshape(-1).astype(np.float32)
RAYS_PER_GROUP = RAY_IDX.shape[1]
F32 = np.float32
from actions import CV, CW, N_ACT  # shared discrete (speed, turn-rate) table


def obb_dist(px, py, ob):
    """Distance from points (...,1 or K) to rectangles ob[..., 5] = (x, y, sx, sy, yaw); 0 inside. Broadcasts."""
    c, s = np.cos(ob[..., 4]), np.sin(ob[..., 4])
    dx, dy = px - ob[..., 0], py - ob[..., 1]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return np.hypot(np.maximum(np.abs(lx) - ob[..., 2] / 2, 0), np.maximum(np.abs(ly) - ob[..., 3] / 2, 0))


class ArenaVecEnv(VecEnv):
    def __init__(self, pool, n_worlds=64, seed=0, mode="train", max_steps=600, layout_life=3000, randomize=True, eval_idx=None, shield=True):
        self.pool, self.mode, self.max_steps, self.layout_life, self.randomize = pool, mode, max_steps, layout_life, randomize
        self.W = len(eval_idx) if mode == "eval" else n_worlds
        self.shield, self.overrides = shield, 0.0
        self.rng = np.random.default_rng(seed)
        self.n_layouts = len(pool["obst"])
        super().__init__(self.W * N_AG, gym.spaces.Box(-1.0, 1.0, (OBS_DIM,), F32), gym.spaces.Discrete(N_ACT))
        W = self.W
        self.lay = np.zeros(W, int)
        self.obst = np.zeros((W, pool["obst"].shape[1], 5), F32)
        self.pos, self.yaw = np.zeros((W, N_AG, 2), F32), np.zeros((W, N_AG), F32)
        self.v, self.w = np.zeros((W, N_AG), F32), np.zeros((W, N_AG), F32)
        self.goal, self.prev_act = np.zeros((W, N_AG, 2), F32), np.zeros((W, N_AG, 2), F32)
        self.cmd_prev = np.zeros((W, N_AG, 2), F32)
        self.t, self.age = np.zeros((W, N_AG), int), np.zeros(W, int)
        self.d_prev = np.zeros((W, N_AG), F32)
        self.ep_ret = np.zeros((W, N_AG), F32)
        self.speed_scale, self.lag, self.delay = np.ones((W, N_AG), F32), np.ones((W, N_AG), F32), np.zeros((W, N_AG), bool)
        self.active = np.ones((W, N_AG), bool)
        self.outcome = np.zeros((W, N_AG), int)       # eval: 0 running, 1 reached goal, 2 collided, 3 timed out
        self.cause = np.zeros((W, N_AG), int)         # eval: collision cause 1 obstacle, 2 wall, 3 other robot
        self.finish_t = np.zeros((W, N_AG), int)
        self._eval_idx = np.asarray(eval_idx) if eval_idx is not None else None
        self.actions = None
        self.reset()

    # ------------------------------------------------------------------ world / agent resets
    def _load_layout(self, w, li):
        self.lay[w] = li
        self.obst[w] = self.pool["obst"][li]
        self.age[w] = 0

    def _free_cells(self, w):
        return np.flatnonzero(self.pool["free"][self.lay[w]])

    @staticmethod
    def _cell_xy(flat, rng_jitter=None):
        r, c = np.divmod(flat, 200)
        return (-ARENA_W / 2 + (c + 0.5) * 0.02, -ARENA_H / 2 + (r + 0.5) * 0.02)

    def _sample_free(self, w, avoid, min_d, tries=30):
        cells = self._free_cells(w)
        for _ in range(tries):
            x, y = self._cell_xy(self.rng.choice(cells))
            if all(math.hypot(x - ax, y - ay) >= min_d for ax, ay in avoid):
                return x, y
        return x, y

    def _randomize_agent(self, w, a):
        if self.randomize and self.mode == "train":
            self.speed_scale[w, a] = self.rng.uniform(0.9, 1.1)
            self.lag[w, a] = self.rng.uniform(0.5, 1.0)
            self.delay[w, a] = self.rng.random() < 0.3
        self.t[w, a], self.ep_ret[w, a] = 0, 0.0
        self.prev_act[w, a], self.cmd_prev[w, a] = 0.0, 0.0

    def _new_goal(self, w, a):
        others = [tuple(self.pos[w, b]) for b in range(N_AG) if b != a] + [tuple(self.goal[w, b]) for b in range(N_AG) if b != a]
        for _ in range(20):
            g = self._sample_free(w, [(self.pos[w, a, 0], self.pos[w, a, 1])] * 0 + others, 0.4)
            if math.hypot(g[0] - self.pos[w, a, 0], g[1] - self.pos[w, a, 1]) >= GOAL_MIN_DIST:
                break
        self.goal[w, a] = g
        self.d_prev[w, a] = math.hypot(g[0] - self.pos[w, a, 0], g[1] - self.pos[w, a, 1])

    def _respawn(self, w, a):
        avoid = [tuple(self.pos[w, b]) for b in range(N_AG) if b != a]
        self.pos[w, a] = self._sample_free(w, avoid, 0.45)
        self.yaw[w, a] = self.rng.uniform(-math.pi, math.pi)
        self.v[w, a] = self.w[w, a] = 0.0

    def _reset_world(self, w, li=None):
        li = int(self.rng.integers(self.n_layouts)) if li is None else li
        self._load_layout(w, li)
        self.pos[w] = self.pool["starts"][li][:, :2]
        self.yaw[w] = self.pool["starts"][li][:, 2]
        self.goal[w] = self.pool["goals"][li]
        self.v[w] = self.w[w] = 0.0
        self.active[w], self.outcome[w], self.finish_t[w] = True, 0, 0
        for a in range(N_AG):
            self._randomize_agent(w, a)
            self.d_prev[w, a] = math.hypot(*(self.goal[w, a] - self.pos[w, a]))

    # ------------------------------------------------------------------ observations
    def _obs(self, wi, ai):
        """Observation for agent (wi[k], ai[k]) pairs -> (P, OBS_DIM)."""
        P = len(wi)
        pos, yaw = self.pos[wi, ai], self.yaw[wi, ai]
        c, s = np.cos(yaw), np.sin(yaw)
        o = pos + np.stack([c * LIDAR_X, s * LIDAR_X], -1)
        ang = yaw[:, None] + RAY_ANG[None, :]
        dx, dy = np.cos(ang), np.sin(ang)
        inf = F32(np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            tmin_all = np.full(dx.shape, inf, F32)
            for b, d, oo in ((ARENA_W / 2, dx, o[:, 0]), (-ARENA_W / 2, dx, o[:, 0]), (ARENA_H / 2, dy, o[:, 1]), (-ARENA_H / 2, dy, o[:, 1])):
                t = (b - oo[:, None]) / d
                tmin_all = np.minimum(tmin_all, np.where(t > 0, t, inf))
            ob = self.obst[wi]                                   # (P,K,5)
            oc, os_ = np.cos(ob[..., 4]), np.sin(ob[..., 4])
            rox = oc * (o[:, None, 0] - ob[..., 0]) + os_ * (o[:, None, 1] - ob[..., 1])
            roy = -os_ * (o[:, None, 0] - ob[..., 0]) + oc * (o[:, None, 1] - ob[..., 1])
            rdx = oc[:, None, :] * dx[:, :, None] + os_[:, None, :] * dy[:, :, None]
            rdy = -os_[:, None, :] * dx[:, :, None] + oc[:, None, :] * dy[:, :, None]
            hx, hy = (ob[..., 2] / 2)[:, None, :], (ob[..., 3] / 2)[:, None, :]
            t1x, t2x = (-hx - rox[:, None, :]) / rdx, (hx - rox[:, None, :]) / rdx
            t1y, t2y = (-hy - roy[:, None, :]) / rdy, (hy - roy[:, None, :]) / rdy
            tn = np.fmax(np.fmin(t1x, t2x), np.fmin(t1y, t2y))
            tf = np.fmin(np.fmax(t1x, t2x), np.fmax(t1y, t2y))
            hit = (tf >= tn) & (tf > 0)
            t_ob = np.where(hit, np.maximum(tn, 0), inf).min(-1)
            tmin_all = np.minimum(tmin_all, t_ob)
            oth = OTHERS[ai]                                      # (P,3) other robots' indices
            opos, oyaw = self.pos[wi[:, None], oth], self.yaw[wi[:, None], oth]
            head = opos + np.stack([np.cos(oyaw) * LIDAR_X, np.sin(oyaw) * LIDAR_X], -1)
            ocx, ocy = o[:, None, 0] - head[..., 0], o[:, None, 1] - head[..., 1]
            bb = dx[:, :, None] * ocx[:, None, :] + dy[:, :, None] * ocy[:, None, :]
            cc = (ocx ** 2 + ocy ** 2 - HEAD_R ** 2)[:, None, :]
            disc = bb * bb - cc
            th = -bb - np.sqrt(np.maximum(disc, 0))
            t_h = np.where((disc >= 0) & (th > 0), th, inf).min(-1)
            r = np.minimum(tmin_all, t_h)
        n = np.where(r < RANGE_MIN, 0.0, np.where(r > RANGE_MAX, 1.0, r / RANGE_MAX)).astype(F32)
        if self.randomize and self.mode == "train":
            n = np.clip(n + self.rng.normal(0, 0.01 / RANGE_MAX, n.shape).astype(F32), 0, 1)
            n = np.where(self.rng.random(n.shape) < 0.005, F32(1.0), n)
        self._last_n = n  # raw normalised ranges, kept for the parity test
        lidar = n.reshape(P, N_POOL, RAYS_PER_GROUP).min(-1)

        g = self.goal[wi, ai] - pos
        d = np.hypot(g[:, 0], g[:, 1])
        gl = np.stack([np.minimum(d, 4.0) / 4.0, (c * g[:, 0] + s * g[:, 1]) / np.maximum(d, 1e-6), (-s * g[:, 0] + c * g[:, 1]) / np.maximum(d, 1e-6)], -1)
        rel = opos - pos[:, None, :]
        ov = self.v[wi[:, None], oth]
        relv = np.stack([ov * np.cos(oyaw), ov * np.sin(oyaw)], -1) - np.stack([self.v[wi, ai] * c, self.v[wi, ai] * s], -1)[:, None, :]
        rot = lambda q: np.stack([c[:, None] * q[..., 0] + s[:, None] * q[..., 1], -s[:, None] * q[..., 0] + c[:, None] * q[..., 1]], -1)
        relr, relvr = rot(rel), rot(relv)
        if self.randomize and self.mode == "train":
            relr = relr + self.rng.normal(0, 0.01, relr.shape).astype(F32)
            relvr = relvr + self.rng.normal(0, 0.03, relvr.shape).astype(F32)
        feat = np.concatenate([np.clip(relr / 3.0, -1, 1), np.clip(relvr / 0.4, -1, 1)], -1)
        others = feat.reshape(P, -1)  # fixed slots: other robots in ascending ID order, so a slot always means the same robot
        own = np.stack([self.v[wi, ai] / VMAX, self.w[wi, ai] / WMAX], -1)
        return np.concatenate([lidar, gl, own, others, self.prev_act[wi, ai], np.eye(N_AG, dtype=F32)[ai]], -1).astype(F32)

    def _all_obs(self):
        wi, ai = np.meshgrid(np.arange(self.W), np.arange(N_AG), indexing="ij")
        return self._obs(wi.ravel(), ai.ravel()).reshape(self.W * N_AG, OBS_DIM)

    # ------------------------------------------------------------------ gym API
    def reset(self):
        for w in range(self.W):
            self._reset_world(w, int(self._eval_idx[w]) if self.mode == "eval" else None)
        return self._all_obs()

    def step_async(self, actions):
        """actions: int indices (N,) into the discrete (v, w) table, or continuous (N,2) = (v/VMAX*2-1, w/WMAX) which is snapped to the nearest entry."""
        a = np.asarray(actions)
        if a.ndim == 2:
            v, w = (np.clip(a[:, 0], -1, 1) + 1) / 2 * VMAX, np.clip(a[:, 1], -1, 1) * WMAX
            a = ((v[:, None] - CV[None]) ** 2 / VMAX ** 2 + (w[:, None] - CW[None]) ** 2 / WMAX ** 2).argmin(1)
        self.actions = a.reshape(self.W, N_AG).astype(int)
        if self.shield:  # inter-robot safety layer: same information the overhead camera provides on the real system
            W, N = self.W, self.W * N_AG
            widx, oth = np.repeat(np.arange(W), N_AG), OTHERS[np.tile(np.arange(N_AG), W)]
            ov, oy = self.v[widx[:, None], oth], self.yaw[widx[:, None], oth]
            new = SH.apply(self.pos.reshape(N, 2), self.yaw.reshape(N), self.pos[widx[:, None], oth], np.stack([ov * np.cos(oy), ov * np.sin(oy)], -1), self.actions.reshape(N))
            self.overrides = float((new != self.actions.reshape(N)).mean())
            self.actions = new.reshape(W, N_AG)

    def step_wait(self):
        idx = self.actions
        a = np.stack([CV[idx] / VMAX * 2 - 1, CW[idx] / WMAX], -1)   # normalised (v, w) of the chosen motion, as used in obs/reward
        cmd = np.stack([CV[idx], CW[idx]], -1) * self.speed_scale[..., None]
        applied = np.where(self.delay[..., None], self.cmd_prev, cmd)
        self.cmd_prev = cmd
        run = self.active
        self.v = np.where(run, self.v + (applied[..., 0] - self.v) * self.lag, 0.0).astype(F32)
        self.w = np.where(run, self.w + (applied[..., 1] - self.w) * self.lag, 0.0).astype(F32)
        mid = self.yaw + self.w * DT / 2
        self.pos = self.pos + np.stack([np.cos(mid), np.sin(mid)], -1) * (self.v * DT)[..., None]
        self.yaw = ((self.yaw + self.w * DT + math.pi) % (2 * math.pi) - math.pi).astype(F32)
        self.t += run
        self.age += 1

        px, py = self.pos[..., 0:1], self.pos[..., 1:2]
        d_ob = obb_dist(px, py, self.obst[:, None, :, :]).min(-1)              # (W,4)
        d_wall = np.minimum(ARENA_W / 2 - np.abs(self.pos[..., 0]), ARENA_H / 2 - np.abs(self.pos[..., 1]))
        pd = np.hypot(self.pos[:, :, None, 0] - self.pos[:, None, :, 0], self.pos[:, :, None, 1] - self.pos[:, None, :, 1])
        pd[:, np.arange(N_AG), np.arange(N_AG)] = np.inf
        d_rob = pd.min(-1) - 2 * R_ROBOT                                         # gap between the two chassis surfaces (each radius R)
        clearance = np.minimum(np.minimum(d_ob, d_wall) - R_ROBOT, d_rob)   # metres between chassis and nearest thing; < 0 = contact
        collided = run & (clearance < 0)
        d_goal = np.hypot(*(self.goal - self.pos).transpose(2, 0, 1))
        reached = run & ~collided & (d_goal < GOAL_TOL)
        timeout = run & ~collided & ~reached & (self.t >= self.max_steps)

        # early-warning shaping: closest approach to each other robot within the next 3 s on current velocities
        vel = np.stack([self.v * np.cos(self.yaw), self.v * np.sin(self.yaw)], -1)
        rp = self.pos[:, None, :, :] - self.pos[:, :, None, :]            # position of b relative to a
        ru = vel[:, None, :, :] - vel[:, :, None, :]
        uu, pu = (ru * ru).sum(-1), (rp * ru).sum(-1)
        tstar = np.clip(-pu / np.maximum(uu, 1e-6), 0, 3.0)
        cpa = np.linalg.norm(rp + ru * tstar[..., None], axis=-1)
        risk = np.clip((0.45 - cpa) / (0.45 - 2 * R_ROBOT), 0, 1) * ((uu > 1e-4) & (pu < 0))
        risk[:, np.arange(N_AG), np.arange(N_AG)] = 0
        ttc_pen = 0.3 * risk.max(-1)
        # traffic convention (keep right): when another robot ahead is on a collision course, penalise not turning right,
        # so head-on meetings resolve instead of both robots averaging "left" and "right" into "straight"
        bearing = np.arctan2(rp[..., 1], rp[..., 0]) - self.yaw[:, :, None]
        ahead = np.abs((bearing + np.pi) % (2 * np.pi) - np.pi) < math.radians(70)
        right_pen = 0.15 * (risk * ahead).max(-1) * (1 + a[..., 1]) / 2        # a[...,1] = +1 is full left turn, -1 full right
        rew = 10.0 * (self.d_prev - d_goal) - 0.01 - 0.15 * np.clip((0.3 - clearance) / 0.3, 0, 1) - ttc_pen - right_pen - 1.0 * np.clip((0.10 - clearance) / 0.10, 0, 1)  # ttc_pen: predicted-collision shaping; 2nd term: stay out of the lidar's 0.15 m blind zone - 0.01 * np.abs(a[..., 1] - self.prev_act[..., 1])
        rew = np.where(reached, rew + 10.0, np.where(collided, -15.0, rew)) * run
        self.d_prev = d_goal.astype(F32)
        self.prev_act = a.copy()
        self.ep_ret += rew
        done = collided | reached | timeout

        infos = [{} for _ in range(self.W * N_AG)]
        if self.mode == "eval":
            self.cause = np.where(run & collided, np.where(d_rob < 0, 3, np.where(d_ob - R_ROBOT < 0, 1, 2)), self.cause)
            self.outcome = np.where(run & collided, 2, np.where(run & reached, 1, self.outcome))
            self.finish_t = np.where(run & (collided | reached), self.t, self.finish_t)
            self.active = run & ~collided & ~reached
            ended = (~self.active).all(1) | (self.t.max(1) >= self.max_steps)
            self.outcome = np.where(self.active & ended[:, None], 3, self.outcome)
            obs = self._all_obs()
            dones = np.repeat(ended, N_AG)
            return obs, rew.reshape(-1).astype(F32), dones, infos

        redraw = (self.age >= self.layout_life)
        term_obs = self._all_obs() if (done.any() or redraw.any()) else None
        flat_done, flat_trunc = done.reshape(-1), (timeout | np.repeat(redraw[:, None], N_AG, 1)).reshape(-1)
        for w, ag in zip(*np.nonzero(done | redraw[:, None])):
            n = w * N_AG + ag
            infos[n] = {"terminal_observation": term_obs[n], "TimeLimit.truncated": bool(flat_trunc[n] and not (collided[w, ag] or reached[w, ag])),
                        "is_success": bool(reached[w, ag]), "collision": bool(collided[w, ag]), "episode": {"r": float(self.ep_ret[w, ag]), "l": int(self.t[w, ag])}}
        dones = flat_done | np.repeat(redraw, N_AG)
        for w in np.nonzero(redraw)[0]:
            self._reset_world(w)
        for w, ag in zip(*np.nonzero((done | timeout) & ~redraw[:, None])):
            if collided[w, ag]:
                self._respawn(w, ag)
            self._new_goal(w, ag)
            self._randomize_agent(w, ag)
        return self._all_obs(), rew.reshape(-1).astype(F32), dones, infos

    # ------------------------------------------------------------------ VecEnv boilerplate
    def close(self): pass
    def get_attr(self, attr_name, indices=None): return [getattr(self, attr_name, None)] * self.num_envs
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *args, indices=None, **kwargs): return [None] * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def get_images(self): return []
    def seed(self, seed=None): self.rng = np.random.default_rng(seed); return [seed] * self.num_envs
