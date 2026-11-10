"""Correctness tests for the benchmark code (no simulator). Run: .venv/bin/python benchmark/test_benchmark.py
Each test checks a piece of logic whose silent failure would bias the benchmark."""
import math
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import E  # noqa: E402
import rule  # noqa: E402
import tabq  # noqa: E402


def blank_obs(goal_dist=2.0, goal_bearing=0.0):
    """One observation: nothing seen by the lidar, others far away (clipped slots), goal at (dist, bearing)."""
    o = np.zeros(E.OBS_DIM, np.float32)
    o[:E.N_POOL] = 1.0
    o[36], o[37], o[38] = min(goal_dist, 4) / 4, math.cos(goal_bearing), math.sin(goal_bearing)
    o[41:53] = np.tile([1.0, 1.0, 0.0, 0.0], 3)  # three other robots >= 3 m away, not moving
    o[55] = 1.0
    return o


def sector_of(angle_deg):
    return int((angle_deg + 180) // 10)


def test_discretize():
    o = blank_obs(2.0, 0.0)
    s0 = tabq.discretize(o[None])[0]
    o2 = o.copy(); o2[sector_of(0)] = 0.3 / 2.0                       # obstacle 0.3 m straight ahead -> front level 0
    s2 = tabq.discretize(o2[None])[0]
    dec = lambda s: (s // 2 // 5 // 27 // 3, s // 2 // 5 // 27 % 3, s // 2 // 5 % 27, s // 2 % 5, s % 2)
    b, db, lid, rb, vb = dec(s0)
    assert db == 2 and lid == 26 and rb == 0 and vb == 0, dec(s0)
    assert b == 4, b                                                    # bearing 0 lies in bin 4 of [-pi, pi)
    assert dec(s2)[2] == 0 * 9 + 2 * 3 + 2, dec(s2)                   # front 0, left 2, right 2
    o3 = o.copy(); o3[sector_of(60)] = 0.5 / 2.0                       # obstacle at +60 deg (left) at 0.5 m -> front-left level 1
    assert dec(tabq.discretize(o3[None])[0])[2] == 2 * 9 + 1 * 3 + 2
    o4 = o.copy(); o4[41:45] = [0.5 / 3, 0.0, 0, 0]                    # robot 0.5 m ahead -> "ahead"
    assert dec(tabq.discretize(o4[None])[0])[3] == 1
    o5 = o.copy(); o5[41:45] = [0.0, 0.6 / 3, 0, 0]                    # robot 0.6 m to the left (+y) -> "left"
    assert dec(tabq.discretize(o5[None])[0])[3] == 2
    assert tabq.discretize(np.stack([o, o2, o3]))[2] == tabq.discretize(o3[None])[0]  # batched == single
    print("discretize ok")


def test_discretize_fine():
    o = blank_obs(2.0, 0.0)
    s0 = tabq.discretize_fine(o[None])[0]
    assert 0 <= s0 < tabq.N_FINE
    vb, rb = s0 % 2, s0 // 2 % 9
    lid, db, b = s0 // 18 % 243, s0 // 18 // 243 % 3, s0 // 18 // 243 // 3
    assert (b, db, lid, rb, vb) == (6, 2, 242, 0, 0), (b, db, lid, rb, vb)   # bearing 0 -> bin 6 of 12; all lidar regions free
    o2 = o.copy(); o2[sector_of(0)] = 0.3 / 2.0                               # front blocked -> first lidar digit 0
    assert tabq.discretize_fine(o2[None])[0] // 18 % 243 == 0 * 81 + 2 * 27 + 2 * 9 + 2 * 3 + 2
    o3 = o.copy(); o3[sector_of(-80)] = 0.5 / 2.0                             # right (-110..-60) at 0.5 m -> last digit 1
    assert tabq.discretize_fine(o3[None])[0] // 18 % 243 == 242 - 1
    o4 = o.copy(); o4[41:45] = [0.5 / 3, 0.0, -0.2 / 0.4, 0.0]               # robot ahead, closing
    o5 = o.copy(); o5[41:45] = [0.5 / 3, 0.0, 0.2 / 0.4, 0.0]                # robot ahead, moving away
    assert tabq.discretize_fine(o4[None])[0] // 2 % 9 == 1 + 0 * 2 + 1
    assert tabq.discretize_fine(o5[None])[0] // 2 % 9 == 1 + 0 * 2 + 0
    print("discretize_fine ok")


def test_sparse_update():
    """The unique()-based update equals the original dense bincount update (duplicates averaged)."""
    rng = np.random.default_rng(1)
    Q1 = rng.normal(size=(50, E.N_ACT)).astype(np.float32); Q2 = Q1.copy()
    s, a, td = rng.integers(0, 50, 300), rng.integers(0, E.N_ACT, 300), rng.normal(size=300)
    flat = s * E.N_ACT + a
    sums, cnt = np.bincount(flat, weights=td, minlength=Q1.size), np.bincount(flat, minlength=Q1.size)
    hit = np.nonzero(cnt)[0]
    Q1.ravel()[hit] += (0.1 * sums[hit] / cnt[hit]).astype(np.float32)
    h2, inv = np.unique(flat, return_inverse=True)
    Q2.ravel()[h2] += (0.1 * np.bincount(inv, weights=td) / np.bincount(inv)).astype(np.float32)
    assert np.array_equal(hit, h2) and np.abs(Q1 - Q2).max() < 1e-6
    print("sparse update ok")


def test_rule():
    ctrl = rule.make()
    fast_straight = int(np.argmin((E.CV - 0.2) ** 2 + E.CW ** 2))
    o = blank_obs(2.0, 0.0)
    assert ctrl(o[None])[0] == fast_straight, "free space, goal ahead: should drive fast and straight"
    o2 = o.copy()
    for a in range(-20, 21, 10):                                        # wall 0.25 m ahead across +-20 deg
        o2[sector_of(a)] = 0.25 / 2.0
    k = ctrl(o2[None])[0]
    assert not (E.CV[k] > 0.1 and abs(E.CW[k]) < 0.5), (E.CV[k], E.CW[k])
    o3 = blank_obs(2.0, math.radians(90))                               # goal to the left: turn left (positive w)
    assert E.CW[ctrl(o3[None])[0]] > 0
    o4 = o.copy(); o4[41:45] = [0.35 / 3, 0.0, -0.2 / 0.4, 0.0]        # robot 0.35 m ahead coming at us
    k4 = ctrl(o4[None])[0]
    assert k4 != fast_straight, "head-on robot ignored"
    print("rule ok")


def test_dqn_export():
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import VecNormalize
    import dqn
    tp = C.train_pool()
    env = VecNormalize(E.ArenaVecEnv(C.subset(tp, C.TRAIN_IDX[:20]), n_worlds=2, seed=0), norm_obs=True, norm_reward=True)
    model = DQN("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh), learning_starts=100,
                buffer_size=5000, batch_size=32, device="cpu", seed=0)
    model.learn(600)
    with tempfile.TemporaryDirectory() as d:
        dqn.CFG_ACT = "tanh"
        dqn.export_npz(model, env, pathlib.Path(d) / "q.npz")
        import deploy
        pol = deploy.Policy(str(pathlib.Path(d) / "q.npz"))
        raw = env.get_original_obs()
        q_np = pol.logits(raw)
        with torch.no_grad():
            q_t = model.q_net(torch.as_tensor(env.normalize_obs(raw), dtype=torch.float32)).numpy()
        assert np.abs(q_np - q_t).max() < 1e-4, np.abs(q_np - q_t).max()
        assert (pol.index(raw) == q_t.argmax(1)).all()
    print("dqn export ok")


def test_stats():
    o = np.ones((10, 4), int); o[0, 0] = 2; o[1, :] = 3
    raw = {"layouts": list(range(10)), "outcome": o.tolist(), "cause": np.where(o == 2, 1, 0).tolist(), "finish_t": np.full((10, 4), 50).tolist()}
    s = C.summarize(raw, ref=raw, n_boot=500)
    assert abs(s["success"] - 35 / 40) < 1e-9 and abs(s["collision"] - 1 / 40) < 1e-9 and abs(s["timeout"] - 4 / 40) < 1e-9
    assert abs(s["coll_obstacle"] - 1 / 40) < 1e-9 and abs(s["all_four"] - 0.8) < 1e-9 and abs(s["time_s"] - 5.0) < 1e-9
    assert s["d_success_vs_ref"] == 0 and s["d_success_vs_ref_ci"] == [0.0, 0.0]
    assert s["success_ci"][0] <= s["success"] <= s["success_ci"][1]
    print("stats ok")


def test_split():
    tp = C.train_pool()
    assert set(tp["seeds"][C.TRAIN_IDX]).isdisjoint(tp["seeds"][C.VAL_IDX])
    assert tp["seeds"][C.VAL_IDX].min() == 2900 and tp["seeds"][C.TRAIN_IDX].max() == 2899
    ep = C.eval_pool()
    assert (ep["seeds"][C.TEST] == np.arange(100, 200)).all()
    print("split ok")


def test_behaviour_metrics():
    """Behaviour bookkeeping is read-only and physically sane, on a few test layouts."""
    ep = C.eval_pool()
    raw = C.run_protocol(ep, C.npz_ctrl(C.ROOT / "weights/final_policy.npz"), C.TEST[:8])
    o, path, sg = np.asarray(raw["outcome"]), np.asarray(raw["path_m"]), np.asarray(raw["start_goal_m"])
    ok = o == 1
    assert (path[ok] >= sg[ok] - E.GOAL_TOL - 1e-6).all(), "path shorter than the straight line"
    coll = (o == 2)
    gaps = np.minimum(np.asarray(raw["min_gap_obstacle_m"]), np.asarray(raw["min_gap_robot_m"]))
    assert (gaps[coll] < 0).all(), "a collided robot must have a negative closest gap"
    assert (gaps[ok] >= 0).all(), "a successful robot cannot have touched anything"
    hp, he = np.asarray(raw["action_hist_proposed"]), np.asarray(raw["action_hist_executed"])
    assert hp.sum() == he.sum() == np.sum(raw["steps_alive"]), (hp.sum(), np.sum(raw["steps_alive"]))
    assert abs(raw["override_rate"] - np.sum(raw["safety_swaps"]) / hp.sum()) < 1e-12
    b = C.behaviour(raw)
    assert 0 < b["path_efficiency"] <= 1.1 and 0 < b["mean_speed_mps"] <= E.VMAX + 1e-6, b
    print("behaviour metrics ok", {k: round(v, 3) for k, v in b.items()})


def test_ppo_regression():
    """The harness must reproduce PPO's known protocol result on layouts 0-99 (91.5% / 4.0% / 4.5%)."""
    s = C.summarize(C.run_protocol(C.eval_pool(), C.npz_ctrl(C.ROOT / "weights/final_policy.npz"), C.PPO_SELECTION), n_boot=100)
    assert (round(s["success"], 3), round(s["collision"], 3), round(s["timeout"], 3)) == (0.915, 0.04, 0.045), s
    print("ppo regression ok")


if __name__ == "__main__":
    test_split(); test_stats(); test_discretize(); test_discretize_fine(); test_sparse_update(); test_rule(); test_dqn_export(); test_behaviour_metrics()
    if "--slow" in sys.argv:
        test_ppo_regression()
    print("BENCHMARK TESTS PASSED")
