"""The training-env observation and the sensor-level deployment observation must agree. Run: .venv/bin/python tests/test_obs_parity.py"""
import pathlib, sys
import numpy as np
R = pathlib.Path(__file__).resolve().parent.parent / "rl"
sys.path.insert(0, str(R))
import arena_env as E, deploy as D, pool as PL

P = PL.build(PL.EVAL_SEEDS, R / "pool_eval.npz")
env = E.ArenaVecEnv(P, n_worlds=8, seed=3, randomize=False)
rng = np.random.default_rng(0)
for _ in range(40):  # random walk so v, w, prev_act are non-trivial
    env.step_async(rng.uniform(-1, 1, (env.num_envs, 2))); env.step_wait()
fast = env._all_obs()                                           # what the policy was trained on (2 rays per group)
E.RAY_ANG = (-np.pi + np.arange(360) * E._STEP).astype(np.float32); E.RAYS_PER_GROUP = 10   # emulate the full raw 360-beam scan
env._all_obs(); raw = env._last_n * E.RANGE_MAX                  # (N, 360) raw ranges in metres, 0 = too close, 2 = max
worst = 0.0
for n in range(env.num_envs):
    w, a = divmod(n, E.N_AG)
    vel = lambda ww, aa: np.stack([env.v[ww, aa] * np.cos(env.yaw[ww, aa]), env.v[ww, aa] * np.sin(env.yaw[ww, aa])], -1)
    dep = D.obs_from_sensors(raw[n], env.pos[w, a], env.yaw[w, a], env.goal[w, a], env.pos[w, E.OTHERS[a]], vel(w, a), vel(w, E.OTHERS[a]), env.v[w, a], env.w[w, a], env.prev_act[w, a], a)
    worst = max(worst, np.abs(dep - fast[n]).max())
print(f"deploy obs (from raw 360-beam scan + camera/odom inputs) vs training obs: max abs diff {worst:.2e}")
assert worst < 1e-4
assert np.array_equal(D.CV, E.CV) and np.array_equal(D.CW, E.CW), "discrete action tables differ"
print("PARITY OK (observations + action table)")
