"""PPO with one shared policy over all robots. Usage: .venv/bin/python rl/train.py RUN_NAME TOTAL_STEPS"""
import pathlib
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arena_env as E
import evaluate as EV
import pool as PL

RUN, TOTAL = sys.argv[1], int(float(sys.argv[2]))
OUT = HERE.parent / "runs" / RUN
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)


def export_npz(model, path):
    """Actor weights as plain numpy arrays (deployment needs no torch): tanh MLP 59-256-256-45 logits."""
    p = model.policy
    ls = [p.mlp_extractor.policy_net[0], p.mlp_extractor.policy_net[2], p.action_net]
    rms = model.get_vec_normalize_env().obs_rms
    np.savez(path, obs_mean=rms.mean, obs_var=rms.var, **{f"{k}{i}": getattr(l, k).detach().numpy() for i, l in enumerate(ls) for k in ("weight", "bias")})


class EvalCb(BaseCallback):
    def __init__(self, pool_eval, every):
        super().__init__()
        self.pool_eval, self.every, self.best, self.last = pool_eval, every, -1, 0
        self.log = open(OUT / "eval.csv", "a")

    def _act(self, obs):
        return self.model.predict(self.training_env.normalize_obs(obs), deterministic=True)[0]

    def _on_step(self):
        if self.num_timesteps - self.last < self.every:
            return True
        self.last = self.num_timesteps
        m = EV.run_eval(self.pool_eval, self._act, n_layouts=100)
        row = f"{self.num_timesteps},{m['success']:.3f},{m['collision']:.3f},{m['timeout']:.3f},{m['mean_time_s']:.1f},{m['all_four_success']:.3f}"
        self.log.write(row + "\n"); self.log.flush()
        print(f"[eval] steps={self.num_timesteps:>9d} success={m['success']:.3f} collision={m['collision']:.3f} timeout={m['timeout']:.3f} t={m['mean_time_s']:.1f}s all4={m['all_four_success']:.3f}", flush=True)
        self.model.save(OUT / "latest"); export_npz(self.model, OUT / "latest.npz")
        score = m["success"] - m["collision"] * 0.5
        if score > self.best:
            self.best = score
            self.model.save(OUT / "best"); export_npz(self.model, OUT / "best.npz")
        return True


if __name__ == "__main__":
    train_pool = PL.build(PL.TRAIN_SEEDS, HERE / "pool_train.npz")
    eval_pool = PL.build(PL.EVAL_SEEDS, HERE / "pool_eval.npz")
    base = EV.run_eval(eval_pool, EV.goto_goal, n_layouts=100)
    print("[baseline go-to-goal]", {k: round(v, 3) for k, v in base.items()}, flush=True)
    env = VecNormalize(E.ArenaVecEnv(train_pool, n_worlds=64, seed=0), norm_obs=True, norm_reward=True, gamma=0.99, clip_obs=10.0)  # obs mean/var are exported with the weights; deploy.Policy applies them
    model = PPO("MlpPolicy", env, n_steps=128, batch_size=2048, n_epochs=10, learning_rate=lambda f: 1e-4 + (3e-4 - 1e-4) * f,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh, log_std_init=-0.5),
                verbose=1, seed=0, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=TOTAL, callback=EvalCb(eval_pool, every=1_000_000), log_interval=10)
    model.save(OUT / "final"); export_npz(model, OUT / "final.npz")
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
