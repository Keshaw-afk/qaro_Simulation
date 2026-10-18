"""PPO with one shared policy over all robots. Usage: .venv/bin/python rl/train.py RUN_NAME TOTAL_STEPS"""
import pathlib
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

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
    """Actor weights as plain numpy arrays (deployment needs no torch): tanh MLP 55-256-256-2."""
    p = model.policy
    ls = [p.mlp_extractor.policy_net[0], p.mlp_extractor.policy_net[2], p.action_net]
    np.savez(path, **{f"{k}{i}": getattr(l, k).detach().numpy() for i, l in enumerate(ls) for k in ("weight", "bias")})


class EvalCb(BaseCallback):
    def __init__(self, pool_eval, every):
        super().__init__()
        self.pool_eval, self.every, self.best, self.last = pool_eval, every, -1, 0
        self.log = open(OUT / "eval.csv", "a")

    def _act(self, obs):
        return self.model.predict(obs, deterministic=True)[0]

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
    env = E.ArenaVecEnv(train_pool, n_worlds=64, seed=0)
    model = PPO("MlpPolicy", env, n_steps=128, batch_size=4096, n_epochs=4, learning_rate=lambda f: 5e-5 + (3e-4 - 5e-5) * f,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.002, clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh, log_std_init=-0.5),
                verbose=1, seed=0, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=TOTAL, callback=EvalCb(eval_pool, every=1_000_000), log_interval=10)
    model.save(OUT / "final"); export_npz(model, OUT / "final.npz")
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
