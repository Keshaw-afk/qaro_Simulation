"""Deep Q-learning (DQN, Stable-Baselines3) with PPO's network shape: MLP 59-256-256-45, tanh, observation normalisation.

    python benchmark/dqn.py CONFIG SEED STEPS NAME [DEVICE=auto]      # writes runs/benchmark/dqn/NAME/{best,latest}.npz, log.csv

Same environment as PPO (64 worlds x 4 robots, domain randomisation, safety layer on), training seeds 1000-2899 only.
Every 1M steps: protocol evaluation on validation layouts (seeds 2900-2999); the best (success - 0.5 collision) is kept.
The exported .npz has the same format as weights/final_policy.npz (rl/deploy.py Policy takes the argmax of the Q-values),
so it also runs unchanged in Gazebo (rl/eval_gazebo.py)."""
import json
import sys
import time

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

import common as C
from common import E
from meter import Meter

BASE = dict(learning_rate=1e-4, buffer_size=1_000_000, learning_starts=50_000, batch_size=1024, train_freq=1, gradient_steps=2,
            target_update_interval=10_000, exploration_fraction=0.2, exploration_final_eps=0.05, gamma=0.99, max_grad_norm=10, act="tanh")
CONFIGS = {  # tuning grid (benchmark/PROTOCOL.md: up to 6 configurations x 1M steps, judged on validation)
    "a": {},
    "b": dict(learning_rate=3e-4),
    "c": dict(gradient_steps=4),
    "d": dict(exploration_fraction=0.5),
    "e": dict(target_update_interval=2_000),
    "f": dict(act="relu"),
}
# train_freq=1 means one update round per vectorised step = 256 transitions; gradient_steps x batch 1024 / 256 = replay ratio.


def export_npz(model, venv, path):
    """Q-network as plain numpy arrays (same layout as rl/train.py export_npz; deploy.Policy argmax = greedy action)."""
    net = model.q_net.q_net
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    np.savez(path, obs_mean=venv.obs_rms.mean, obs_var=venv.obs_rms.var, act=np.array(CFG_ACT),
             **{f"{k}{i}": getattr(l, k).detach().cpu().numpy() for i, l in enumerate(lin) for k in ("weight", "bias")})


def ctrl_from(path):
    """Greedy controller from an exported .npz (handles the relu variant, which deploy.Policy does not)."""
    d = np.load(path)
    if "act" in d.files and str(d["act"]) == "relu":
        layers = [(d[f"weight{i}"], d[f"bias{i}"]) for i in range(3)]
        def ctrl(obs, env):
            x = np.clip((obs - d["obs_mean"]) / np.sqrt(d["obs_var"] + 1e-8), -10, 10).astype(np.float32)
            for k, (w, b) in enumerate(layers):
                x = x @ w.T + b
                if k < 2:
                    x = np.maximum(x, 0)
            return x.argmax(1)
        return ctrl
    return C.npz_ctrl(path)


class EvalCb(BaseCallback):
    def __init__(self, out, tp, every=1_000_000):
        super().__init__()
        self.out, self.tp, self.every, self.next, self.best, self.t0 = out, tp, every, every, -1e9, time.time()
        self.log = open(out / "log.csv", "w")
        self.log.write("steps,wall_s,eps,val_success,val_collision,val_timeout,val_score,steps_per_s\n")
        self.last_t, self.last_n = time.time(), 0

    def _on_step(self):
        if self.num_timesteps < self.next:
            return True
        self.next += self.every
        self.evaluate()
        return True

    def evaluate(self):
        venv = self.model.get_env()
        export_npz(self.model, venv, self.out / "latest.npz")
        sm = C.summarize(C.run_protocol(self.tp, ctrl_from(self.out / "latest.npz"), C.VAL_IDX), n_boot=200)
        rate = (self.num_timesteps - self.last_n) / max(time.time() - self.last_t, 1e-9)
        self.last_t, self.last_n = time.time(), self.num_timesteps
        self.log.write(f"{self.num_timesteps},{time.time() - self.t0:.0f},{self.model.exploration_rate:.3f},{sm['success']:.4f},{sm['collision']:.4f},{sm['timeout']:.4f},{C.score(sm):.4f},{rate:.0f}\n")
        self.log.flush()
        print(f"[dqn {self.out.name}] steps {self.num_timesteps:>9d} eps {self.model.exploration_rate:.2f} val success {sm['success']:.3f} "
              f"collision {sm['collision']:.3f} timeout {sm['timeout']:.3f} ({rate:.0f} steps/s, {time.time() - self.t0:.0f}s)", flush=True)
        if C.score(sm) > self.best:
            self.best = C.score(sm)
            export_npz(self.model, venv, self.out / "best.npz")
            (self.out / "best.json").write_text(json.dumps({"steps": self.num_timesteps, **{k: sm[k] for k in ("success", "collision", "timeout")}}))


CFG_ACT = "tanh"


def train(cfg_name, seed, steps, name, device="auto"):
    global CFG_ACT
    cfg = {**BASE, **CONFIGS[cfg_name]}
    CFG_ACT = cfg.pop("act")
    out = C.RUNS / "dqn" / name
    out.mkdir(parents=True, exist_ok=True)
    tp = C.train_pool()
    env = VecNormalize(E.ArenaVecEnv(C.subset(tp, C.TRAIN_IDX), n_worlds=64, seed=seed), norm_obs=True, norm_reward=True, gamma=cfg["gamma"], clip_obs=10.0)
    act = torch.nn.Tanh if CFG_ACT == "tanh" else torch.nn.ReLU
    model = DQN("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256], activation_fn=act), seed=seed, device=device, verbose=0, **cfg)
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(str(out / "sb3"), ["csv"]))  # sb3/progress.csv: loss, exploration rate, episode return/length, success rate, fps
    n_params = sum(p.numel() for p in model.q_net.parameters())
    print(f"[dqn {name}] config {cfg_name} {cfg} act {CFG_ACT} device {model.device} params {n_params}", flush=True)
    cb = EvalCb(out, tp)
    with Meter(f"dqn_train_{name}", kind="train", method="dqn", config=cfg_name, seed=seed, device=str(model.device)) as m:
        model.learn(total_timesteps=steps, callback=cb, log_interval=1000)  # log every 1000 finished episodes (~150k steps)
        if cb.last_n < model.num_timesteps:
            cb.evaluate()
        m.add(env_steps=int(model.num_timesteps), gradient_steps=int(model._n_updates), best_val_score=cb.best, q_net_params=n_params,
              model_bytes_float32=n_params * 4, config_params={**cfg, "act": CFG_ACT}, batch_size=cfg["batch_size"])
    print(f"[dqn {name}] done: {m.record['wall_s']:.0f}s, {m.record.get('env_steps_per_s')} steps/s, gpu {m.record.get('gpu_J')} J, "
          f"cpu package {m.record.get('cpu_package_J')} J", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(int(__import__("os").environ.get("SNR_TORCH_THREADS", "2")))
    train(sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3])), sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "auto")
