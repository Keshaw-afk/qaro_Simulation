"""DAgger: imitate the (privileged, collision-free) dynamic-window planner with a policy that only sees the deployable
observation (lidar, camera poses, odometry). Student acts with prob 1-beta, the planner labels every visited state.
Output: runs/NAME/bc.npz in the same format deploy.Policy reads (obs standardisation folded into layer 0).
Usage: .venv/bin/python rl/dagger.py NAME [ITERS] [STEPS_PER_ITER]"""
import pathlib, sys, time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arena_env as E
import evaluate as EV
import expert as X
import pool as PL

NAME = sys.argv[1]
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 500
OUT = HERE.parent / "runs" / NAME
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
rng = np.random.default_rng(0)
BETAS = [1.0, 0.5, 0.3, 0.15, 0.05] + [0.0] * 20


def build_net():
    return torch.nn.Sequential(torch.nn.Linear(E.OBS_DIM, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh(), torch.nn.Linear(256, E.N_ACT))


def export(net, mean, std, path):
    l0, l1, l2 = net[0], net[2], net[4]
    w0 = l0.weight.detach().numpy() / std[None, :]
    b0 = l0.bias.detach().numpy() - w0 @ mean
    np.savez(path, obs_mean=np.zeros(E.OBS_DIM, np.float32), obs_var=np.ones(E.OBS_DIM, np.float32), weight0=w0.astype(np.float32), bias0=b0.astype(np.float32),
             weight1=l1.weight.detach().numpy(), bias1=l1.bias.detach().numpy(), weight2=l2.weight.detach().numpy(), bias2=l2.bias.detach().numpy())


def fit(net, O, A, mean, std, epochs, lr):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    On = torch.tensor(np.clip((O - mean) / std, -10, 10), dtype=torch.float32)
    At = torch.tensor(A, dtype=torch.long)
    for ep in range(epochs):
        perm = torch.randperm(len(On))
        tot = acc = 0.0
        for i in range(0, len(On), 2048):
            b = perm[i:i + 2048]
            logits = net(On[b])
            loss = torch.nn.functional.cross_entropy(logits, At[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b); acc += (logits.argmax(1) == At[b]).sum().item()
    return tot / len(On), acc / len(On)


if __name__ == "__main__":
    train_pool, eval_pool = PL.build(PL.TRAIN_SEEDS, HERE / "pool_train.npz"), PL.build(PL.EVAL_SEEDS, HERE / "pool_eval.npz")
    env = E.ArenaVecEnv(train_pool, n_worlds=48, seed=1)
    net, mean, std = build_net(), None, None
    obs, data_o, data_a, t0 = env.reset(), [], [], time.time()
    for it in range(ITERS):
        beta = BETAS[it]
        for _ in range(STEPS):
            label = X.dwa_actions(env)
            if mean is None:
                act = label
            else:
                with torch.no_grad():
                    stud = net(torch.tensor(np.clip((obs - mean) / std, -10, 10), dtype=torch.float32)).argmax(1).numpy()
                act = np.where(rng.random(len(obs)) < beta, label, stud)
            data_o.append(obs.copy()); data_a.append(label)
            env.step_async(act); obs, *_ = env.step_wait()
        O, A = np.concatenate(data_o), np.concatenate(data_a)
        if mean is None:  # fixed after the first (expert-only) round
            mean, std = O.mean(0), np.sqrt(O.var(0) + 1e-4)
        loss, acc = fit(net, O, A, mean, std, epochs=8 if it < ITERS - 1 else 30, lr=1e-3 if it < ITERS - 1 else 3e-4)
        export(net, mean, std, OUT / "bc.npz")
        import deploy as D
        m = EV.run_eval(eval_pool, D.Policy(OUT / "bc.npz"), n_layouts=60)
        print(f"[dagger] iter {it} beta {beta:.2f} samples {len(O)} loss {loss:.3f} label-acc {acc:.3f} | 2D eval: success {m['success']:.3f} collision {m['collision']:.3f} (robot {m['coll_robot']:.3f}, obstacle {m['coll_obstacle']:.3f}) timeout {m['timeout']:.3f} [{(time.time()-t0)/60:.1f} min]", flush=True)
    print("done", flush=True)
