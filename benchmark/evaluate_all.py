"""Final 2D evaluation of benchmark contenders (benchmark/PROTOCOL.md): test layouts 100-199 (+ 0-99 for reference),
safety layer on and off, paired statistics against PPO. One JSON per (contender, layout set, shield) in benchmark/results/.

    .venv/bin/python benchmark/evaluate_all.py NAME=SPEC [NAME=SPEC ...]
SPEC: npz:PATH (PPO or DQN export) | dqnnpz:PATH (DQN, also relu) | tabq:PATH | rule | goto | dwa
Example: evaluate_all.py ppo=npz:weights/final_policy.npz goto=goto dwa=dwa"""
import json
import sys
import time

import common as C
from meter import Meter

SETS = {"test": C.TEST, "ref0-99": C.PPO_SELECTION}


def model_size(spec):
    """Parameters and bytes the controller needs at run time."""
    import numpy as np
    kind, _, arg = spec.partition(":")
    if kind in ("npz", "dqnnpz"):
        d = np.load(arg)
        n = sum(d[k].size for k in d.files if k.startswith(("weight", "bias")))
        return {"params": int(n), "bytes": int(sum(d[k].nbytes for k in d.files))}
    if kind == "tabq":
        q = np.load(arg)["Q"]
        return {"params": int(q.size), "bytes": int(q.nbytes)}
    if kind == "rule":
        return {"params": 7, "bytes": 0}
    return {"params": 0, "bytes": 0}


def single_robot_latency(ctrl, n=2000):
    """Median / p99 time of ONE robot's decision (batch of 1), the way a robot's own computer would call it.
    Not available for privileged controllers that need the simulator state (DWA)."""
    import numpy as np
    ep = C.eval_pool()
    env = C.E.ArenaVecEnv(ep, mode="eval", eval_idx=C.TEST[:5], seed=0, randomize=False)
    obs = env.reset()
    ts = []
    for i in range(n):
        o = obs[i % len(obs)][None]
        t0 = time.perf_counter()
        ctrl(o, None)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts[100:]) * 1e6  # drop warm-up
    return {"single_us_median": float(np.median(ts)), "single_us_p99": float(np.percentile(ts, 99)), "single_us_mean": float(ts.mean())}


def make(spec):
    kind, _, arg = spec.partition(":")
    if kind == "npz":
        return C.npz_ctrl(arg)
    if kind == "dqnnpz":
        import dqn
        return dqn.ctrl_from(arg)
    if kind == "tabq":
        import tabq
        return tabq.ctrl_from(arg)
    if kind == "rule":
        import rule
        return rule.best_ctrl()
    return {"goto": C.goto_ctrl, "dwa": C.dwa_ctrl}[kind]


def raw_path(name, set_name, shield):
    return C.OUT / "raw" / f"{name}__{set_name}__shield{int(shield)}.json"


def main(pairs):
    ep = C.eval_pool()
    (C.OUT / "raw").mkdir(parents=True, exist_ok=True)
    for name, spec in pairs:
        ctrl = make(spec)
        info = {"name": name, "spec": spec, **model_size(spec)}
        if not spec.startswith("dwa"):
            with Meter(f"latency_{name}", kind="latency", method=name, spec=spec) as m:
                info.update(single_robot_latency(ctrl))
        print(f"{name}: {info}", flush=True)
        C.save(f"deploy_cost__{name}", info)
        for set_name, layouts in SETS.items():
            for shield in (True, False):
                with Meter(f"eval_{name}_{set_name}_shield{int(shield)}", kind="eval", method=name, spec=spec, layouts=set_name, shield=shield) as m:
                    raw = C.run_protocol(ep, ctrl, layouts, shield=shield)
                    m.add(decision_us_per_robot=raw["decision_us_per_robot"], decision_s_total=raw["decision_s_total"], robot_runs=len(layouts) * 4)
                raw["spec"], raw["name"], raw["wall_s"] = spec, name, m.record["wall_s"]
                raw_path(name, set_name, shield).write_text(json.dumps(raw))
                ref = raw_path("ppo", set_name, shield)
                s = C.summarize(raw, ref=json.loads(ref.read_text()) if (ref.exists() and name != "ppo") else None)
                print(C.fmt(f"{name} {set_name} sh{int(shield)}", s), f"({m.record['wall_s']:.0f}s)", flush=True)
                print("   behaviour:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k in C.behaviour(raw)}, flush=True)


if __name__ == "__main__":
    main([a.split("=", 1) for a in sys.argv[1:]])
