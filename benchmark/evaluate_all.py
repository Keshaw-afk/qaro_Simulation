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


if __name__ == "__main__":
    main([a.split("=", 1) for a in sys.argv[1:]])
