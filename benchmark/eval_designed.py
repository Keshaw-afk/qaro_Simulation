"""All benchmark controllers on the hand-designed crossing layouts 9001-9005 (tools/designed_layouts.py), 2D simulator,
safety layer on and off. Exploratory stress test, NOT part of the pre-registered benchmark (5 layouts x 4 robots, one
deterministic episode each, so no confidence intervals are claimed).

    .venv/bin/python benchmark/eval_designed.py      # -> benchmark/results/designed_2d.json + a table per layout"""
import json

import numpy as np

import common as C
import evaluate_all as EA
import pool as PL

sys_ids = (9001, 9002, 9003, 9004, 9005)
SPECS = {"ppo": "npz:../weights/final_policy.npz", "dqn_s2": "npz:../runs/benchmark/dqn/final_s2/best.npz", "rule": "rule",
         "tabq_s3": "tabq:../runs/benchmark/tabq/final_s3/best.npz", "dwa": "dwa", "goto": "goto"}
OUT = {1: "ok", 2: "coll", 3: "time"}


if __name__ == "__main__":
    pool = PL.build(sys_ids, procs=5)          # not cached: always rebuilt from tools/designed_layouts.py
    assert list(pool["seeds"]) == list(sys_ids)
    res = {}
    for name, spec in SPECS.items():
        ctrl = EA.make(spec)
        for sh in (1, 0):
            r = C.run_protocol(pool, ctrl, np.arange(len(sys_ids)), shield=bool(sh))
            o, c = np.asarray(r["outcome"]), np.asarray(r["cause"])
            res[f"{name}|{sh}"] = {"outcome": o.tolist(), "cause": c.tolist(), "finish_t": np.asarray(r["finish_t"]).tolist(),
                                   "success": float((o == 1).mean()), "collision": float((o == 2).mean()),
                                   "coll_robot": float(((o == 2) & (c == 3)).mean()), "timeout": float((o == 3).mean())}
            print(f"{name:8s} shield {sh}: success {res[f'{name}|{sh}']['success']:.2f}  collision {res[f'{name}|{sh}']['collision']:.2f} "
                  f"(robot {res[f'{name}|{sh}']['coll_robot']:.2f})  timeout {res[f'{name}|{sh}']['timeout']:.2f}   per layout: "
                  + "  ".join(f"{k}:" + "/".join(OUT.get(int(x), '?') for x in row) for k, row in zip(sys_ids, o)), flush=True)
    C.save("designed_2d", {"layouts": list(sys_ids), "specs": SPECS, "results": res,
                           "note": "exploratory; one deterministic episode per layout; codes 1 success 2 collision 3 timeout; cause 3 = robot"})
