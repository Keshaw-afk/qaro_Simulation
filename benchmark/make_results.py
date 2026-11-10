"""Collect every benchmark result into benchmark/results/summary.json and print the Markdown tables used in RESULTS.md.
Recomputes all statistics from the raw per-robot files (results/raw/*.json, results/gazebo__*.json); nothing is copied
by hand. Run: .venv/bin/python benchmark/make_results.py"""
import json

import numpy as np

import common as C

R = C.OUT
MAIN = ["ppo", "ppo_final", "dqn_s1", "dqn_s2", "dqn_s3", "rule", "tabq_s1", "tabq_s2", "tabq_s3", "tabq_fine_explore", "dwa", "goto"]
LABEL = {"ppo": "PPO (deployed)", "ppo_final": "PPO, 8M checkpoint", "dqn_s1": "DQN seed 1", "dqn_s2": "**DQN seed 2 (chosen)**",
         "dqn_s3": "DQN seed 3", "rule": "**Rule controller**", "tabq_s1": "Tabular Q seed 1", "tabq_s2": "Tabular Q seed 2",
         "tabq_s3": "**Tabular Q seed 3 (chosen)**", "tabq_fine_explore": "Tabular Q fine (exploratory)",
         "dwa": "DWA (privileged: true map)", "goto": "Go-to-goal"}
CPU_W_PER_THREAD = 45.0 / 12  # estimate, see LOGBOOK "Energy measurement"


def raw(name, s, sh):
    return json.loads((R / "raw" / f"{name}__{s}__shield{sh}.json").read_text())


def pct(x):
    return f"{100 * x:.1f}"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def dci(d, c):
    return f"{100 * d:+.1f} [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"


def gazebo_stats(name, ref=None, n_boot=10000):
    """Gazebo outcomes per layout; bootstrap over the 20 layouts (paired with PPO on the same layouts)."""
    g = json.loads((R / f"gazebo__{name}.json").read_text())
    st = np.array([r["status"] for r in g["results"]])
    succ = (st == "success").mean(1)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(st), (n_boot, len(st)))
    out = {"layouts": [r["seed"] for r in g["results"]], "robots": int(st.size), "success": float((st == "success").mean()),
           "success_ci": np.percentile(succ[idx].mean(1), [2.5, 97.5]).tolist(), "collision": float((st == "collision").mean()),
           "timeout": float((st == "timeout").mean()), "counts": {k: int((st == k).sum()) for k in ("success", "collision", "timeout", "error")},
           **{k: g["summary"][k] for k in ("mean_time_s", "all_four", "decision_us_median", "decision_us_p99")}}
    if ref is not None:
        assert out["layouts"] == ref["layouts"]
        d = succ - ref["_succ"]
        out["d_success_vs_ppo"] = float(d.mean())
        out["d_success_vs_ppo_ci"] = np.percentile(d[idx].mean(1), [2.5, 97.5]).tolist()
    out["_succ"] = succ
    return out


def compute():
    """Training/tuning/evaluation cost per method from measurements (laptop + Colab)."""
    rec = [json.loads(l) for f in ("measurements.jsonl", "measurements_colab.jsonl") for l in open(R / f)]
    rows = {}
    for r in rec:
        if r.get("kind") not in ("train", "tune", "eval", "gazebo") or "smoke" in r["name"]:
            continue
        m = r.get("method") or r["name"].split("_")[1]
        key = (m, r["kind"])
        a = rows.setdefault(key, {"wall_s": 0.0, "cpu_s": 0.0, "cpu_J_est": 0.0, "gpu_J": 0.0, "jobs": 0})
        cpu = (r.get("cpu_user_s") or 0) + (r.get("cpu_sys_s") or 0)
        a["wall_s"] += r["wall_s"]; a["cpu_s"] += cpu; a["jobs"] += 1
        a["cpu_J_est"] += r.get("cpu_energy_est_J") or cpu * CPU_W_PER_THREAD
        a["gpu_J"] += r.get("gpu_J") or 0.0
    return {f"{m}/{k}": v for (m, k), v in sorted(rows.items())}


def dqn_final_cost():
    """The DQN finals never wrote a Meter record (Colab reset); reconstruct from their own logs."""
    out = {}
    for s in (1, 2, 3):
        rows = list(np.genfromtxt(C.RUNS / f"dqn/final_s{s}/log.csv", delimiter=",", names=True))
        out[f"s{s}"] = {"steps_logged": int(rows[-1]["steps"]), "wall_s_logged": float(rows[-1]["wall_s"]),
                        "steps_per_s": float(rows[-1]["steps_per_s"])}
    return out


if __name__ == "__main__":
    S = {"twod": {}, "behaviour": {}, "deploy": {}, "gazebo": {}}
    ref = {(s, sh): raw("ppo", s, sh) for s in ("test", "ref0-99") for sh in (1, 0)}
    for n in MAIN:
        for s in ("test", "ref0-99"):
            for sh in (1, 0):
                r = raw(n, s, sh)
                sm = C.summarize(r, ref=None if n == "ppo" else ref[(s, sh)])
                sm["override_rate"] = r.get("override_rate")
                S["twod"][f"{n}|{s}|{sh}"] = sm
                if s == "test":
                    S["behaviour"][f"{n}|{sh}"] = C.behaviour(r)
        S["deploy"][n] = json.loads((R / f"deploy_cost__{n}.json").read_text())

    print("### 2D, test layouts 100-199 (400 robots), safety layer ON\n")
    print("| Method | Success % [95% CI] | Collision % | Timeout % | All four % | Time to goal s | Overrides % | dSuccess vs PPO [95% CI] |")
    print("|---|---|---|---|---|---|---|---|")
    for sh in (1, 0):
        if sh == 0:
            print("\n### 2D, test layouts 100-199, safety layer OFF\n")
            print("| Method | Success % [95% CI] | Collision % (robot-robot) | Timeout % | All four % | Time to goal s | dSuccess vs PPO (also off) [95% CI] |")
            print("|---|---|---|---|---|---|---|")
        for n in MAIN:
            m = S["twod"][f"{n}|test|{sh}"]
            d = "-" if n == "ppo" else dci(m["d_success_vs_ref"], m["d_success_vs_ref_ci"])
            if sh:
                print(f"| {LABEL[n]} | {pct(m['success'])} {ci(m['success_ci'])} | {pct(m['collision'])} | {pct(m['timeout'])} | "
                      f"{pct(m['all_four'])} | {m['time_s']:.1f} | {pct(m['override_rate'])} | {d} |")
            else:
                print(f"| {LABEL[n]} | {pct(m['success'])} {ci(m['success_ci'])} | {pct(m['collision'])} ({pct(m['coll_robot'])}) | "
                      f"{pct(m['timeout'])} | {pct(m['all_four'])} | {m['time_s']:.1f} | {d} |")

    print("\n### Reference: layouts 0-99 (PPO's checkpoint-selection set), safety layer on / off: success %\n")
    print("| Method | on | off |\n|---|---|---|")
    for n in MAIN:
        print(f"| {LABEL[n]} | {pct(S['twod'][f'{n}|ref0-99|1']['success'])} | {pct(S['twod'][f'{n}|ref0-99|0']['success'])} |")

    print("\n### Gazebo, layouts 100-119 (80 robots), safety layer on\n")
    print("| Method | Success % [95% CI] | Robots ok / coll / timeout | All four % | Time to goal s | dSuccess vs PPO [95% CI] | Decision us (median / p99) |")
    print("|---|---|---|---|---|---|---|")
    gp = gazebo_stats("ppo")
    for n in ("ppo", "dqn_s2", "rule"):
        g = gp if n == "ppo" else gazebo_stats(n, gp)
        S["gazebo"][n] = {k: v for k, v in g.items() if k != "_succ"}
        c = g["counts"]
        d = "-" if n == "ppo" else dci(g["d_success_vs_ppo"], g["d_success_vs_ppo_ci"])
        print(f"| {LABEL[n]} | {pct(g['success'])} {ci(g['success_ci'])} | {c['success']} / {c['collision']} / {c['timeout']} | "
              f"{pct(g['all_four'])} | {g['mean_time_s']:.1f} | {d} | {g['decision_us_median']:.0f} / {g['decision_us_p99']:.0f} |")

    print("\n### Behaviour (2D test, safety layer on)\n")
    keys = ["path_efficiency", "mean_speed_mps", "stopped_frac_success", "turn_flips_per_min", "min_gap_obstacle_m_median",
            "min_gap_robot_m_median", "near_miss_obstacle_steps_per_robot", "action_entropy_bits"]
    print("| Method | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    for n in MAIN:
        b = S["behaviour"][f"{n}|1"]
        print(f"| {LABEL[n]} | " + " | ".join(f"{b[k]:.3g}" for k in keys) + " |")

    print("\n### Deployment cost (one robot's decision, laptop CPU)\n")
    print("| Method | Parameters | Bytes | Decision us median / p99 |\n|---|---|---|---|")
    for n in MAIN:
        d = S["deploy"][n]
        print(f"| {LABEL[n]} | {d.get('params')} | {d.get('bytes')} | "
              + (f"{d['single_us_median']:.0f} / {d['single_us_p99']:.0f}" if d.get("single_us_median") else "n/a (needs simulator state)") + " |")

    S["compute"] = compute()
    S["dqn_final_cost"] = dqn_final_cost()
    print("\n### Compute (sum over jobs; CPU energy estimated at 3.75 W per busy thread, GPU energy measured)\n")
    print("| Method / kind | Jobs | Wall h | CPU h | CPU energy est. kWh | GPU energy kWh |\n|---|---|---|---|---|---|")
    for k, v in S["compute"].items():
        print(f"| {k} | {v['jobs']} | {v['wall_s'] / 3600:.2f} | {v['cpu_s'] / 3600:.2f} | {v['cpu_J_est'] / 3.6e6:.3f} | {v['gpu_J'] / 3.6e6:.3f} |")
    print("\nDQN finals (no Meter record):", S["dqn_final_cost"])
    C.save("summary", S)
