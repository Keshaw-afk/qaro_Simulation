"""Gazebo confirmation for benchmark contenders (benchmark/PROTOCOL.md): full pipeline (overhead camera + AprilTags,
360-beam lidar, odometry), four robots, safety layer on, layouts FIRST..N_END-1 (protocol: 100-119).
Reuses rl/eval_gazebo.py run_layout unchanged; any benchmark controller is wrapped as a policy with .index(obs).

    source /opt/ros/jazzy/setup.bash
    GZ_PARTITION=snr_bench .venv/bin/python benchmark/eval_gazebo_bench.py NAME SPEC FIRST N_END
SPEC as in evaluate_all.py (npz:/dqnnpz:/tabq:/rule; not dwa, which needs the 2D simulator's true state).
Writes benchmark/results/gazebo__NAME.json (per layout status/times + decision timing) and, via SNR_LOG=1, per-step
logs runs/benchmark/gazebo/NAME/log_seed{k}.csv (positions, clearances, safety-layer vetoes)."""
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
NAME, SPEC, FIRST, N_END = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.environ.setdefault("SNR_OUT_DIR", str(ROOT / "runs" / "benchmark" / "gazebo" / NAME))
os.environ.setdefault("SNR_LOG", "1")
sys.path[:0] = [str(ROOT / "benchmark"), str(ROOT / "rl"), str(ROOT / "tools")]
import numpy as np  # noqa: E402

import eval_gazebo as G  # noqa: E402  (reads SNR_* at import)
import evaluate_all as EA  # noqa: E402
from meter import Meter  # noqa: E402


class Wrapped:
    """Policy interface expected by eval_gazebo.run_layout: index(obs) -> action indices. Times every decision."""
    def __init__(self, ctrl):
        self.ctrl, self.times = ctrl, []

    def index(self, obs):
        t0 = time.perf_counter()
        a = np.asarray(self.ctrl(np.atleast_2d(obs), None))
        self.times.append(time.perf_counter() - t0)
        assert a.ndim == 1, "Gazebo wrapper needs action indices"
        return a


if __name__ == "__main__":
    assert not SPEC.startswith("dwa")
    pol = Wrapped(EA.make(SPEC))
    out = ROOT / "benchmark" / "results" / f"gazebo__{NAME}.json"
    results = []
    with Meter(f"gazebo_{NAME}_{FIRST}-{N_END - 1}", kind="gazebo", method=NAME, spec=SPEC, partition=os.environ.get("GZ_PARTITION")) as m:
        for k in range(FIRST, N_END):
            for attempt in (1, 2):
                try:
                    t0 = time.time()
                    r = G.run_layout(pol, k)
                    r["wall_s"] = round(time.time() - t0, 1)
                    r["attempt"] = attempt
                    break
                except RuntimeError as e:  # simulator start-up hiccup: retry once, and record it
                    print(f"seed {k}: simulator failed ({e}); attempt {attempt}", flush=True)
                    if attempt == 2:
                        r = {"seed": k, "status": ["error"] * 4, "time": [None] * 4, "steps": 0, "error": str(e)[:300]}
            results.append(r)
            print(f"seed {k:3d}: {r['status']}  ({r['steps']} policy steps, {r.get('wall_s')}s)", flush=True)
            out.write_text(json.dumps({"name": NAME, "spec": SPEC, "layouts": [FIRST, N_END - 1], "results": results}, indent=1))
        st = [s for r in results for s in r["status"]]
        tt = [t for r in results for s, t in zip(r["status"], r["time"]) if s == "success"]
        lat = np.array(pol.times) * 1e6
        summary = {"success": st.count("success") / len(st), "collision": st.count("collision") / len(st), "timeout": st.count("timeout") / len(st),
                   "error": st.count("error") / len(st), "mean_time_s": float(np.mean(tt)) if tt else None,
                   "all_four": float(np.mean([all(s == "success" for s in r["status"]) for r in results])),
                   "decision_us_median": float(np.median(lat)), "decision_us_p99": float(np.percentile(lat, 99)), "decisions": len(lat)}
        m.add(**{f"gz_{k}": v for k, v in summary.items()}, layouts=f"{FIRST}-{N_END - 1}")
    out.write_text(json.dumps({"name": NAME, "spec": SPEC, "layouts": [FIRST, N_END - 1], "summary": summary, "results": results}, indent=1))
    print(f"GAZEBO {NAME}: {summary}", flush=True)
    sys.stdout.flush()
    os._exit(0)  # same reason as rl/eval_gazebo.py: gz-transport threads can segfault during interpreter teardown
