"""One demo layout in Gazebo for any benchmark controller (used by tools/make_demo.sh with SNR_SPEC).
Same wrapper as benchmark/eval_gazebo_bench.py, but writes nothing into benchmark/results/ and no Meter record:
this is for watching, not for the benchmark numbers.

    SNR_DEMO=se [SNR_LIVE=1] SNR_OUT_DIR=runs/demo/NAME .venv/bin/python benchmark/demo_bench.py SPEC SEED
SPEC as in evaluate_all.py (npz:/dqnnpz:/tabq:/rule)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "benchmark"), str(ROOT / "rl"), str(ROOT / "tools")]
import os  # noqa: E402

import eval_gazebo as G  # noqa: E402  (reads SNR_* at import)
import evaluate_all as EA  # noqa: E402
import numpy as np  # noqa: E402


class Wrapped:
    """Policy interface for eval_gazebo.run_layout (as in eval_gazebo_bench.py, whose module reads argv at import)."""
    def __init__(self, ctrl):
        self.ctrl = ctrl

    def index(self, obs):
        a = np.asarray(self.ctrl(np.atleast_2d(obs), None))
        assert a.ndim == 1, "Gazebo wrapper needs action indices"
        return a

if __name__ == "__main__":
    spec, seed = sys.argv[1], int(sys.argv[2])
    assert not spec.startswith("dwa"), "DWA needs the 2D simulator's true state"
    r = G.run_layout(Wrapped(EA.make(spec)), seed)
    t = [x for s, x in zip(r["status"], r["time"]) if s == "success"]
    print(f"seed {seed}: {r['status']}  ({r['steps']} policy steps)", flush=True)
    print(f"DEMO {spec} layout {seed}: {r['status'].count('success')}/4 success, mean time-to-goal "
          f"{sum(t) / len(t) if t else float('nan'):.1f}s", flush=True)
    sys.stdout.flush()
    os._exit(0)  # gz-transport threads can segfault during interpreter teardown (see rl/eval_gazebo.py)
