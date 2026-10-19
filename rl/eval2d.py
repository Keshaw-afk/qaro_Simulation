"""Command-line evaluation in the fast 2D environment (no Gazebo needed).

Usage: .venv/bin/python rl/eval2d.py WEIGHTS.npz [N_LAYOUTS=100] [SHIELD=1]
Prints one line per metric and a final 'RESULT2D {json}' line (parsed by the GUI app)."""
import json
import sys

import deploy
import evaluate as EV
import pool as PL

if __name__ == "__main__":
    weights = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    shield = (sys.argv[3] if len(sys.argv) > 3 else "1") != "0"
    pool = PL.build(PL.EVAL_SEEDS, PL.pathlib.Path(__file__).parent / "pool_eval.npz")
    r = EV.run_eval(pool, deploy.Policy(weights), n_layouts=n, shield=shield)
    for k, v in r.items():
        print(f"{k:18s} {v:.3f}")
    print("RESULT2D " + json.dumps(r))
