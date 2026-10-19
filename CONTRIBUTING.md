# Contributing

Humans and AI agents follow the same workflow. `AGENTS.md` has the rules and commands; this file is the short procedure.

1. **Set up**: `docs/SETUP.md`, then `make check-env`. Training/2D work also needs `.venv` (`pip install -r requirements.txt`).
2. **Branch**, make a focused change. Keep the style of the surrounding code: module docstring (purpose + usage), named and commented constants, small functions with one-line docstrings.
3. **Test**: `make test` (fast, no simulator). Add a test next to the existing ones in `tests/` for new logic (they are plain scripts that `assert` and print a pass line).
4. **Check behaviour** when the policy path is touched:
   - `make eval-2d` before and after (100 held-out layouts).
   - A Gazebo run on several layouts (`make eval-small FIRST=1 N=5`); for anything that could depend on scale also `make eval-big FIRST=51 N=3`.
   - Quote counts (e.g. "73/80 robots"), not only percentages, and say what you did not check.
5. **Docs**: update `docs/DOCUMENTATION.md` (behaviour/results), `docs/CODE_GUIDE.md` (structure), `CHANGELOG.md`. If you add an environment variable or a command-line option, document it in the docstring of the script that reads it.
6. **Commit**: imperative subject line, body explains why. No generated files (`runs/`, pools, videos).
7. **Retraining checklist** (observation, action or reward changes): see "Common changes" in `docs/CODE_GUIDE.md`; keep seeds 0-199 evaluation-only; save the new curve as `docs/<run>_eval.csv` and the evaluation logs under `docs/results/`; only replace `weights/final_policy.npz` if the new policy beats the current numbers on the same layouts.
