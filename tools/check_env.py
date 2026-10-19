"""Check that this machine has what the project needs. Run: source /opt/ros/jazzy/setup.bash; python3 tools/check_env.py
Prints OK/MISSING per item and exits non-zero if something needed for the Gazebo evaluation is missing."""
import importlib
import shutil
import subprocess
import sys

NEEDED = [  # (what, test) for the Gazebo evaluation and the desktop app
    ("python module cv2 (OpenCV, with aruco)", lambda: importlib.import_module("cv2").aruco),
    ("python module numpy", lambda: importlib.import_module("numpy")),
    ("python module scipy", lambda: importlib.import_module("scipy")),
    ("python module yaml", lambda: importlib.import_module("yaml")),
    ("python module tkinter (desktop app)", lambda: importlib.import_module("tkinter")),
    ("gz-transport python bindings (gz.transport13, gz.msgs10)", lambda: (importlib.import_module("gz.transport13"), importlib.import_module("gz.msgs10"))),
    ("`gz` command (Gazebo Harmonic)", lambda: shutil.which("gz") or (_ for _ in ()).throw(RuntimeError("not on PATH"))),
]
OPTIONAL = [  # training / 2D evaluation only
    ("stable_baselines3", lambda: importlib.import_module("stable_baselines3")),
    ("gymnasium", lambda: importlib.import_module("gymnasium")),
    ("torch", lambda: importlib.import_module("torch")),
]


def run(items):
    bad = 0
    for what, test in items:
        try:
            test()
            print(f"OK       {what}")
        except Exception as e:  # any failure means missing or broken
            bad += 1
            print(f"MISSING  {what}  ({type(e).__name__}: {e})")
    return bad


if __name__ == "__main__":
    print("== needed for the Gazebo evaluation and the app ==")
    bad = run(NEEDED)
    out = subprocess.run(["gz", "sim", "--versions"], capture_output=True, text=True).stdout.strip() if shutil.which("gz") else ""
    print(f"gz sim version: {out or 'unknown'} (project developed on Gazebo Harmonic, gz sim 8.x)")
    print("== optional: training and 2D evaluation (use the venv) ==")
    run(OPTIONAL)
    sys.exit(1 if bad else 0)
