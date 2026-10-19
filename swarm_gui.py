#!/usr/bin/env python3
"""Swarm Nav RL launcher: a small native (Tk) desktop app to run the trained policies.

Start it from the project folder with:   python3 swarm_gui.py        (or ./run_gui.sh)

What it does
  * lists every policy file (*.npz) found in weights/ and runs/
  * runs the closed-loop Gazebo evaluation (rl/eval_gazebo.py) or the fast 2D evaluation (rl/eval2d.py)
    on the layouts you choose, optionally with the Gazebo window open ("live") and/or videos
  * shows the output live, summarises the result, and can stop the run or kill stray simulators

How it works: it never imports the RL code. Each run is a subprocess of the existing scripts, configured through
command-line arguments and the environment variables documented in eval_gazebo.py (SNR_LIVE, SNR_FIRST, SNR_SHIELD).
Gazebo runs need ROS sourced, so they are started through `bash -c "source /opt/ros/jazzy/setup.bash && ..."`.
With the Gazebo window open the world plays in real time; without it runs are stepped for speed.
Only one run at a time is allowed (two simulators would share topics and corrupt each other).
"""
import json
import os
import pathlib
import queue
import re
import signal
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = pathlib.Path(__file__).resolve().parent
ROS_SETUP = "/opt/ros/jazzy/setup.bash"
PY_SYSTEM = "python3"                 # Gazebo evaluation: needs ROS + gz python bindings
PY_VENV = str(ROOT / ".venv/bin/python")  # 2D evaluation: needs gymnasium / stable-baselines3


def find_models():
    """Return {label: path} for every .npz policy under weights/ and runs/ (final first, then newest first)."""
    found = []
    for d in ("weights", "runs"):
        found += [p for p in (ROOT / d).rglob("*.npz") if "pool_" not in p.name]
    found.sort(key=lambda p: (p.parent.name != "weights", -p.stat().st_mtime))
    return {str(p.relative_to(ROOT)): p for p in found}


class App(ttk.Frame):
    """Main window: a settings panel at the top, a live log below, a status bar at the bottom."""

    def __init__(self, master):
        super().__init__(master, padding=10)
        master.title("Swarm Nav RL")
        master.geometry("900x640")
        self.pack(fill="both", expand=True)
        self.proc = None                    # running subprocess, or None
        self.lines = queue.Queue()          # output lines from the reader thread
        self.models = find_models()
        self._build()
        self.after(100, self._drain)

    # ---------------------------------------------------------------- layout of the window
    def _build(self):
        box = ttk.LabelFrame(self, text="Run settings", padding=8)
        box.pack(fill="x")
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Model").grid(row=0, column=0, sticky="w")
        self.model = tk.StringVar(value=next(iter(self.models), ""))
        self.model_box = ttk.Combobox(box, textvariable=self.model, values=list(self.models), state="readonly")
        self.model_box.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(box, text="Browse...", command=self._browse).grid(row=0, column=2)

        ttk.Label(box, text="Mode").grid(row=1, column=0, sticky="w")
        self.mode = tk.StringVar(value="gazebo")
        modes = ttk.Frame(box)
        modes.grid(row=1, column=1, sticky="w", padx=6)
        ttk.Radiobutton(modes, text="Gazebo (real simulator, slow)", variable=self.mode, value="gazebo").pack(side="left")
        ttk.Radiobutton(modes, text="2D (fast, no Gazebo)", variable=self.mode, value="2d").pack(side="left", padx=12)

        ttk.Label(box, text="First layout").grid(row=2, column=0, sticky="w")
        self.first = tk.IntVar(value=0)
        ttk.Spinbox(box, from_=0, to=199, textvariable=self.first, width=6).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(box, text="Number of layouts").grid(row=3, column=0, sticky="w")
        self.count = tk.IntVar(value=1)
        ttk.Spinbox(box, from_=1, to=200, textvariable=self.count, width=6).grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(box, text="(layouts are fixed random worlds, seeds 0-199, never used in training)", foreground="gray").grid(row=2, column=2, rowspan=2, sticky="w")

        opts = ttk.Frame(box)
        opts.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.live = tk.BooleanVar(value=True)
        self.shield = tk.BooleanVar(value=True)
        self.video = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Open Gazebo window (live, Gazebo mode only)", variable=self.live).pack(side="left")
        ttk.Checkbutton(opts, text="Safety layer on", variable=self.shield).pack(side="left", padx=12)
        ttk.Checkbutton(opts, text="Save videos (Gazebo, no window)", variable=self.video).pack(side="left")

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=8)
        self.run_btn = ttk.Button(bar, text="Run", command=self.start)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Kill stray simulators", command=self.kill_sims).pack(side="left", padx=6)
        ttk.Button(bar, text="Open runs folder", command=lambda: subprocess.Popen(["xdg-open", str(ROOT / "runs")])).pack(side="left", padx=6)
        ttk.Button(bar, text="Clear log", command=lambda: self.log.delete("1.0", "end")).pack(side="right")

        self.summary = tk.StringVar(value="No run yet.")
        ttk.Label(self, textvariable=self.summary, font=("TkDefaultFont", 11, "bold"), wraplength=860).pack(fill="x")
        self.log = tk.Text(self, height=20, wrap="none", font=("monospace", 10))
        self.log.pack(fill="both", expand=True, pady=(6, 0))
        self.status = tk.StringVar(value="Idle")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(fill="x", pady=(6, 0))

    def _browse(self):
        """Pick a policy file that is not in weights/ or runs/."""
        f = filedialog.askopenfilename(initialdir=ROOT, filetypes=[("Policy weights", "*.npz")])
        if f:
            self.models[f] = pathlib.Path(f)
            self.model_box["values"] = list(self.models)
            self.model.set(f)

    # ---------------------------------------------------------------- running
    def _command(self):
        """Build the shell command and environment for the chosen settings."""
        weights = str(self.models[self.model.get()])
        first, n = self.first.get(), self.count.get()
        env = dict(os.environ, SNR_FIRST=str(first), SNR_SHIELD="1" if self.shield.get() else "0")
        if self.mode.get() == "2d":
            return f'"{PY_VENV}" rl/eval2d.py "{weights}" {n} {1 if self.shield.get() else 0}', env
        live = self.live.get()
        if live:
            env["SNR_LIVE"] = "1"
        videos = n if (self.video.get() and not live) else 0
        # eval_gazebo evaluates layouts SNR_FIRST .. N-1, so N is the end index, not the count
        return f'source {ROS_SETUP} && {PY_SYSTEM} rl/eval_gazebo.py "{weights}" {first + n} {first + videos if videos else 0}', env

    def start(self):
        if self.proc:
            return
        if not self.model.get():
            messagebox.showerror("No model", "No policy (.npz) found. Use Browse... to pick one.")
            return
        if self.mode.get() == "gazebo" and subprocess.run(["pgrep", "-f", "[g]z sim"], capture_output=True).returncode == 0:
            if not messagebox.askyesno("Simulator already running", "A Gazebo simulator is already running. Kill it and continue?"):
                return
            self.kill_sims()
        cmd, env = self._command()
        self.log.insert("end", f"$ {cmd}\n")
        self.summary.set("Running...")
        self.proc = subprocess.Popen(["bash", "-c", cmd], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.set("Running (Gazebo runs take about 1-2 minutes per layout)")

    def _reader(self, proc):
        """Background thread: push each output line to the queue; a None marks the end."""
        for line in proc.stdout:
            self.lines.put(line)
        proc.wait()
        self.lines.put(None)

    def _drain(self):
        """Runs in the Tk thread every 100 ms: move queued output into the log and react to results."""
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    self._finished()
                    continue
                self.log.insert("end", line)
                self.log.see("end")
                self._parse(line)
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _parse(self, line):
        """Pick the summary out of the scripts' final result lines."""
        if line.startswith("GAZEBO RESULT"):
            self.summary.set(line.strip())
        elif line.startswith("RESULT2D"):
            r = json.loads(line.split(" ", 1)[1])
            self.summary.set(f"2D result: success {r['success']:.1%}, collision {r['collision']:.1%}, timeout {r['timeout']:.1%}, mean time {r['mean_time_s']:.1f} s, all four succeed {r['all_four_success']:.0%}")
        elif re.match(r"seed\s+\d+:", line):
            self.status.set(line.strip()[:120])

    def _finished(self):
        code = self.proc.returncode if self.proc else None
        self.proc = None
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.set("Idle" if code == 0 else f"Stopped (exit code {code})")

    def stop(self):
        """Terminate the run (whole process group) and any simulator it started."""
        if self.proc:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.kill_sims()

    def kill_sims(self):
        """Stop every Gazebo server. Uses the helper script so the pkill pattern cannot match this app's own command line."""
        subprocess.run(["bash", str(ROOT / "tools/killsim.sh")], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "[g]z sim"], capture_output=True)
        self.status.set("Simulators stopped")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
