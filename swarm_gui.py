#!/usr/bin/env python3
"""Swarm Nav RL launcher: a small native (Tk) desktop app to run the trained policies.

Start it from the project folder with:   python3 swarm_gui.py        (or ./run_gui.sh)

What it does
  * lists every policy file (*.npz) found in weights/ and runs/
  * runs the closed-loop Gazebo evaluation (rl/eval_gazebo.py) in the standard 4 x 3 m arena or the big 8 x 6 m arena,
    or the fast 2D evaluation (rl/eval2d.py, standard arena only), optionally with the Gazebo window open and/or videos
  * shows the output live, summarises the result, and can stop its own run

Safety design (so nothing you did not start gets disturbed)
  * Every run gets its own Gazebo "partition" (GZ_PARTITION=snr_gui_<app pid>). Runs in different partitions cannot see
    each other, so this app never fights another evaluation or a simulator you started elsewhere.
  * Stop only ends THIS app's run and the simulator of that run (found by its partition). It asks for confirmation.
  * There is deliberately no "kill all simulators" button. "Clean up leftovers" only offers simulators that no running
    evaluation owns, shows them, and asks first.
  * Settings are locked while a run is active; closing the window during a run asks first.
  * If another evaluation is already running (for example a long background test), starting a second one asks first.
  * Results and videos go to runs/gui/, never over the files of other runs.

How it works: it never imports the RL code. Each run is a subprocess of the existing scripts, configured through
command-line arguments and the environment variables documented at the top of rl/eval_gazebo.py. Gazebo runs need ROS
sourced, so they start through `bash -c "source /opt/ros/jazzy/setup.bash && ..."`. Only one run at a time per app.
"""
import json
import os
import pathlib
import queue
import re
import signal
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = pathlib.Path(__file__).resolve().parent
ROS_SETUP = "/opt/ros/jazzy/setup.bash"
PY_SYSTEM = "python3"                      # Gazebo evaluation: needs ROS + gz python bindings
PY_VENV = str(ROOT / ".venv/bin/python")   # 2D evaluation: needs gymnasium / stable-baselines3
OUT_DIR = ROOT / "runs" / "gui"            # this app's videos and result files
PARTITION = f"snr_gui_{os.getpid()}"       # isolates this app's simulator from every other one

# Rough time per layout (seconds, from earlier runs): used only for the estimate shown to the user.
SECONDS_PER_LAYOUT = {"standard": 60, "big": 180}
ARENA_TEXT = {"standard": "Standard 4 x 3 m (the arena the model was trained in)",
              "big": "Big 8 x 6 m (4x the area, never trained in; slower)"}


def find_models():
    """Return {label: path} for every .npz policy under weights/ and runs/ (weights first, then newest first)."""
    found = []
    for d in ("weights", "runs"):
        found += [p for p in (ROOT / d).rglob("*.npz") if "pool_" not in p.name]
    found.sort(key=lambda p: (p.parent.name != "weights", -p.stat().st_mtime))
    return {str(p.relative_to(ROOT)): p for p in found}


# ------------------------------------------------------------------ process helpers (read /proc, no extra packages)
def _processes():
    """Yield (pid, cmdline, environ dict) for this user's processes; unreadable ones are skipped."""
    for d in pathlib.Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            env = dict(kv.split("=", 1) for kv in (d / "environ").read_bytes().decode(errors="replace").split("\0") if "=" in kv)
        except OSError:
            continue
        yield int(d.name), cmd, env


def evaluations():
    """{partition: pid} of every running python process of rl/eval_gazebo.py (partition '' = default).
    Only real python processes count, not shells whose command line merely mentions the script."""
    out = {}
    for pid, cmd, env in _processes():
        parts = cmd.split()
        if len(parts) > 1 and parts[0].endswith(("python3", "python")) and parts[1].endswith("rl/eval_gazebo.py"):
            out[env.get("GZ_PARTITION", "")] = pid
    return out


def simulators(partition=None):
    """List of (pid, cmd, partition) of running `gz sim` processes; optionally only those in one partition."""
    out = []
    for pid, cmd, env in _processes():
        if re.match(r"gz sim\b", cmd):
            part = env.get("GZ_PARTITION", "")
            if partition is None or part == partition:
                out.append((pid, cmd, part))
    return out


def kill_pids(pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


class App(ttk.Frame):
    """Main window: a settings panel at the top, a live log below, a status bar at the bottom."""

    def __init__(self, master):
        super().__init__(master, padding=10)
        self.master = master
        master.title("Swarm Nav RL")
        master.geometry("980x700")
        master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.pack(fill="both", expand=True)
        self.proc = None                    # running subprocess, or None
        self.lines = queue.Queue()          # output lines from the reader thread
        self.models = find_models()
        self.locked = []                    # widgets disabled while a run is active
        self._build()
        self._on_mode()
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
        browse = ttk.Button(box, text="Browse...", command=self._browse)
        browse.grid(row=0, column=2)

        ttk.Label(box, text="Simulator").grid(row=1, column=0, sticky="w")
        self.mode = tk.StringVar(value="gazebo")
        modes = ttk.Frame(box)
        modes.grid(row=1, column=1, sticky="w", padx=6)
        self.mode_gz = ttk.Radiobutton(modes, text="Gazebo (real simulator, slow)", variable=self.mode, value="gazebo", command=self._on_mode)
        self.mode_2d = ttk.Radiobutton(modes, text="2D (fast, standard arena only)", variable=self.mode, value="2d", command=self._on_mode)
        self.mode_gz.pack(side="left")
        self.mode_2d.pack(side="left", padx=12)

        ttk.Label(box, text="Arena").grid(row=2, column=0, sticky="w")
        self.arena = tk.StringVar(value="standard")
        arenas = ttk.Frame(box)
        arenas.grid(row=2, column=1, sticky="w", padx=6)
        self.arena_std = ttk.Radiobutton(arenas, text=ARENA_TEXT["standard"], variable=self.arena, value="standard", command=self._on_mode)
        self.arena_big = ttk.Radiobutton(arenas, text=ARENA_TEXT["big"], variable=self.arena, value="big", command=self._on_mode)
        self.arena_std.pack(anchor="w")
        self.arena_big.pack(anchor="w")

        ttk.Label(box, text="First layout").grid(row=3, column=0, sticky="w")
        self.first = tk.StringVar(value="1")
        first = ttk.Spinbox(box, from_=0, to=999, textvariable=self.first, width=6, command=self._on_mode)
        first.grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(box, text="Number of layouts").grid(row=4, column=0, sticky="w")
        self.count = tk.StringVar(value="1")
        count = ttk.Spinbox(box, from_=1, to=200, textvariable=self.count, width=6, command=self._on_mode)
        count.grid(row=4, column=1, sticky="w", padx=6)
        for e in (first, count):
            e.bind("<KeyRelease>", lambda _e: self._on_mode())
        ttk.Label(box, text="A layout is a fixed random world (its number is the random seed).\nStandard arena: use 0-199 (never used in training).",
                  foreground="gray").grid(row=3, column=2, rowspan=2, sticky="w")

        opts = ttk.Frame(box)
        opts.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.live = tk.BooleanVar(value=True)
        self.shield = tk.BooleanVar(value=True)
        self.video = tk.BooleanVar(value=False)
        self.live_chk = ttk.Checkbutton(opts, text="Open Gazebo window (plays in real time)", variable=self.live, command=self._on_mode)
        self.shield_chk = ttk.Checkbutton(opts, text="Safety layer on (off = robots may hit each other)", variable=self.shield)
        self.video_chk = ttk.Checkbutton(opts, text="Save a video of each layout", variable=self.video)
        self.live_chk.pack(side="left")
        self.shield_chk.pack(side="left", padx=12)
        self.video_chk.pack(side="left")
        self.estimate = tk.StringVar()
        ttk.Label(box, textvariable=self.estimate, foreground="#555").grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.locked = [self.model_box, browse, self.mode_gz, self.mode_2d, self.arena_std, self.arena_big, first, count, self.live_chk, self.shield_chk, self.video_chk]

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=8)
        self.run_btn = ttk.Button(bar, text="Run", command=self.start)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop this run", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Clean up leftovers...", command=self.cleanup).pack(side="left", padx=6)
        ttk.Button(bar, text="Open results folder", command=self._open_results).pack(side="left", padx=6)
        ttk.Button(bar, text="Clear log", command=lambda: self.log.delete("1.0", "end")).pack(side="right")

        self.summary = tk.StringVar(value="No run yet.")
        ttk.Label(self, textvariable=self.summary, font=("TkDefaultFont", 11, "bold"), wraplength=940).pack(fill="x")
        logf = ttk.Frame(self)
        logf.pack(fill="both", expand=True, pady=(6, 0))
        self.log = tk.Text(logf, height=18, wrap="none", font=("monospace", 10))
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.status = tk.StringVar(value="Idle")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(fill="x", pady=(6, 0))

    def _browse(self):
        """Pick a policy file that is not in weights/ or runs/."""
        f = filedialog.askopenfilename(initialdir=ROOT, filetypes=[("Policy weights", "*.npz")])
        if f:
            self.models[f] = pathlib.Path(f)
            self.model_box["values"] = list(self.models)
            self.model.set(f)

    def _open_results(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["xdg-open", str(OUT_DIR)])

    def _on_mode(self):
        """Keep options consistent: 2D mode only knows the standard arena and has no window or videos; refresh the estimate."""
        two_d = self.mode.get() == "2d"
        if two_d:
            self.arena.set("standard")
        self.arena_big.config(state="disabled" if two_d else "normal")
        self.live_chk.config(state="disabled" if two_d else "normal")
        self.video_chk.config(state="disabled" if two_d else "normal")
        try:
            n = max(1, int(self.count.get()))
        except ValueError:
            n = 1
        if two_d:
            self.estimate.set(f"Estimated time: a few seconds for {n} layouts.")
        else:
            secs = n * SECONDS_PER_LAYOUT[self.arena.get()] + (10 if self.live.get() else 0)
            self.estimate.set(f"Estimated time: about {max(1, round(secs / 60))} min for {n} layout(s) (rough; layouts with a stuck robot take up to {'3' if self.arena.get() == 'big' else '1.2'} min each).")

    # ---------------------------------------------------------------- running
    def _read_settings(self):
        """Validate the number fields; return (first, count) or None after telling the user what is wrong."""
        try:
            first, count = int(self.first.get()), int(self.count.get())
        except ValueError:
            messagebox.showerror("Invalid number", "First layout and Number of layouts must be whole numbers.")
            return None
        if first < 0 or not 1 <= count <= 200:
            messagebox.showerror("Invalid number", "First layout must be 0 or more, and Number of layouts between 1 and 200.")
            return None
        if self.arena.get() == "standard" and first + count > 200:
            if not messagebox.askyesno("Layouts overlap the training set", "Standard-arena layouts above 199 were used for training, so results on them are optimistic. Continue anyway?"):
                return None
        return first, count

    def _command(self, first, count):
        """Build the shell command and environment for the chosen settings."""
        weights = str(self.models[self.model.get()])
        env = dict(os.environ, SNR_FIRST=str(first), SNR_SHIELD="1" if self.shield.get() else "0", SNR_OUT_DIR=str(OUT_DIR), GZ_PARTITION=PARTITION)
        if self.mode.get() == "2d":
            return f'"{PY_VENV}" rl/eval2d.py "{weights}" {count} {1 if self.shield.get() else 0}', env
        if self.arena.get() == "big":
            env["SNR_ARENA"] = "big"
        if self.live.get():
            env["SNR_LIVE"] = "1"
        videos = first + count if self.video.get() else 0  # eval_gazebo saves videos for layouts below this number
        # eval_gazebo evaluates layouts SNR_FIRST .. N_END-1, so its argument is the end index, not the count
        return f'source {ROS_SETUP} && {PY_SYSTEM} rl/eval_gazebo.py "{weights}" {first + count} {videos}', env

    def start(self):
        if self.proc:
            return
        if not self.model.get():
            messagebox.showerror("No model", "No policy (.npz) found. Use Browse... to pick one.")
            return
        settings = self._read_settings()
        if settings is None:
            return
        if self.mode.get() == "gazebo":
            others = evaluations()
            if others and not messagebox.askyesno("Another evaluation is running",
                                                  "A Gazebo evaluation (for example a long background test) is already running.\nYour run will not disturb it, but both will be slower.\n\nStart anyway?", default="no"):
                return
        if not self.shield.get() and not messagebox.askyesno("Safety layer off", "With the safety layer off the robots can collide with each other much more often.\n\nRun anyway?", default="no"):
            return
        cmd, env = self._command(*settings)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.log.insert("end", f"$ {cmd}\n")
        self.summary.set("Running...")
        self.proc = subprocess.Popen(["bash", "-c", cmd], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self._set_running(True)
        self.status.set("Running. Layouts restart the simulator one after another; that is normal.")

    def _set_running(self, running):
        """Lock the settings and swap the Run/Stop buttons while a run is active."""
        for w in self.locked:
            w.config(state="disabled" if running else ("readonly" if isinstance(w, ttk.Combobox) else "normal"))
        self.run_btn.config(state="disabled" if running else "normal")
        self.stop_btn.config(state="normal" if running else "disabled")
        if not running:
            self._on_mode()  # re-apply the per-mode enabling of the options

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
        """Pick the summary out of the scripts' result lines."""
        if line.startswith("GAZEBO RESULT"):
            self.summary.set(line.strip())
        elif line.startswith("RESULT2D"):
            r = json.loads(line.split(" ", 1)[1])
            self.summary.set(f"2D result: success {r['success']:.1%}, collision {r['collision']:.1%}, timeout {r['timeout']:.1%}, mean time {r['mean_time_s']:.1f} s, all four succeed {r['all_four_success']:.0%}")
        elif re.match(r"seed\s+\d+:", line):
            self.status.set("Finished: " + line.strip()[:110])

    def _finished(self):
        code = self.proc.returncode if self.proc else None
        self.proc = None
        self._set_running(False)
        self.status.set("Idle (run finished)" if code == 0 else f"Run ended (exit code {code})")
        self._remove_own_simulators()

    def _remove_own_simulators(self):
        """Make sure no simulator of this app's partition survives its run (found by partition, so nothing else is touched)."""
        kill_pids([pid for pid, _, _ in simulators(PARTITION)])

    def stop(self):
        """Ask first, then end this app's run (SIGTERM so the evaluator cleans up; SIGKILL if it does not react)."""
        if not self.proc or not messagebox.askyesno("Stop this run?", "Stop the run you started from this window?\nThe layouts done so far are kept in the log; the rest is lost.", default="no"):
            return
        self._terminate()

    def _terminate(self):
        proc = self.proc
        if not proc:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        self.status.set("Stopping...")

        def force():
            if self.proc is proc and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._remove_own_simulators()
        self.after(6000, force)

    def cleanup(self):
        """Offer to remove simulators that no running evaluation owns (e.g. left over after a crash). Never touches an active run."""
        live_partitions = set(evaluations())
        owned = {PARTITION} if self.proc else set()
        left = [(pid, cmd, part) for pid, cmd, part in simulators() if part not in live_partitions and part not in owned]
        if not left:
            messagebox.showinfo("Clean up leftovers", "No leftover simulators found.\n(Simulators belonging to running evaluations are never listed.)")
            return
        text = "\n".join(f"pid {pid}: {cmd[:70]}" for pid, cmd, _ in left)
        if messagebox.askyesno("Clean up leftovers", f"These simulators belong to no running evaluation:\n\n{text}\n\nKill them?", default="no"):
            kill_pids([pid for pid, _, _ in left])
            self.status.set(f"Removed {len(left)} leftover simulator(s)")

    def on_close(self):
        """Closing the window during a run asks first and then stops the run cleanly."""
        if self.proc:
            if not messagebox.askyesno("Run in progress", "A run is still going. Stop it and quit?", default="no"):
                return
            self._terminate()
            time.sleep(1.0)
            self._remove_own_simulators()
        self.master.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
