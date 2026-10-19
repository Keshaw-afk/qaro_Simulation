"""Run a trained policy in Gazebo on the evaluation layouts. Closed loop: camera tags -> poses/goals, lidar, odometry.

    source /opt/ros/jazzy/setup.bash; python3 rl/eval_gazebo.py WEIGHTS.npz N_END [N_VIDEO_END]

Layouts SNR_FIRST .. N_END-1 are run (SNR_FIRST defaults to 0). Videos are saved for layouts below N_VIDEO_END.
Environment variables:
  SNR_ARENA=big   large 8 x 6 m arena instead of the 4 x 3 m training arena (read by tools/layout.py)
  SNR_LIVE=1      open the Gazebo window; the world plays in real time instead of being stepped while paused
  SNR_SHIELD=0    switch the inter-robot safety layer off
  SNR_FIRST=k     first layout seed
  SNR_OUT_DIR     folder for videos and gazebo_results.json (default runs/)
  GZ_PARTITION    standard Gazebo variable: runs with different values cannot see each other (so several runs can coexist)
Prints one 'seed k: [...]' line per layout and a final 'GAZEBO RESULT ...' line."""
import json, math, os, pathlib, signal, subprocess, sys, time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "rl")]
import deploy as D
import shield as SH
from actions import CV, CW
LIVE = bool(os.environ.get("SNR_LIVE"))              # open the Gazebo GUI; the world plays in real time (not lockstep)
OUT_DIR = pathlib.Path(os.environ.get("SNR_OUT_DIR", ROOT / "runs"))  # where videos and gazebo_results.json go (the GUI app uses its own folder)
OUT_DIR.mkdir(parents=True, exist_ok=True)
USE_SHIELD = os.environ.get("SNR_SHIELD", "1") != "0"  # inter-robot safety layer on/off
import gen_world, layout as L, vision
from gzsim import GzSim

NAMES = [f"r{i}" for i in range(1, 5)]
R_COL, GOAL_TOL = 0.095 + 0.01, 0.15
MAX_STEPS = 1800 if L.BIG else 700  # policy steps of 0.1 s: 180 s in the big arena, 70 s otherwise
TMP = pathlib.Path(os.environ.get("TMPDIR", "/tmp"))


def clearance(lay, pos, i):
    """Ground-truth gap between robot i's chassis and the nearest obstacle, wall or other robot (metres)."""
    x, y = pos[i]
    gaps = [L.ARENA_W / 2 - abs(x) - 0.095, L.ARENA_H / 2 - abs(y) - 0.095]
    gaps += [float(L._box_dist(np.array(x), np.array(y), ob)) - 0.095 for ob in lay.obstacles]
    gaps += [math.hypot(x - pos[j][0], y - pos[j][1]) - 0.19 for j in range(4) if j != i]
    return min(gaps)


def run_layout(policy, seed, video=None):
    lay = L.generate_layout(seed)
    sdf = TMP / f"eval_{seed}.sdf"
    sdf.write_text(gen_world.world_sdf(lay, real_time=LIVE))
    proc = subprocess.Popen(["gz", "sim", "-s", "--headless-rendering", str(sdf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    gui = None
    try:
        sim = GzSim(NAMES, camera=True)
        if LIVE:  # watch it: attach the Gazebo GUI to this server (in live mode the world is playing in real time; otherwise it is stepped while paused)
            gui = subprocess.Popen(["gz", "sim", "-g"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            time.sleep(8)
        sim.step(40)
        sim.wait_for(lambda: sim._image is not None, 30, "camera")
        det = vision.detect(sim.image())
        if LIVE:
            sim.play()  # live mode: the simulator really plays (real time, the GUI shows "playing", lidar can be visualised)
        goals = {i: np.array(det["goals"][i]) for i in det["goals"]}
        assert len(goals) == 4, f"only {len(goals)} goal tags seen"
        cam = {i: det["robots"][i] for i in det["robots"]}
        prev = {i: np.zeros(2, np.float32) for i in range(4)}
        last_xy = {i: np.array(cam[i][:2]) for i in range(4)}
        vel = {i: np.zeros(2) for i in range(4)}
        status = {i: "running" for i in range(4)}
        t_done = {i: None for i in range(4)}
        writer = None
        if video:
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
        for step in range(MAX_STEPS):
            d = vision.detect(sim.image())
            cam.update(d["robots"])
            for i in range(4):  # world-frame velocity by differencing consecutive camera frames
                vel[i] = (np.array(cam[i][:2]) - last_xy[i]) / D.DT
                last_xy[i] = np.array(cam[i][:2])
            gt = [sim.pose(n)[:2] for n in NAMES]
            for i in range(4):
                if status[i] != "running":
                    continue
                if math.hypot(gt[i][0] - lay.goals[i][0], gt[i][1] - lay.goals[i][1]) < GOAL_TOL:
                    status[i], t_done[i] = "success", step * D.DT
                elif clearance(lay, gt, i) < R_COL - 0.095:
                    status[i], t_done[i] = "collision", step * D.DT
            if writer is not None:
                f = cv2.cvtColor(sim.image(), cv2.COLOR_RGB2BGR)
                for i in range(4):
                    px = lambda p: (int(gen_world.CAM_W / 2 + p[0] / (vision.CAM_Z - vision.TAG_Z) * vision.F_PX), int(gen_world.CAM_H / 2 - p[1] / (vision.CAM_Z - vision.TAG_Z) * vision.F_PX))
                    cv2.circle(f, px(lay.goals[i]), 22, (255, 255, 255), 2)
                writer.write(cv2.resize(f, (640, 480)))
            if all(s != "running" for s in status.values()):
                break
            for i, n in enumerate(NAMES):
                if status[i] != "running":
                    sim.cmd_vel(n, 0.0, 0.0)
                    continue
                x, y, yaw = cam[i]
                others = np.array([cam[j][:2] for j in range(4) if j != i])
                others_vel = np.array([vel[j] for j in range(4) if j != i])
                od = sim.odom(n)
                obs = D.obs_from_sensors(sim.scan(n), (x, y), yaw, goals[i], others, vel[i], others_vel, od[3], od[4], prev[i], i)
                k = policy.index(obs)[0]
                if USE_SHIELD:  # inter-robot safety layer (SNR_SHIELD=0 turns it off)
                    k = SH.apply(np.array([[x, y]]), np.array([yaw]), others[None], others_vel[None], np.array([k]))[0]
                a = np.array([CV[k] / D.VMAX * 2 - 1, CW[k] / D.WMAX], np.float32)
                prev[i] = a.astype(np.float32)
                sim.cmd_vel(n, *D.action_to_cmd(a))
            sim.step(20)
        for n in NAMES:
            sim.cmd_vel(n, 0, 0)
        if writer is not None:
            writer.release()
        for i in range(4):
            if status[i] == "running":
                status[i] = "timeout"
        return {"seed": seed, "status": [status[i] for i in range(4)], "time": [t_done[i] for i in range(4)], "steps": step}
    finally:
        if gui is not None:
            gui.terminate()
        try:  # stop only this run's simulator (its own process group), so other simulators, e.g. one being watched live, keep running
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so "Stop" in the GUI app unwinds through run_layout's finally and stops its own simulator
    policy = D.Policy(sys.argv[1])
    n = int(sys.argv[2]); n_vid = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    results, t0 = [], time.time()
    for k in range(int(os.environ.get("SNR_FIRST", 0)), n):
        vid = OUT_DIR / f"gazebo_seed{k}.mp4" if k < n_vid else None
        r = run_layout(policy, k, vid)
        results.append(r)
        print(f"seed {k:3d}: {r['status']}  ({r['steps']} policy steps, {time.time() - t0:.0f}s elapsed)", flush=True)
    st = [s for r in results for s in r["status"]]
    tt = [t for r in results for s, t in zip(r["status"], r["time"]) if s == "success"]
    print(f"GAZEBO RESULT over {len(results)} layouts x 4 robots: success {st.count('success')/len(st):.3f}, collision {st.count('collision')/len(st):.3f}, "
          f"timeout {st.count('timeout')/len(st):.3f}, mean time-to-goal {np.mean(tt) if tt else float('nan'):.1f}s, all-four-success {np.mean([all(s == 'success' for s in r['status']) for r in results]):.2f}")
    json.dump(results, open(OUT_DIR / "gazebo_results.json", "w"))
