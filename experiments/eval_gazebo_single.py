"""Hardware case in Gazebo: ONE robot, no wheel encoders, existing policy unchanged.

    source /opt/ros/jazzy/setup.bash; python3 experiments/eval_gazebo_single.py WEIGHTS.npz N_END

Only one robot is spawned (it is r1, tag 0) at the start pose of layout robot k, and drives to robot k's goal with own-ID k.
The three absent robots are fed to the policy and the safety layer as fixed virtual robots 0.3 m outside the walls, not moving
(see experiments/no_encoder_2d.py: parking them far away is out of the training distribution and drops success to about 40%).
The two encoder inputs (own v, w) come from SNR_VW instead of the wheel odometry.

Environment variables (same meaning as rl/eval_gazebo.py where shared):
  SNR_FIRST=k     first layout seed (default 0); layouts SNR_FIRST .. N_END-1 are run
  SNR_ROBOT=0,1   which layout robots to run, one simulator launch each (default 0,1,2,3)
  SNR_VW=camera   own v, w source: camera (pose differencing, default) | cmdmodel (lagged commands) | odom (encoders, reference)
  SNR_LIVE=1      open the Gazebo window, real time
  SNR_SHIELD=0    switch the safety layer off
  SNR_OUT_DIR     where gazebo_single_results.json goes (default runs/)
  GZ_PARTITION    isolate concurrent simulators
Does not modify any existing module; reuses tools/ and rl/ as libraries."""
import json, math, os, pathlib, signal, subprocess, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "rl")]
import deploy as D  # noqa: E402
import shield as SH  # noqa: E402
from actions import CV, CW  # noqa: E402
import gen_world, layout as L, vision  # noqa: E402
from gzsim import GzSim  # noqa: E402

LIVE = bool(os.environ.get("SNR_LIVE"))
USE_SHIELD = os.environ.get("SNR_SHIELD", "1") != "0"
VW = os.environ.get("SNR_VW", "camera")
assert VW in ("camera", "cmdmodel", "odom"), VW
OUT_DIR = pathlib.Path(os.environ.get("SNR_OUT_DIR", ROOT / "runs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP = pathlib.Path(os.environ.get("TMPDIR", "/tmp"))
R_ROBOT, R_COL, GOAL_TOL = 0.095, 0.095 + 0.01, 0.15
MAX_STEPS = 1800 if L.BIG else 700
LAG = 0.75  # cmdmodel: first-order lag toward the commanded (v, w), middle of the training range U(0.5, 1.0)
VIRTUAL = np.array([(-L.ARENA_W / 2 - 0.3, 0.0), (L.ARENA_W / 2 + 0.3, 0.0), (0.0, L.ARENA_H / 2 + 0.3)])


def clearance(lay, x, y):
    gaps = [L.ARENA_W / 2 - abs(x) - R_ROBOT, L.ARENA_H / 2 - abs(y) - R_ROBOT]
    gaps += [float(L._box_dist(np.array(x), np.array(y), ob)) - R_ROBOT for ob in lay.obstacles]
    return min(gaps)


def run_one(policy, seed, k):
    full = L.generate_layout(seed)
    lay = L.generate_layout(seed)
    lay.starts, lay.goals = [full.starts[k]], [full.goals[k]]  # spawned as r1 / tag 0; other robots and goal tags absent
    sdf = TMP / f"single_{seed}_{k}.sdf"
    sdf.write_text(gen_world.world_sdf(lay, real_time=LIVE))
    simlog = TMP / f"single_{seed}_{k}.simlog"
    proc = subprocess.Popen(["gz", "sim", "-s", "--headless-rendering", str(sdf)], stdout=open(simlog, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    gui = None
    try:
        try:
            sim = GzSim(["r1"], camera=True)
        except TimeoutError as e:
            tail = simlog.read_text()[-1500:] if simlog.exists() else ""
            raise RuntimeError(f"{e}. Simulator output ({simlog}) ends with:\n{tail}") from e
        if LIVE:
            gui = subprocess.Popen(["gz", "sim", "-g"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            time.sleep(8)
        sim.step(40)
        sim.wait_for(lambda: sim._image is not None, 30, "camera")
        det = vision.detect(sim.image())
        assert 0 in det["goals"] and 0 in det["robots"], f"tags not seen: {det}"
        goal = np.array(det["goals"][0])
        if LIVE:
            sim.play()
        x, y, yaw = det["robots"][0]
        last_xy, last_yaw = np.array([x, y]), yaw
        prev, est = np.zeros(2, np.float32), np.zeros(2)
        others_vel = np.zeros((3, 2))
        status, t_done, overrides = "running", None, 0
        for step in range(MAX_STEPS):
            d = vision.detect(sim.image())
            if 0 in d["robots"]:
                x, y, yaw = d["robots"][0]
            vel = (np.array([x, y]) - last_xy) / D.DT  # world-frame velocity from the camera
            w_cam = ((yaw - last_yaw + math.pi) % (2 * math.pi) - math.pi) / D.DT
            last_xy, last_yaw = np.array([x, y]), yaw
            gx, gy = sim.pose("r1")[:2]  # ground truth, scoring only
            if math.hypot(gx - full.goals[k][0], gy - full.goals[k][1]) < GOAL_TOL:
                status, t_done = "success", step * D.DT
            elif clearance(lay, gx, gy) < R_COL - R_ROBOT:
                status, t_done = "collision", step * D.DT
            if status != "running":
                break
            if VW == "camera":
                v, w = vel[0] * math.cos(yaw) + vel[1] * math.sin(yaw), w_cam
            elif VW == "cmdmodel":
                v, w = est
            else:
                od = sim.odom("r1")
                v, w = od[3], od[4]
            obs = D.obs_from_sensors(sim.scan("r1"), (x, y), yaw, goal, VIRTUAL, vel, others_vel, v, w, prev, k)
            a_idx = policy.index(obs)[0]
            if USE_SHIELD:
                new = SH.apply(np.array([[x, y]]), np.array([yaw]), VIRTUAL[None], others_vel[None], np.array([a_idx]))[0]
                overrides += int(new != a_idx)
                a_idx = new
            prev = np.array([CV[a_idx] / D.VMAX * 2 - 1, CW[a_idx] / D.WMAX], np.float32)
            est += (np.array([CV[a_idx], CW[a_idx]]) - est) * LAG
            sim.cmd_vel("r1", *D.action_to_cmd(prev))
            sim.step(20)
        sim.cmd_vel("r1", 0, 0)
        if status == "running":
            status = "timeout"
        return {"seed": seed, "robot": k, "status": status, "time": t_done, "steps": step, "shield_overrides": overrides, "vw": VW}
    finally:
        if gui is not None:
            gui.terminate()
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    policy = D.Policy(sys.argv[1])
    n = int(sys.argv[2])
    robots = [int(r) for r in os.environ.get("SNR_ROBOT", "0,1,2,3").split(",")]
    results, t0 = [], time.time()
    for seed in range(int(os.environ.get("SNR_FIRST", 0)), n):
        for k in robots:
            r = run_one(policy, seed, k)
            results.append(r)
            print(f"seed {seed:3d} robot {k}: {r['status']:9s} ({r['steps']} steps, shield overrides {r['shield_overrides']}, {time.time() - t0:.0f}s elapsed)", flush=True)
    st = [r["status"] for r in results]
    tt = [r["time"] for r in results if r["status"] == "success"]
    print(f"GAZEBO SINGLE RESULT (v,w from {VW}) over {len(st)} single-robot runs: success {st.count('success')/len(st):.3f}, "
          f"collision {st.count('collision')/len(st):.3f}, timeout {st.count('timeout')/len(st):.3f}, mean time-to-goal {np.mean(tt) if tt else float('nan'):.1f}s")
    with open(OUT_DIR / f"gazebo_single_{VW}.json", "w") as f:
        json.dump(results, f)
    sys.stdout.flush()
    os._exit(0)
