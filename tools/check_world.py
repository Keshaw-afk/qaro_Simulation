"""Four-robot arena integration test (lockstep). Run from repo root: python3 tools/check_world.py"""
import math, os, pathlib, subprocess, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gen_world, layout as L
from gzsim import GzSim

TMP = pathlib.Path(os.environ.get("TMPDIR", "/tmp"))
NAMES = [f"r{i}" for i in range(1, 5)]
ok = True


def with_sim(lay, fn):
    sdf = TMP / "check_world.sdf"
    sdf.write_text(gen_world.world_sdf(lay))
    p = subprocess.Popen(["gz", "sim", "-s", "--headless-rendering", str(sdf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        sim = GzSim(NAMES)
        sim.step(2)
        return fn(sim)
    finally:
        p.terminate()
        subprocess.run(["bash", str(pathlib.Path(__file__).parent / "killsim.sh")], capture_output=True)


def test_drive(sim):
    good = True
    p0 = {n: sim.pose(n) for n in NAMES}
    good &= all(abs(p0[n][3]) < .02 and abs(p0[n][4]) < .02 for n in NAMES)
    for n in NAMES:
        sc = sim.scan(n); fin = [v for v in sc if math.isfinite(v)]
        good &= len(sc) == 360 and all(.149 <= v <= 2.001 for v in fin)
    for n in NAMES: sim.cmd_vel(n, 0.1)
    sim.step(400)  # 2 sim-s
    for n in NAMES: sim.cmd_vel(n, 0)
    sim.step(40)
    for n in NAMES:
        x0, y0, yaw0, *_ = p0[n]; x1, y1, *_ = sim.pose(n)
        along = (x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0)
        lat = -(x1 - x0) * math.sin(yaw0) + (y1 - y0) * math.cos(yaw0)
        print(f"  {n}: along {along:+.3f} m (expect ~0.2, less if blocked) lateral {lat:+.3f}")
        good &= along > 0.05 and abs(lat) < 0.03
    return good


def test_visibility(sim):
    sc = sim.scan("r1")  # r2 is 0.8 m dead ahead facing r1; beam index 180 is straight ahead
    fwd = [(i, v) for i, v in enumerate(sc) if 150 <= i <= 210 and math.isfinite(v) and v < 1.2]
    print(f"  r1 forward beams hitting r2: {len(fwd)}, nearest {min((v for _, v in fwd), default=None)}")
    print("  (chassis top is below the scan plane, so only r2's lidar head should show: a few beams)")
    return 1 <= len(fwd) <= 8


print("A: arena + four robots drive");
a = with_sim(L.generate_layout(3), test_drive); ok &= a; print("  ->", a)
lay_b = L.Layout(seed=0, obstacles=[], starts=[(-1.0, 0, 0), (-0.2, 0, math.pi), (1.5, 1.0, 0), (1.5, -1.0, 0)], goals=[(1, 1), (1, -1), (0, 1), (0, -1)])
print("B: robot-to-robot lidar visibility")
b = with_sim(lay_b, test_visibility); ok &= b; print("  ->", b)
print("RESULT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
