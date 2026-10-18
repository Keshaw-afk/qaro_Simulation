"""Step-1 acceptance test against a running sim (world with robot 'r1' at rest, RTF=1).
Checks: upright pose, lidar range clipping, 0.1 m/s * 5 s ~= 0.5 m odometry, wheel encoders move."""
import math
import re
import subprocess
import sys
import time

NAME = sys.argv[1] if len(sys.argv) > 1 else "r1"


def gz_echo(topic, timeout=10):
    return subprocess.run(["gz", "topic", "-e", "-t", topic, "-n", "1"], capture_output=True, text=True, timeout=timeout).stdout


def pose_of(name):
    t = gz_echo("/world/arena/pose/info")
    blk = t[t.index(f'name: "{name}"'):]
    num = lambda k, s: float(re.search(rf"{k}: ([-\d.e+]+)", s).group(1))
    pos, ori = blk[blk.index("position"):], blk[blk.index("orientation"):]
    q = [num(k, ori) for k in "xyzw"]
    pitch = math.degrees(math.asin(max(-1, min(1, 2 * (q[3] * q[1] - q[2] * q[0])))))
    return num("x", pos), num("y", pos), pitch


def clock_sim():
    t = gz_echo("/clock")
    s = re.search(r"sim \{\s*sec: (\d+)\s*(?:nsec: (\d+))?", t)
    return int(s.group(1)) + int(s.group(2) or 0) * 1e-9


ok = True
x0, y0, pitch = pose_of(NAME)
print(f"rest pose x={x0:.3f} y={y0:.3f} pitch={pitch:.2f} deg"); ok &= abs(pitch) < 1.0

t = gz_echo(f"/{NAME}/scan")
r = [float(v) for v in re.findall(r"ranges: ([-\w.+]+)", t)]
fin = [v for v in r if v == v and abs(v) != float("inf")]
print(f"lidar beams={len(r)} finite={len(fin)} min={min(fin, default=float('nan')):.3f} max={max(fin, default=float('nan')):.3f}")
ok &= len(r) == 360 and all(0.15 - 1e-3 <= v <= 2.0 + 1e-3 for v in fin)

j0 = gz_echo(f"/{NAME}/joint_states")
ts0 = clock_sim()
subprocess.run(["gz", "topic", "-t", f"/{NAME}/cmd_vel", "-m", "gz.msgs.Twist", "-p", "linear: {x: 0.1}"])
time.sleep(5)
subprocess.run(["gz", "topic", "-t", f"/{NAME}/cmd_vel", "-m", "gz.msgs.Twist", "-p", "linear: {x: 0.0}"])
ts1 = clock_sim()
x1, y1, pitch1 = pose_of(NAME)
dist = math.hypot(x1 - x0, y1 - y0)
fwd = x1 - x0  # robot starts at yaw 0, so world +x is forward
o = gz_echo(f"/{NAME}/odom")
odom_x = float(re.search(r"position \{\s*x: ([-\d.e+]+)", o).group(1))
print(f"forward displacement (ground truth) {fwd:+.3f} m, odom {odom_x:+.3f} m, mismatch {abs(fwd - odom_x) * 1000:.1f} mm")
j1 = gz_echo(f"/{NAME}/joint_states")
pos = lambda s: [float(v) for v in re.findall(r"position: ([-\d.e+]+)", s)]
print(f"moved {dist:.3f} m in ~{ts1 - ts0:.1f} sim-s (expect ~0.5), pitch {pitch1:.2f}")
print("wheel encoder positions before/after:", pos(j0)[:2], pos(j1)[:2])
ok &= 0.35 < fwd < 0.75 and abs(fwd - odom_x) < 0.03 and abs(y1 - y0) < 0.03 and abs(pitch1) < 2.0 and pos(j1) != pos(j0)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
