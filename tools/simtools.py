"""Small helpers for talking to a running headless Gazebo via the gz CLI."""
import math
import re
import subprocess
import time


def gz_echo(topic, timeout=15):
    return subprocess.run(["gz", "topic", "-e", "-t", topic, "-n", "1"], capture_output=True, text=True, timeout=timeout).stdout


def sim_time():
    t = gz_echo("/clock")
    s = re.search(r"sim \{\s*sec: (\d+)\s*(?:nsec: (\d+))?", t)
    return int(s.group(1)) + int(s.group(2) or 0) * 1e-9


def wait_sim(seconds, poll=0.3):
    t0 = sim_time()
    while sim_time() - t0 < seconds:
        time.sleep(poll)


def poses():
    """{model_name: (x, y, yaw, pitch, roll)} from the world pose stream (top-level models only)."""
    t = gz_echo("/world/arena/pose/info")
    out = {}
    for blk in re.split(r"\npose \{", "\n" + t):
        m = re.search(r'name: "([^"]+)"', blk)
        if not m or "position" not in blk or "orientation" not in blk:
            continue
        num = lambda k, s: float(re.search(rf"\b{k}: ([-\d.e+]+)", s).group(1)) if re.search(rf"\b{k}: ([-\d.e+]+)", s) else 0.0
        pos, ori = blk[blk.index("position"):], blk[blk.index("orientation"):]
        x, y = num("x", pos), num("y", pos)
        qx, qy, qz, qw = (num(k, ori) for k in "xyzw")
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        pitch = math.degrees(math.asin(max(-1, min(1, 2 * (qw * qy - qz * qx)))))
        roll = math.degrees(math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
        out.setdefault(m.group(1), (x, y, yaw, pitch, roll))
    return out


def scan(name):
    t = gz_echo(f"/{name}/scan")
    return [float(v) for v in re.findall(r"ranges: ([-\w.+]+)", t)]


def cmd_vel(name, lin, ang=0.0):
    subprocess.run(["gz", "topic", "-t", f"/{name}/cmd_vel", "-m", "gz.msgs.Twist", "-p", f"linear: {{x: {lin}}} angular: {{z: {ang}}}"], capture_output=True)


def odom_xy(name):
    o = gz_echo(f"/{name}/odom")
    p = o[o.index("position"):]
    g = lambda k: float(re.search(rf"\b{k}: ([-\d.e+]+)", p).group(1))
    return g("x"), g("y")
