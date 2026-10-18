"""Overhead camera vs ground truth over several random layouts. Run from repo root."""
import math, os, pathlib, subprocess, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cv2
import gen_world, layout as L, vision
from gzsim import GzSim

TMP = pathlib.Path(os.environ.get("TMPDIR", "/tmp"))
NAMES = [f"r{i}" for i in range(1, 5)]
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
pos_err, yaw_err, missed, total = [], [], 0, 0
for k, seed in enumerate([3, 11, 12]):
    lay = L.generate_layout(seed)
    (TMP / "cv.sdf").write_text(gen_world.world_sdf(lay))
    p = subprocess.Popen(["gz", "sim", "-s", "--headless-rendering", str(TMP / "cv.sdf")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        sim = GzSim(NAMES, camera=True)
        sim.step(60)
        sim.wait_for(lambda: sim._image is not None, 20, "camera frame")
        img = sim.image()
        if k == 0:
            cv2.imwrite(str(TMP / "overhead.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        det = vision.detect(img)
        for i, n in enumerate(NAMES):
            total += 2
            gx, gy, gyaw, *_ = sim.pose(n)
            if i in det["robots"]:
                x, y, yaw = det["robots"][i]
                pos_err.append(math.hypot(x - gx, y - gy)); yaw_err.append(wrap(yaw - gyaw))
            else:
                missed += 1
            if i in det["goals"]:
                pos_err.append(math.hypot(det["goals"][i][0] - lay.goals[i][0], det["goals"][i][1] - lay.goals[i][1]))
            else:
                missed += 1
    finally:
        p.terminate(); subprocess.run(["bash", str(pathlib.Path(__file__).parent / "killsim.sh")], capture_output=True)
import numpy as np
print(f"detected {total - missed}/{total} tags (4 robots + 4 goals x 3 layouts)")
if pos_err:
    print(f"position error: mean {np.mean(pos_err)*100:.1f} cm, max {np.max(pos_err)*100:.1f} cm")
    ye = np.array(yaw_err); print(f"yaw error (raw, before offset): mean {np.degrees(np.angle(np.exp(1j*ye).mean())):.1f} deg, spread (std) {np.degrees(ye.std()):.1f} deg")
