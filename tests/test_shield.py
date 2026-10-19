"""Safety-layer tests (numpy only). Run: python3 tests/test_shield.py"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "rl"))
import shield as SH
from actions import CV, CW, N_ACT

assert N_ACT == 45 and len(CV) == len(CW) == 45
fast_straight = int(np.argmin((CV - 0.20) ** 2 + CW ** 2))   # full speed, no turning

# head-on: robot at the origin heading +x, another 0.9 m ahead driving toward it: going straight must be vetoed
pos, yaw = np.zeros((1, 2)), np.zeros(1)
opos, ovel = np.array([[[0.9, 0.0]]]), np.array([[[-0.2, 0.0]]])
out = SH.apply(pos, yaw, opos, ovel, np.array([fast_straight]))[0]
assert out != fast_straight, "head-on course was not vetoed"
assert SH.clearance(pos, yaw, opos, ovel)[0, out] > SH.MARGIN, "replacement is not safe"

# nothing nearby: the policy's choice is left alone, for every action
far_pos, far_vel = np.array([[[3.0, 3.0]]]), np.zeros((1, 1, 2))
for a in range(N_ACT):
    assert SH.apply(pos, yaw, far_pos, far_vel, np.array([a]))[0] == a

# a stopped robot parked beside the path but not on it is not a reason to change course
side_pos = np.array([[[0.6, 0.8]]])
assert SH.apply(pos, yaw, side_pos, far_vel, np.array([fast_straight]))[0] == fast_straight
print("shield tests passed")
