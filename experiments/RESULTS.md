# Hardware experiments: no encoders, one robot (branch `hardware-experimentation`, 2026-11-01)

Target hardware: one TortoiseBot-style robot (Raspberry Pi 4, BO gear motors **without encoders**, no IMU/gyro,
YDLIDAR X2 lidar) under the overhead AprilTag camera. Question: can the existing policy `weights/final_policy.npz`
run on it **without retraining**? All results below use those weights unchanged. No existing file was modified.

## What the encoders feed
Of the 59 policy inputs, exactly two come from wheel odometry: own forward speed v and turn rate w (observation
columns 39-40, `rl/deploy.py` `obs_from_sensors`, `od[3]`/`od[4]` in `rl/eval_gazebo.py`). Own position, heading,
goal and the other robots come from the overhead camera; 36 inputs from the lidar. The safety layer (`rl/shield.py`)
uses camera data only.

Substitutes tested for (v, w):
- `cmdmodel`: first-order lag of the commanded (v, w), est += (cmd - est) * 0.75. No sensor.
- `camera`: differencing consecutive camera poses: v = displacement along heading / 0.1 s, w = wrapped heading change / 0.1 s.
  In 2D with Gaussian pose noise 1 cm / 0.02 rad; in Gazebo the real simulated camera + AprilTag pipeline.
- `zeros`: input removed.

Motor models (without encoders there is no closed-loop wheel-speed control, so actual speed can differ from the command):
`nominal` = exact; `train` = training randomisation (speed x U(0.9,1.1), lag U(0.5,1.0), 30% one-step delay);
`harsh` = speed x U(0.7,1.3), lag U(0.3,1.0), 50% delay (wider than training).

## Single robot: how the three absent robots are filled in
The policy has fixed input slots for three other robots and was never trained with them absent. Filling matters
(2D, nominal motors, real v/w, layouts 0-49 x 4 start/goal pairs = 200 runs each):

| Absent robots | Success | Collision | Timeout |
|---|---|---|---|
| parked 50 m away | 43.5% | 22.0% | 34.5% |
| left inactive at their start poses (physically present) | 82.5% | 3.0% | 14.5% |
| **virtual robots fixed 0.3 m outside the walls** (used below) | **94.5%** | 2.5% | 3.0% |

Parking far away saturates all six relative-position inputs at +-1 simultaneously, which the 4 x 3 m training arena
practically never produced (each value is ~2.6 sd from its training mean; the joint pattern is unseen). Exact internal
mechanism not traced. The virtual positions are constants on the real robot: (-2.3, 0), (2.3, 0), (0, 1.8) m.
Note: the fill was chosen on evaluation seeds 0-49 (AGENTS.md rule 5); the full grid below uses seeds 0-99.

## 2D results (`experiments/no_encoder_2d.py`, layouts 0-99, success rate)
Four robots (400 robot runs per cell):

| Motors | encoder | cmdmodel | camera | zeros |
|---|---|---|---|---|
| nominal | 91.5% | 90.7% | 89.7% | 90.0% |
| train | 90.7% | 90.5% | 90.5% | 90.0% |
| harsh | 93.0% | 91.2% | 91.5% | 92.0% |

Single robot, virtual robots outside the walls (each layout run once per start/goal pair, 400 runs per cell):

| Motors | encoder | cmdmodel | camera | zeros |
|---|---|---|---|---|
| nominal | 93.2% | 94.8% | 96.3% | 94.5% |
| train | 94.8% | 95.0% | 95.3% | 94.8% |
| harsh | 96.0% | 96.8% | 96.8% | 94.8% |

Differences of 1-2 points are within run-to-run noise (harsh+encoder beating nominal+encoder shows this).
Conclusion in 2D: the policy barely uses the v/w inputs (it also sees its own previous command), and works with one robot.

## Gazebo, single robot (`experiments/eval_gazebo_single.py`)
One robot, full Gazebo pipeline (simulated overhead camera + AprilTags for pose and goal, 360-beam lidar), own v/w
**from the camera, encoders not used**, virtual robots outside the walls, safety layer on. Layouts 0-19, each start/goal
pair run separately (80 single-robot runs):

| Success | Collision | Timeout | Mean time to goal |
|---|---|---|---|
| **97.5% (78/80)** | 2.5% (2/80) | 0% | 10.6 s |

The two collisions (layout 16 robot 3 after 16.8 s, layout 17 robot 0 after 6.8 s) are with an obstacle or wall (no
other robot present); not diagnosed. For reference, four robots with encoders on held-out layouts: 91.2% (73/80).
The batch was run in two parts (layouts 0-9, then 10-19 after a simulator startup failure "world control never
advanced the clock"; the script now retries once). One run on layout 14 took ~55 min wall time instead of ~1 min
(it finished, success); cause not diagnosed.

Live view with the policy dashboard: `bash experiments/live_dashboard.sh 29 0 camera` (layout 29, success in 16.9 s and 17.2 s).

## Not yet shown
- Real motors: how far actual wheel speed drifts from the command (battery, left/right mismatch). Measure with the
  overhead camera; also gives the PWM-to-speed table the Pi needs for open-loop control.
- Real lidar: YDLIDAR X2 scans need resampling to 360 beams over [-pi, pi] (0 = forward, counter-clockwise) and
  its scan rate vs the 10 Hz policy loop must be handled.
- Real camera noise and latency (simulated camera used here).

## Reproduce
```
.venv/bin/python experiments/no_encoder_2d.py 100 four,single          # 2D grid (~1.5 min per four-robot cell, ~5 min per single cell)
source /opt/ros/jazzy/setup.bash
SNR_VW=camera python3 experiments/eval_gazebo_single.py weights/final_policy.npz 20      # Gazebo, layouts 0-19 x 4
SNR_LIVE=1 SNR_FIRST=45 SNR_ROBOT=0 python3 experiments/eval_gazebo_single.py weights/final_policy.npz 46   # watch one
```
