# Task requirements and how they are met

Original task (paraphrased): four robots reach their specified endpoints while avoiding obstacles and each other, using RL for obstacle avoidance;
one shared model decides every robot's movement from its own sensor feed.

| Requirement | Where / how |
|---|---|
| Plain terrain, rectangular arena | `tools/gen_world.py` (4 x 3 m; `SNR_ARENA=big` gives 8 x 6 m); forest-green floor |
| Four robots, differential drive, two wheels, wheel encoders | `tools/robot_sdf.py`; diff-drive plugin with odometry |
| Chassis colours red, orange, blue, yellow; wheels and lidar black | `tools/robot_sdf.py` (robots r1-r4 in that order) |
| Unique AprilTag per robot | tag36h11 ids 0-3 on top of each robot; goals are ids 4-7 |
| Lidar 0.15 m to 2 m | gpu_lidar, 360 beams at 10 Hz, min range 0.15, max 2.0 (`config/robot.yaml`) |
| Random obstacles: random number, size, position, no softlock | `tools/layout.py`: 3-8 obstacles (12-32 in the big arena), sides 0.15-0.60 m, all starts/goals in one connected free region |
| Pink obstacles | `tools/gen_world.py` |
| Overhead camera giving coordinates of endpoints and robots in real time | camera + AprilTag detection `tools/vision.py`; poses, goals and velocities feed the policy |
| RL for obstacle avoidance, shared model (one model, each robot's sensor feed -> its movement command) | `rl/train.py` PPO; `rl/deploy.py`; one network used by all four robots, own-id one-hot input |
| Robot-robot avoidance | policy trained with camera-derived positions/velocities of the other robots, plus the rule-based safety layer `rl/shield.py` |

Robot dimensions that were not given were assumed from photos and the published TortoiseBot model (`config/robot.yaml` marks every value SOURCED, DERIVED or ESTIMATED).
