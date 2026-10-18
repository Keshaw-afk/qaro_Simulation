# swarm_nav_rl

Four TortoiseBot-style differential-drive robots navigate a Gazebo arena to goals, avoiding obstacles and
each other with one shared RL policy (spec: `../my_gifts/spec.txt`).

- `config/robot.yaml`: robot parameters, each marked SOURCED / DERIVED / ESTIMATED
- `tools/robot_sdf.py`, `tools/gen_world.py`: generate robot and world SDF
- `tools/check_robot.py`: acceptance test against a running sim (upright, lidar clipping, drive direction, odom vs ground truth)

Run headless: `gz sim -s -r --headless-rendering worlds/step1.sdf`

Milestones are git tags (`git tag -n`).
