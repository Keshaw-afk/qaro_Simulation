# Instructor artifacts

One folder per delivery, named `YYYY-MM-DD_topic`, newest last. Run scripts from anywhere; they find the repository
themselves.

| Folder | What is in it |
|---|---|
| `2026-10-28_demo_kit/` | Live demo scripts and their notes: `instructor_demo LAYOUT` (play a layout live in Gazebo), `instructor_demo_advanced log LAYOUT` (same, with per-second positions and collision signals), `layout_info.txt` (recommended layouts), `ros_info_dump` + `ros_info.txt` (ROS 2 / Gazebo topics and nodes), `training_curve_and_setup.png` (PPO training curve and setup). |
| `2026-11-11_model_comparison/` | One-page comparison of PPO, DQN, the rule controller and DWA: training cost, decision time, driving style, and a verdict for normal and adversarial arenas (`model_comparison.pdf`; `.md` and `.html` are its editable sources). No accuracy figures; those are in `benchmark/RESULTS.md`. |
| `2026-11-11_controller_videos/` | Three videos, PPO, DQN and the rule controller, each on the same four arenas, with per-robot results in that folder's `README.md`. |

Full benchmark (numbers, method, PDF report): `benchmark/RESULTS.md`, `benchmark/REPORT.pdf`.
