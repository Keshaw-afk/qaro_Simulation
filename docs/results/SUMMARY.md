# Gazebo sweep results (final_policy.npz, shield on)

| Arena | Layout seeds | Layouts run | Success | Collision | Timeout |
|---|---|---|---|---|---|
| Standard 4x3 m | 1-50 except 37 | 49 (196 robot runs) | 89.8% (176) | 6.1% (12) | 4.1% (8) |
| Big 8x6 m (`SNR_ARENA=big`) | 51-60 | 10 (40 robot runs) | 65.0% (26) | 2.5% (1) | 32.5% (13) |

Notes
- Standard: logs `gazebo_standard_arena_seeds_1-50.log` (stopped after seed 36 by a Gazebo clock stall) and `..._38-50.log` (rerun). Seed 37 was not run.
- Big: sweep stopped by hand after seed 60; pattern was clear (the policy was trained on the small arena only; failures are mostly timeouts at the 1800-step cap). No JSON summary exists for it, the log is the record.
