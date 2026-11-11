# Model comparison: cost, decision time, driving (no accuracy figures)

Sources: `benchmark/results/summary.json` (benchmark of 2026-11-10, safety layer on; driving figures over the 400 robots of
the held-out test layouts 100-199 in 2D), `docs/DESIGNED_LAYOUTS.md` (adversarial arenas 9001-9005). PDF version:
`model_comparison.pdf` (printed from `model_comparison.html`:
`google-chrome --headless=new --no-pdf-header-footer --print-to-pdf=model_comparison.pdf model_comparison.html`). For success/collision numbers see `benchmark/RESULTS.md`.

## Resources and driving

| Model | Training / tuning cost | Model size | Decision time per robot, 2D / Gazebo (median) | Avg speed | Path efficiency¹ | Time to goal² | Steering flips / min | Time stopped |
|---|---|---|---|---|---|---|---|---|
| PPO | 8M steps, about 2 h on the laptop CPU (original project) | 92,717 weights, 372 KB | 113 µs / 98 µs | 0.127 m/s | 0.97 | 10.9 s | 218 | 4% |
| DQN (seed 2) | 6M steps, 1.7 h on a Colab T4 GPU (about 960 steps/s); tuning 6 x 1M steps, 1.0 h | 92,717 weights, 372 KB | 83 µs / 84 µs | 0.090 m/s | 0.96 | 16.1 s | 249 | 10% |
| Rule controller | No training; 50 parameter sets tuned, 1.7 h (4.3 CPU-h) | 7 parameters | 160 µs / 1.25 ms | 0.085 m/s | 0.99 | 9.6 s | 12 | 1% |
| DWA (privileged³) | None | none | not timed (needs simulator state) | 0.066 m/s | 0.93 | 11.9 s | 4 | 0% |

¹ Straight-line distance / distance driven (1.0 = perfectly direct). ² Mean over robots that reached their goal.
³ Uses the true map of the arena, so it cannot be deployed on the real robots. Every model decides in under 2% of the 100 ms decision period.

## Verdict

| Model | Normal arenas | Adversarial arenas (9001-9005) |
|---|---|---|
| PPO | Among the best: fast and direct, but its steering wobbles constantly. | Fine at open crossings; breaks down at doors and narrow gaps, and crashes into robots without the safety layer. |
| DQN | On par with PPO but more cautious: slower, stops more, crashes less, wobbles as much. | Held up best in 2D; when it fails it gets stuck rather than crashes. |
| Rule controller | Slightly behind the learned policies; the smoothest, steadiest driver, but gets trapped behind clusters of obstacles. | Good in corridors and at the door in 2D, but deadlocks around a central pillar and stalls in clutter. |
| DWA (privileged) | Decent, very smooth but slow; not deployable (needs the true map). | Stalls often, even with the true map. |
| Go-to-goal | Baseline only: crashes whenever something is in the way. | Fails almost everywhere. |

Caveats: PPO's training cost is from the original project's documentation, not measured in the benchmark. DQN's cost is
for the one seed kept; three seeds were trained at the same cost each. The adversarial verdicts rest on one episode per
arena, mostly in 2D (Gazebo checks only for PPO), so they are impressions, not measurements.
