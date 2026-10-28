"""Per-run robot log for eval_gazebo.py (SNR_LOG=1): positions and collision signals, read-only (the policy is not affected).

Every policy step (0.1 s of simulated time) one row per robot goes to OUT_DIR/log_seed{k}.csv. Once per simulated second
a table is printed: ground-truth position, heading, speed, distance to goal, and the collision signals of that second:
  gap     smallest ground-truth clearance to an obstacle, wall or other robot during the second (metres; 0 = touching)
  near    number of steps with gap below NEAR_GAP
  shield  number of steps on which the safety layer replaced the policy's action (a predicted robot-robot collision course)
Events (goal reached, collision) are printed as soon as they happen."""
import csv
import math

NEAR_GAP = 0.05     # metres: a step with less clearance than this counts as a near miss
STEPS_PER_LINE = 10  # policy steps per printed table (10 x 0.1 s = 1 s)


class RunLog:
    def __init__(self, path, seed, goals):
        self.f = open(path, "w", newline="")
        self.csv = csv.writer(self.f)
        self.csv.writerow(["t_s", "robot", "x_m", "y_m", "yaw_deg", "speed_mps", "to_goal_m", "gap_m", "near_miss", "shield_veto", "status"])
        self.path, self.goals = path, goals
        self.prev_status = ["running"] * 4
        self._reset()
        print(f"logging layout {seed}: 1 table per simulated second, every 0.1 s step in {path}", flush=True)
        print(f"signals per second: gap = smallest clearance [m], near = steps with gap < {NEAR_GAP} m, shield = safety-layer interventions (of {STEPS_PER_LINE})", flush=True)

    def _reset(self):
        self.min_gap, self.near, self.vetoes = [math.inf] * 4, [0] * 4, [0] * 4

    def step(self, step, pose, speed, gaps, vetoed, status):
        """pose[i] = ground-truth (x, y, yaw); speed[i] m/s; gaps[i] m; vetoed[i] bool; status[i] str. Called once per policy step."""
        t = step * 0.1
        for i in range(4):
            x, y, yaw = pose[i]
            to_goal = math.hypot(x - self.goals[i][0], y - self.goals[i][1])
            near = gaps[i] < NEAR_GAP and status[i] == "running"
            self.csv.writerow([f"{t:.1f}", f"r{i + 1}", f"{x:.3f}", f"{y:.3f}", f"{math.degrees(yaw):.0f}", f"{speed[i]:.3f}",
                               f"{to_goal:.3f}", f"{gaps[i]:.3f}", int(near), int(vetoed[i]), status[i]])
            if status[i] == "running" or status[i] != self.prev_status[i]:
                self.min_gap[i] = min(self.min_gap[i], gaps[i])
            self.near[i] += near
            self.vetoes[i] += bool(vetoed[i])
            if status[i] != self.prev_status[i]:
                what = "REACHED ITS GOAL" if status[i] == "success" else "COLLISION" if status[i] == "collision" else status[i].upper()
                print(f"  >>> t={t:5.1f}s  r{i + 1} {what}  at ({x:+.2f}, {y:+.2f}), gap {gaps[i]:.3f} m", flush=True)
                self.prev_status[i] = status[i]
        if (step + 1) % STEPS_PER_LINE == 0:
            self._table(t, pose, speed, status)

    def _table(self, t, pose, speed, status):
        print(f"t = {t:5.1f} s   robot   x[m]    y[m]  heading  speed   goal  |  gap[m]  near  shield  status", flush=True)
        for i in range(4):
            x, y, yaw = pose[i]
            to_goal = math.hypot(x - self.goals[i][0], y - self.goals[i][1])
            gap = f"{self.min_gap[i]:6.3f}" if self.min_gap[i] < math.inf else "     -"
            flag = "  <-- close" if self.near[i] else ""
            print(f"                r{i + 1}   {x:+6.2f}  {y:+6.2f}  {math.degrees(yaw):5.0f}°  {speed[i]:5.2f}  {to_goal:5.2f}  |  {gap}  {self.near[i]:4d}  {self.vetoes[i]:6d}  {status[i]}{flag}", flush=True)
        self._reset()

    def close(self):
        self.f.close()
        print(f"log: {self.path}", flush=True)
