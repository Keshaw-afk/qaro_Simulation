"""Live policy dashboard: shows what the model sees and decides, refreshed once per second (ROS 2 + Tk).

    source /opt/ros/jazzy/setup.bash; python3 experiments/dashboard.py [PERIOD_S=1.0]

Subscribes to /snr/policy_debug (std_msgs/String, one JSON object per 0.1 s policy step), published by
experiments/eval_gazebo_single.py with SNR_DASH=1 (a real-robot node can publish the same message later).
Each refresh shows the most recent step:
  - lidar: 36 sector distances around the robot (polar plot, forward = up) with the goal direction
  - the inputs: goal distance/bearing, own v/w fed to the model (and odometry for reference), virtual robots, previous action
  - the decision: top-3 actions with probabilities, action sent, whether the safety layer overrode the policy
  - all 59 observation values as a strip (blue -1 .. white 0 .. red +1)
Read-only: it never sends commands."""
import json, math, sys, threading, tkinter as tk

import rclpy
from std_msgs.msg import String

PERIOD_MS = int(float(sys.argv[1] if len(sys.argv) > 1 else 1.0) * 1000)
GROUPS = [("lidar", 0, 36), ("goal", 36, 39), ("own v,w", 39, 41), ("other robots", 41, 53), ("prev act", 53, 55), ("own ID", 55, 59)]
BG, FG, DIM, ACC, WARN, OK = "#1e1f24", "#e6e6e6", "#8a8f98", "#4aa3ff", "#ff6b5b", "#5bd18b"


class Dash:
    def __init__(self, root):
        self.latest, self.n_msgs, self.lock = None, 0, threading.Lock()
        root.title("snr policy dashboard")
        root.configure(bg=BG)
        self.hdr = tk.Label(root, text="waiting for /snr/policy_debug ...", font=("monospace", 12, "bold"), bg=BG, fg=FG, anchor="w")
        self.hdr.grid(row=0, column=0, columnspan=2, sticky="we", padx=10, pady=(8, 4))
        self.cv = tk.Canvas(root, width=320, height=320, bg=BG, highlightthickness=0)
        self.cv.grid(row=1, column=0, padx=10)
        self.txt = tk.Label(root, text="", font=("monospace", 10), bg=BG, fg=FG, justify="left", anchor="nw")
        self.txt.grid(row=1, column=1, sticky="nw", padx=10)
        self.strip = tk.Canvas(root, width=760, height=56, bg=BG, highlightthickness=0)
        self.strip.grid(row=2, column=0, columnspan=2, padx=10, pady=(4, 10))
        self.root = root
        root.after(PERIOD_MS, self.refresh)

    def on_msg(self, m):
        with self.lock:
            self.latest, self.n_msgs = json.loads(m.data), self.n_msgs + 1

    def refresh(self):
        with self.lock:
            d, n = self.latest, self.n_msgs
        if d is not None:
            self.draw(d, n)
        self.root.after(PERIOD_MS, self.refresh)

    def draw(self, d, n):
        st = d.get("status", "?")
        col = OK if st == "success" else WARN if st in ("collision", "timeout") else FG
        self.hdr.config(text=f"layout {d['seed']}  robot {d['robot']}  t={d['t']:.1f}s  step {d['step']}  status: {st}"
                             + ("  (run finished)" if d.get("final") else "") + f"   msgs {n}", fg=col)
        if d.get("final"):
            return
        self.draw_lidar(d)
        x, y, yaw = d["pose"]
        vf, wf = d["vw_fed"]; vo, wo = d["vw_odom"]
        lines = [
            "INPUTS (what the model sees)",
            f"  pose (camera)    x {x:+.2f}  y {y:+.2f}  yaw {math.degrees(yaw):+.0f} deg",
            f"  goal             {d['goal_dist']:.2f} m at {d['goal_bearing_deg']:+.0f} deg",
            f"  own v, w fed     {vf:+.3f} m/s  {wf:+.2f} rad/s   (source: {d['vw_source']})",
            f"  odometry (ref)   {vo:+.3f} m/s  {wo:+.2f} rad/s   (not used unless source=odom)",
            f"  nearest lidar    {min(d['lidar36_m']):.2f} m",
            f"  other robots     3 virtual, fixed outside the walls",
            f"  previous action  v {d['obs'][53]:+.2f}  w {d['obs'][54]:+.2f}  (normalised)",
            "",
            "DECISION (45 possible actions)",
        ]
        for i, (v, w, p) in enumerate(d["top3"]):
            lines.append(f"  #{i + 1}  v {v:.2f} m/s  w {w:+.1f} rad/s   p = {p:.2f}  " + "#" * int(p * 20))
        sv, sw = d["action_sent"]
        lines += ["", f"  SENT  v {sv:.2f} m/s  w {sw:+.1f} rad/s" + ("   <- SAFETY LAYER OVERRIDE" if d["shield_override"] else "")]
        self.txt.config(text="\n".join(lines))
        self.draw_strip(d["obs"])

    def draw_lidar(self, d):
        c, cx, cy, R = self.cv, 160, 165, 140
        c.delete("all")
        for r in (0.5, 1.0, 1.5, 2.0):
            k = R * r / 2.0
            c.create_oval(cx - k, cy - k, cx + k, cy + k, outline="#33363d")
            c.create_text(cx + k - 12, cy - 6, text=f"{r:g}m", fill=DIM, font=("monospace", 7))
        n = len(d["lidar36_m"])
        for j, r in enumerate(d["lidar36_m"]):  # sector j covers beams 10j..10j+9 of a scan from -pi (behind) counter-clockwise
            a0 = -math.pi + j * 2 * math.pi / n
            pts = [(cx, cy)]
            for a in (a0, a0 + 2 * math.pi / n):
                pts.append((cx - R * r / 2.0 * math.sin(a), cy - R * r / 2.0 * math.cos(a)))  # robot forward = up, left = left
            near = r < 0.4
            c.create_polygon(*sum(pts, ()), fill="#5a2a2a" if near else "#24405f", outline=WARN if near else ACC)
        b = math.radians(d["goal_bearing_deg"])
        gx, gy = cx - R * 0.95 * math.sin(b), cy - R * 0.95 * math.cos(b)
        c.create_line(cx, cy, gx, gy, fill=OK, width=2, arrow="last")
        c.create_text(gx, gy - 8, text="goal", fill=OK, font=("monospace", 8))
        c.create_polygon(cx, cy - 10, cx - 7, cy + 7, cx + 7, cy + 7, fill=FG)
        c.create_text(cx, 10, text="lidar (36 sectors), forward = up", fill=DIM, font=("monospace", 8))

    def draw_strip(self, obs):
        s = self.strip
        s.delete("all")
        wpx = 760 / len(obs)
        for i, v in enumerate(obs):
            v = max(-1.0, min(1.0, v))
            r, g, b = (255, int(255 * (1 - v)), int(255 * (1 - v))) if v >= 0 else (int(255 * (1 + v)), int(255 * (1 + v)), 255)
            s.create_rectangle(i * wpx, 4, (i + 1) * wpx, 30, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")
        for name, a, b_ in GROUPS:
            s.create_line(a * wpx, 2, a * wpx, 34, fill=BG, width=2)
            s.create_text((a + b_) / 2 * wpx, 44, text=name, fill=DIM, font=("monospace", 8))


def main():
    rclpy.init()
    node = rclpy.create_node("snr_dashboard")
    root = tk.Tk()
    import signal
    for sig in (signal.SIGTERM, signal.SIGINT):  # handled within one refresh period (Tk's loop runs Python then)
        signal.signal(sig, lambda *_: root.destroy())
    dash = Dash(root)
    node.create_subscription(String, "/snr/policy_debug", dash.on_msg, 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
