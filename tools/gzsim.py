"""Lockstep Python interface to a running Gazebo world (paused + multi_step), via gz-transport bindings.

    sim = GzSim(["r1", "r2", "r3", "r4"])
    sim.cmd_vel("r1", 0.1, 0.0); sim.step(100)      # 100 physics steps = 0.5 s
    sim.scan("r1"), sim.odom("r1"), sim.pose("r1")
"""
import math
import threading
import time

import numpy as np

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.image_pb2 import Image
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.odometry_pb2 import Odometry
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node

STEP_DT = 0.005  # must match <max_step_size> in the world


def _yaw_pitch_roll(q):
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    return yaw, pitch, roll


class GzSim:
    def __init__(self, robots, world="arena", camera=False):
        self.node, self.world, self.robots = Node(), world, list(robots)
        self._lock = threading.Lock()
        self._clock, self._poses, self._scans, self._odoms = 0.0, {}, {}, {}
        self._scan_n, self._image_n, self._camera = {r: 0 for r in self.robots}, 0, camera
        assert self.node.subscribe(Clock, "/clock", self._on_clock)
        assert self.node.subscribe(Pose_V, f"/world/{world}/pose/info", self._on_pose)
        for r in self.robots:
            assert self.node.subscribe(LaserScan, f"/{r}/scan", lambda m, r=r: self._on_scan(r, m))
            assert self.node.subscribe(Odometry, f"/{r}/odom", lambda m, r=r: self._on_odom(r, m))
        self._image = None
        if camera:
            assert self.node.subscribe(Image, "/overhead/image", self._on_image)
        self._pubs = {r: self.node.advertise(f"/{r}/cmd_vel", Twist) for r in self.robots}
        self.wait_for(lambda: self._poses and all(r in self._scans for r in self.robots), 30, "sim topics")

    # --- callbacks -------------------------------------------------------------------------
    def _on_clock(self, m):
        with self._lock:
            self._clock = m.sim.sec + m.sim.nsec * 1e-9

    def _on_pose(self, m):
        with self._lock:
            seen = set()
            for p in m.pose:  # model, link and sensor poses share one stream; keep the first entry per name
                if p.name in seen:
                    continue
                seen.add(p.name)
                self._poses[p.name] = (p.position.x, p.position.y) + _yaw_pitch_roll(p.orientation)

    def _on_scan(self, r, m):
        with self._lock:
            self._scans[r] = list(m.ranges)
            self._scan_n[r] += 1

    def _on_image(self, m):
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3).copy()
        with self._lock:
            self._image = img
            self._image_n += 1

    def image(self):
        """Latest overhead RGB frame (H, W, 3) or None."""
        with self._lock:
            return self._image

    def _on_odom(self, r, m):
        with self._lock:
            yaw, _, _ = _yaw_pitch_roll(m.pose.orientation)
            self._odoms[r] = (m.pose.position.x, m.pose.position.y, yaw, m.twist.linear.x, m.twist.angular.z)

    # --- control -------------------------------------------------------------------------
    def wait_for(self, cond, timeout, what=""):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if cond():
                    return
            time.sleep(0.002)
        raise TimeoutError(f"timed out waiting for {what}")

    def sim_time(self):
        with self._lock:
            return self._clock

    def step(self, n=1, sync_sensors=True):
        """Advance exactly n physics steps; return once the clock has caught up.

        gz-transport sometimes loses the *reply* to a world-control request even though the request ran, so
        success is judged by the sim clock, never the reply. A request is only re-sent if the clock did not
        move at all (a genuinely lost request), which keeps stepping exact."""
        target = self.sim_time() + n * STEP_DT - 1e-6
        with self._lock:
            scans0, img0 = dict(self._scan_n), self._image_n
        for _ in range(40):
            req = WorldControl()
            req.pause = True  # an unset pause flag un-pauses the world, so always send it with the step
            req.multi_step = n
            self.node.request(f"/world/{self.world}/control", req, WorldControl, Boolean, 200)
            try:
                self.wait_for(lambda: self._clock >= target, 1.0, "clock")
                break
            except TimeoutError:
                continue  # the clock never moved, so the request itself was lost
        else:
            raise RuntimeError("world control never advanced the clock")
        if sync_sensors and n >= 20:  # sensors tick at 10 Hz (every 20 steps): wait until this step's readings arrived
            try:
                self.wait_for(lambda: all(self._scan_n[r] > scans0[r] for r in self.robots) and (not self._camera or self._image_n > img0), 3.0, "sensor update")
            except TimeoutError:
                pass

    def set_paused(self, paused):
        req = WorldControl()
        req.pause = paused
        self.node.request(f"/world/{self.world}/control", req, WorldControl, Boolean, 5000)

    def cmd_vel(self, robot, lin, ang=0.0):
        m = Twist()
        m.linear.x, m.angular.z = float(lin), float(ang)
        self._pubs[robot].publish(m)

    # --- sensing -------------------------------------------------------------------------
    def pose(self, name):
        with self._lock:
            return self._poses[name]

    def scan(self, robot):
        with self._lock:
            return list(self._scans[robot])

    def odom(self, robot):
        with self._lock:
            return self._odoms[robot]
