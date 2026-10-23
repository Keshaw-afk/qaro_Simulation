"""Demo video from the angled demo camera (gen_world.demo_camera), used by eval_gazebo.py when SNR_DEMO=se|sw.

Frames are placed by their simulation timestamps, so the video plays in real time even though the evaluation steps the
world while paused; a frame the renderer skipped is filled by repeating the previous one. Writes MJPG .avi (OpenCV has
no H.264 encoder here); convert for sharing with  ffmpeg -i demo.avi -c:v libx264 -pix_fmt yuv420p -crf 18 demo.mp4"""
import threading

import cv2
import numpy as np
from gz.msgs10.image_pb2 import Image

import gen_world

BGR = {"red": (40, 40, 220), "orange": (0, 140, 255), "blue": (230, 80, 20), "yellow": (20, 215, 240)}


class DemoRecorder:
    def __init__(self, sim, path, seed, status, t_done):
        self.fps = gen_world.DEMO_FPS
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (gen_world.DEMO_W, gen_world.DEMO_H))
        self.seed, self.status, self.t_done = seed, status, t_done  # the evaluator's live dicts, read for the overlay
        self.lock, self.t0, self.n, self.rendered, self.closed = threading.Lock(), None, 0, 0, False
        self.t_start = None  # sim time when the robots start driving (set by the evaluator); the overlay clock counts from it
        assert sim.node.subscribe(Image, gen_world.DEMO_TOPIC, self._on_image)

    def _on_image(self, m):
        t = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
        f = cv2.cvtColor(np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3), cv2.COLOR_RGB2BGR)
        self._overlay(f, t)
        with self.lock:
            if self.closed:
                return
            if self.t0 is None:
                self.t0 = t
            idx = round((t - self.t0) * self.fps)
            self.rendered += 1
            while self.n <= idx:  # repeat the frame over any gap so time stays true
                self.writer.write(f)
                self.n += 1

    def _overlay(self, f, t):
        top = f.shape[0] - 186  # bottom-left corner: open floor in the demo view, never the arena
        panel = f.copy()
        cv2.rectangle(panel, (12, top), (430, top + 174), (25, 25, 25), -1)
        cv2.addWeighted(panel, 0.6, f, 0.4, 0, f)
        put = lambda s, x, y, c=(255, 255, 255), sc=0.6: cv2.putText(f, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, 1, cv2.LINE_AA)
        clock = 0.0 if self.t_start is None else max(0.0, t - self.t_start)
        put(f"4 robots, one shared RL policy   t = {clock:4.1f} s", 26, top + 28, sc=0.62)
        for i, colour in enumerate(gen_world.ROBOT_COLORS):
            s = self.status[i]
            label = {"running": "driving", "success": f"reached goal in {self.t_done[i] or 0:.1f} s",
                     "collision": "collision", "timeout": "timeout"}[s]
            y = top + 64 + 30 * i
            cv2.circle(f, (36, y - 6), 9, BGR[colour], -1)
            put(f"R{i + 1}  {colour}", 54, y)
            put(label, 175, y, (140, 240, 140) if s == "success" else (255, 255, 255))

    def close(self):
        with self.lock:
            self.closed = True
            self.writer.release()
        return self.n, self.rendered
