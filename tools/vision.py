"""Overhead-camera vision: AprilTag detection -> world-frame coordinates of robots and goals.

Camera looks straight down from CAM_Z, rolled so image right = world +x and image up = world +y. Tags are flat on a
known plane (robot top / floor), so a pinhole model with the plane's depth gives metric coordinates."""
import math

import cv2
import numpy as np

import tags
from gen_world import CAM_H, CAM_HFOV, CAM_W, CAM_Z, GOAL_TAG_Z
from robot_sdf import TAG_Z

F_PX = (CAM_W / 2) / math.tan(CAM_HFOV / 2)
YAW_OFFSET = -math.pi / 2  # tag up vs robot forward (measured raw error +89.2 deg, std 1.0 deg)
_PARAMS = cv2.aruco.DetectorParameters_create()
_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX


def _to_world(u, v, plane_z):
    depth = CAM_Z - plane_z
    return (u - CAM_W / 2) * depth / F_PX, -(v - CAM_H / 2) * depth / F_PX


SCALES = (1.0, 0.7, 0.5)  # OpenCV's tag36h11 decoder misses some ids at some pixel scales, so detect at several and merge


def _find_tags(gray):
    """{id: 4x2 corner array in full-resolution pixels}; the first (finest) scale that decodes an id wins."""
    found = {}
    for sc in SCALES:
        im = gray if sc == 1.0 else cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        corners, ids, _ = cv2.aruco.detectMarkers(im, tags.DICT, parameters=_PARAMS)
        for c, tid in zip(corners, ids.flatten() if ids is not None else []):
            found.setdefault(int(tid), c.reshape(4, 2) / sc)
    return found


def detect(img_rgb):
    """-> {'robots': {i: (x, y, yaw)}, 'goals': {i: (x, y)}} for i in 0..3 (missing if not detected)."""
    out = {"robots": {}, "goals": {}}
    for tid, c in _find_tags(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)).items():
        u, v = c.mean(axis=0)
        if tid < 4:
            x, y = _to_world(u, v, TAG_Z)
            top = (c[0] + c[1]) / 2 - c.mean(axis=0)  # image-space vector from tag centre toward its top edge
            yaw = math.atan2(-top[1], top[0]) + YAW_OFFSET  # image right -> world +x, image up -> world +y
            out["robots"][tid] = (x, y, (yaw + math.pi) % (2 * math.pi) - math.pi)
        elif tid < 8:
            out["goals"][tid - 4] = _to_world(u, v, GOAL_TAG_Z)
    return out
