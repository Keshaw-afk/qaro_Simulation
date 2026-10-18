"""AprilTag (tag36h11) textures with a white quiet zone. ids 0-3: robots, ids 4-7: goal markers."""
import pathlib
import cv2
import numpy as np

TAG_DIR = pathlib.Path(__file__).resolve().parent.parent / "worlds" / "tags"
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
CELL = 32  # texture pixels per tag cell (8x8 cells incl. black border) -> 256 px tag, + 1 cell quiet zone each side


def tag_png(tag_id):
    TAG_DIR.mkdir(parents=True, exist_ok=True)
    path = TAG_DIR / f"tag36h11_{tag_id}.png"
    if not path.exists():
        m = cv2.aruco.drawMarker(DICT, tag_id, 8 * CELL)
        cv2.imwrite(str(path), cv2.copyMakeBorder(m, CELL, CELL, CELL, CELL, cv2.BORDER_CONSTANT, value=255))
    return path


TAG_FRACTION = 8 / 10  # black-bordered tag as a fraction of the board (quiet zone is 1 cell each side)


def board_visual(name, size, x, y, z, tag_id, yaw=0.0):
    """Flat square board textured with the tag, lit by emission so shading doesn't wash the pattern out."""
    uri = f"file://{tag_png(tag_id)}"
    return f"""<visual name="{name}"><pose>{x} {y} {z} 0 0 {yaw}</pose>
        <geometry><plane><normal>0 0 1</normal><size>{size} {size}</size></plane></geometry>
        <material><diffuse>1 1 1 1</diffuse><ambient>1 1 1 1</ambient>
          <pbr><metal><albedo_map>{uri}</albedo_map><emissive_map>{uri}</emissive_map></metal></pbr></material></visual>"""
