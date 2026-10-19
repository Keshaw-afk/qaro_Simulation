"""Generate the arena world SDF from a Layout. Usage: gen_world.py SEED OUT.sdf [--rt]"""
import sys

import layout as L
import tags
from robot_sdf import robot_sdf

FOREST_GREEN = "0.13 0.37 0.13 1"
PINK = "1.0 0.41 0.71 1"
WALL_GREY = "0.55 0.55 0.55 1"
ROBOT_COLORS = ["red", "orange", "blue", "yellow"]
WALL_H, WALL_T, OBSTACLE_H = 0.4, 0.05, 0.4  # both taller than the lidar scan plane (0.17 m)


def _static_box(name, x, y, z, sx, sy, sz, yaw, rgba):
    return f"""
  <model name="{name}"><static>true</static><pose>{x} {y} {z} 0 0 {yaw}</pose>
    <link name="l"><visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material></visual>
      <collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision></link></model>"""


GOAL_TAG_SIZE, GOAL_TAG_Z = 0.20, 0.002
# Camera height comes from layout.py. The big arena doubles the image size so pixels per metre (and tag decoding) stay the same.
CAM_Z, CAM_HFOV = L.CAM_Z, 1.45
CAM_W, CAM_H = (2560, 1920) if L.BIG else (1280, 960)


def _goal_marker(i, x, y):
    return f"""
  <model name="goal_{i + 1}"><static>true</static><pose>{x} {y} 0 0 0 0</pose>
    <link name="l">{tags.board_visual("v", GOAL_TAG_SIZE, 0, 0, GOAL_TAG_Z, 4 + i)}</link></model>"""


CAMERA = f"""
  <model name="overhead_cam"><static>true</static><pose>0 0 {CAM_Z} -1.5707963 1.5707963 0</pose>
    <link name="l"><sensor name="cam" type="camera"><topic>/overhead/image</topic><update_rate>10</update_rate><always_on>true</always_on>
      <camera><horizontal_fov>{CAM_HFOV}</horizontal_fov><image><width>{CAM_W}</width><height>{CAM_H}</height><format>R8G8B8</format></image>
        <clip><near>0.1</near><far>10</far></clip></camera></sensor></link></model>"""


def world_sdf(lay, real_time=False):
    w, h = lay.arena
    body = ""
    for i, (x, y, yaw) in enumerate(lay.starts):
        body += robot_sdf(f"r{i + 1}", ROBOT_COLORS[i], x, y, yaw, tag_id=i)
    for i, (x, y) in enumerate(lay.goals):
        body += _goal_marker(i, x, y)
    for i, ob in enumerate(lay.obstacles):
        x, y, sx, sy, yaw = ob
        body += _static_box(f"obstacle_{i}", x, y, OBSTACLE_H / 2, sx, sy, OBSTACLE_H, yaw, PINK)
    body += CAMERA
    for name, x, y, sx, sy in [("wall_n", 0, h / 2 + WALL_T / 2, w + 2 * WALL_T, WALL_T),
                               ("wall_s", 0, -h / 2 - WALL_T / 2, w + 2 * WALL_T, WALL_T),
                               ("wall_e", w / 2 + WALL_T / 2, 0, WALL_T, h),
                               ("wall_w", -w / 2 - WALL_T / 2, 0, WALL_T, h)]:
        body += _static_box(name, x, y, WALL_H / 2, sx, sy, WALL_H, 0, WALL_GREY)
    rtf = 1 if real_time else 0
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
<world name="arena">
  <physics name="p" type="dart"><max_step_size>0.005</max_step_size><real_time_factor>{rtf}</real_time_factor></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose>
    <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.3 0.2 -1</direction></light>
  <model name="ground"><static>true</static><link name="l">
    <collision name="c"><geometry><plane><normal>0 0 1</normal><size>{w} {h}</size></plane></geometry></collision>
    <visual name="v"><geometry><plane><normal>0 0 1</normal><size>{w} {h}</size></plane></geometry>
      <material><ambient>{FOREST_GREEN}</ambient><diffuse>{FOREST_GREEN}</diffuse></material></visual></link></model>{body}
</world>
</sdf>
"""


if __name__ == "__main__":
    seed, out = int(sys.argv[1]), sys.argv[2]
    open(out, "w").write(world_sdf(L.generate_layout(seed), real_time="--rt" in sys.argv))
