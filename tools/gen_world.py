"""Generate the arena world SDF. Step 1: one robot, one test box. Usage: gen_world.py OUT.sdf"""
import sys

from robot_sdf import robot_sdf

ARENA_X, ARENA_Y = 4.0, 3.0
FOREST_GREEN = "0.13 0.37 0.13 1"
PINK = "1.0 0.41 0.71 1"


def world_sdf(robots, obstacles=()):
    body = "".join(robot_sdf(*r) for r in robots)
    for i, (x, y, sx, sy) in enumerate(obstacles):
        body += f"""
  <model name="obstacle_{i}"><static>true</static><pose>{x} {y} 0.2 0 0 0</pose>
    <link name="l"><visual name="v"><geometry><box><size>{sx} {sy} 0.4</size></box></geometry>
      <material><ambient>{PINK}</ambient><diffuse>{PINK}</diffuse></material></visual>
      <collision name="c"><geometry><box><size>{sx} {sy} 0.4</size></box></geometry></collision></link></model>"""
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
<world name="arena">
  <physics name="p" type="dart"><max_step_size>0.005</max_step_size><real_time_factor>0</real_time_factor></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose>
    <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.3 0.2 -1</direction></light>
  <model name="ground"><static>true</static><link name="l">
    <collision name="c"><geometry><plane><normal>0 0 1</normal><size>{ARENA_X} {ARENA_Y}</size></plane></geometry></collision>
    <visual name="v"><geometry><plane><normal>0 0 1</normal><size>{ARENA_X} {ARENA_Y}</size></plane></geometry>
      <material><ambient>{FOREST_GREEN}</ambient><diffuse>{FOREST_GREEN}</diffuse></material></visual></link></model>{body}
</world>
</sdf>
"""


if __name__ == "__main__":
    robots = [("r1", "red", -1.0, 0.0, 0.0)]
    boxes = [(0.0, 0.0, 0.3, 0.3)]
    open(sys.argv[1], "w").write(world_sdf(robots, boxes))
