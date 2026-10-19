"""SDF for one TortoiseBot-style differential-drive robot. All dimensions come from config/robot.yaml."""
import pathlib

import yaml

import tags

CFG = yaml.safe_load((pathlib.Path(__file__).parent.parent / "config" / "robot.yaml").read_text())

COLORS = {  # chassis colours from the spec; wheels and lidar are always black
    "red": (0.85, 0.05, 0.05), "orange": (1.0, 0.5, 0.0),
    "blue": (0.05, 0.2, 0.9), "yellow": (0.95, 0.85, 0.05),
}


def _rgba(c):
    return f"{c[0]} {c[1]} {c[2]} 1"


def _cyl_inertia(m, r, h):  # about the cylinder axis / transverse
    return 0.5 * m * r * r, m * (3 * r * r + h * h) / 12


TAG_BOARD, TAG_Z = 0.12, 0.195  # tag board sits above the lidar scan plane (0.17 m), so it never blocks the scan


def robot_sdf(name, color, x, y, yaw, tag_id=None):
    c = CFG
    ch = c["chassis_z_top"] - c["chassis_z_bottom"]
    chz = c["chassis_z_bottom"] + ch / 2
    izz, ixx = _cyl_inertia(c["base_mass"], c["chassis_radius"], ch)
    wr, ww, wm = c["wheel_radius"], c["wheel_width"], c["wheel_mass"]
    wizz, wixx = _cyl_inertia(wm, wr, ww)
    hs = c["wheel_separation"] / 2
    col = _rgba(COLORS[color])
    black = "0.02 0.02 0.02 1"
    lz = c["lidar_z"]

    def wheel(side, ysign):
        return f"""
    <link name="{side}_wheel">
      <pose relative_to="base_link">0 {ysign * hs} {wr} -1.5707963 0 0</pose>
      <inertial><mass>{wm}</mass><inertia><ixx>{wixx}</ixx><iyy>{wixx}</iyy><izz>{wizz}</izz></inertia></inertial>
      <visual name="v"><geometry><cylinder><radius>{wr}</radius><length>{ww}</length></cylinder></geometry>
        <material><ambient>{black}</ambient><diffuse>{black}</diffuse></material></visual>
      <collision name="c"><geometry><cylinder><radius>{wr}</radius><length>{ww}</length></cylinder></geometry>
        <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
    </link>
    <joint name="{side}_wheel_joint" type="revolute">
      <parent>base_link</parent><child>{side}_wheel</child>
      <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
    </joint>"""

    return f"""
  <model name="{name}">
    <pose>{x} {y} 0 0 0 {yaw}</pose>
    <link name="base_link">
      <inertial><pose>{c['com_x']} 0 {chz} 0 0 0</pose><mass>{c['base_mass']}</mass>
        <inertia><ixx>{ixx}</ixx><iyy>{ixx}</iyy><izz>{izz}</izz></inertia></inertial>
      <visual name="chassis"><pose>0 0 {chz} 0 0 0</pose>
        <geometry><cylinder><radius>{c['chassis_radius']}</radius><length>{ch}</length></cylinder></geometry>
        <material><ambient>{col}</ambient><diffuse>{col}</diffuse></material></visual>
      <collision name="chassis_col"><pose>0 0 {chz} 0 0 0</pose>
        <geometry><cylinder><radius>{c['chassis_radius']}</radius><length>{ch}</length></cylinder></geometry></collision>
      <visual name="lidar_body"><pose>{c['lidar_x']} 0 {lz} 0 0 0</pose>
        <geometry><cylinder><radius>{c['lidar_radius']}</radius><length>{c['lidar_height']}</length></cylinder></geometry>
        <material><ambient>{black}</ambient><diffuse>{black}</diffuse></material></visual>
      {tags.board_visual('tag', TAG_BOARD, 0, 0, TAG_Z, tag_id) if tag_id is not None else ''}
      <collision name="caster_col"><pose>{c['caster_x']} 0 {wr} 0 0 0</pose>
        <geometry><sphere><radius>{wr}</radius></sphere></geometry>
        <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface></collision>
      <sensor name="lidar" type="gpu_lidar">
        <pose>{c['lidar_x']} 0 {lz} 0 0 0</pose>
        <topic>/{name}/scan</topic>
        <update_rate>{c['lidar_rate_hz']}</update_rate>
        <always_on>true</always_on>
        <lidar>
          <scan><horizontal><samples>{c['lidar_samples']}</samples><resolution>1</resolution>
            <min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal></scan>
          <range><min>{c['lidar_min_range']}</min><max>{c['lidar_max_range']}</max><resolution>0.01</resolution></range>
        </lidar>
      </sensor>
    </link>{wheel('left', 1)}{wheel('right', -1)}
    <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>
      <wheel_separation>{c['wheel_separation']}</wheel_separation>
      <wheel_radius>{wr}</wheel_radius>
      <max_linear_velocity>{c['max_linear_speed']}</max_linear_velocity>
      <min_linear_velocity>{-c['max_linear_speed']}</min_linear_velocity>
      <max_angular_velocity>{c['max_angular_speed']}</max_angular_velocity>
      <min_angular_velocity>{-c['max_angular_speed']}</min_angular_velocity>
      <topic>/{name}/cmd_vel</topic><odom_topic>/{name}/odom</odom_topic>
      <tf_topic>/{name}/tf</tf_topic>
      <frame_id>{name}/odom</frame_id><child_frame_id>{name}/base_link</child_frame_id>
      <odom_publisher_frequency>50</odom_publisher_frequency>
    </plugin>
    <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">
      <topic>/{name}/joint_states</topic>
      <joint_name>left_wheel_joint</joint_name><joint_name>right_wheel_joint</joint_name>
    </plugin>
  </model>"""
