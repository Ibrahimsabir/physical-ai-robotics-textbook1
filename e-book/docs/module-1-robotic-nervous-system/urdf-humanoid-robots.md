---
title: URDF for Humanoid Robots
sidebar_label: URDF for Humanoids
description: Creating complete and accurate URDF descriptions for humanoid robots
keywords: [urdf, robot, description, humanoid, ros2, xml, modeling]
---

# URDF for Humanoid Robots

## Introduction

The Unified Robot Description Format (URDF) is an XML-based format used in ROS to describe robot models. For humanoid robots, URDF provides a comprehensive way to define the robot's physical structure, kinematic properties, and visual appearance. This chapter covers creating complete and accurate URDF descriptions specifically for humanoid robots.

## Learning Objectives

By the end of this chapter, you will be able to:
- Create complete URDF models for humanoid robots
- Define links, joints, and their properties accurately
- Include visual and collision geometries
- Implement proper kinematic chains for humanoid structures
- Validate URDF models and troubleshoot common issues

## What is URDF?

URDF (Unified Robot Description Format) is an XML format that describes a robot's physical structure. It defines:
- **Links**: Rigid parts of the robot (e.g., torso, arm, leg)
- **Joints**: Connections between links (e.g., hinges, prismatic joints)
- **Visual**: How the robot looks (for visualization)
- **Collision**: Collision properties (for physics simulation)
- **Inertial**: Mass properties (for dynamics simulation)

## Basic URDF Structure

A basic URDF file follows this structure:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Define links -->
  <link name="base_link">
    <!-- Link properties -->
  </link>

  <!-- Define joints -->
  <joint name="joint_name" type="joint_type">
    <parent link="parent_link_name"/>
    <child link="child_link_name"/>
    <!-- Joint properties -->
  </joint>

  <!-- Other elements -->
</robot>
```

## Links in Humanoid Robots

Links represent rigid bodies in the robot. For humanoid robots, typical links include:
- Torso/Body
- Head
- Arms (upper arm, lower arm, hand)
- Legs (upper leg, lower leg, foot)
- Sensors (camera, IMU, etc.)

### Link Definition Example

```xml
<link name="torso">
  <inertial>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <mass value="5.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <geometry>
      <box size="0.2 0.1 0.2"/>
    </geometry>
    <material name="gray">
      <color rgba="0.5 0.5 0.5 1.0"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <geometry>
      <box size="0.2 0.1 0.2"/>
    </geometry>
  </collision>
</link>
```

### Components of a Link:

1. **Inertial**: Physical properties for dynamics simulation
   - `origin`: Center of mass location and orientation
   - `mass`: Mass in kilograms
   - `inertia`: Inertia tensor values

2. **Visual**: How the link appears in visualization
   - `origin`: Visual geometry offset
   - `geometry`: Shape (box, cylinder, sphere, mesh)
   - `material`: Color and texture

3. **Collision**: Collision detection properties
   - `origin`: Collision geometry offset
   - `geometry`: Shape for collision detection

## Joints in Humanoid Robots

Joints connect links and define how they can move relative to each other. For humanoid robots, common joint types include:

- **revolute**: Rotational joint with limited range (e.g., elbow, knee)
- **continuous**: Rotational joint without limits (e.g., wheel)
- **prismatic**: Linear sliding joint (e.g., linear actuator)
- **fixed**: No movement (e.g., mounting point)

### Joint Definition Example

```xml
<joint name="left_hip_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_thigh"/>
  <origin xyz="0 -0.1 -0.1" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  <dynamics damping="1.0" friction="0.1"/>
</joint>
```

### Components of a Joint:

1. **parent/child**: Links that the joint connects
2. **origin**: Position and orientation of the joint
3. **axis**: Axis of rotation or translation
4. **limit**: Movement constraints (for revolute/prismatic joints)
5. **dynamics**: Physical properties for simulation

## Complete Humanoid Robot URDF Example

Here's a simplified URDF for a humanoid robot with basic structure:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base/World link -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.001"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <inertial>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <material name="body_color">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="head_color">
        <color rgba="0.9 0.9 0.9 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <inertial>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
      <material name="arm_color">
        <color rgba="0.6 0.6 0.6 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <link name="left_lower_arm">
    <inertial>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.003"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
      <material name="arm_color"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="2.0"/>
  </joint>

  <joint name="torso_to_left_upper_arm" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0.1 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <joint name="left_upper_arm_to_lower_arm" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="2.0"/>
  </joint>
</robot>
```

## Best Practices for Humanoid URDF

### 1. Proper Kinematic Chains
- Ensure all links are connected through joints
- Create closed loops for stability if needed
- Follow the physical structure of the actual robot

### 2. Accurate Physical Properties
- Use realistic mass values based on actual robot
- Calculate inertia tensors properly
- Consider the payload capacity of joints

### 3. Appropriate Joint Limits
- Set realistic joint limits based on hardware capabilities
- Include safety margins in the limits
- Consider the range of motion for the intended tasks

### 4. Visual vs Collision Models
- Use detailed meshes for visual representation
- Use simplified geometries for collision detection
- Balance accuracy with computational efficiency

### 5. Naming Conventions
- Use descriptive, consistent names
- Follow a clear hierarchy (e.g., `left_arm_upper`, `left_arm_lower`)
- Use underscores to separate components

## URDF Validation and Troubleshooting

### Common Issues:

1. **Invalid XML**: Check for proper XML syntax and matching tags
2. **Disconnected Links**: Ensure all links are connected through joints
3. **Invalid Inertial Properties**: Check that inertia values are physically plausible
4. **Joint Limit Issues**: Verify joint limits are within reasonable ranges

### Validation Tools:

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Parse URDF and show joint information
urdf_to_graphiz /path/to/robot.urdf
```

## Using URDF with ROS 2

To use your URDF model with ROS 2, you typically need to:

1. **Load the URDF** into a parameter server
2. **Use robot_state_publisher** to publish transforms
3. **Visualize** in RViz or simulation

Example launch file:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get URDF file path
    urdf_file = os.path.join(
        get_package_share_directory('your_package'),
        'urdf',
        'humanoid.urdf'
    )

    return LaunchDescription([
        # Robot State Publisher node
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': Command(['xacro ', urdf_file])
            }]
        )
    ])
```

## Hands-on Exercise: Create a Humanoid URDF

### Objective
Create a complete URDF description for a simple humanoid robot with torso, head, and two arms.

### Steps
1. Define the base link and torso
2. Add head and neck joint
3. Create left and right arm kinematic chains
4. Include proper visual and collision geometries
5. Set realistic mass and inertia properties
6. Validate the URDF model

### Expected Outcome
- Complete URDF file for a humanoid robot
- Proper kinematic structure with correct joint limits
- Valid physical properties for simulation
- Model ready for use in ROS 2 and simulation environments

## Summary

In this chapter, we've covered creating URDF descriptions for humanoid robots:
- Basic structure of URDF files
- Links and joints definitions
- Proper kinematic chains for humanoid structures
- Best practices for accurate descriptions
- Validation and usage with ROS 2

URDF is fundamental for simulating and controlling humanoid robots in ROS 2. Creating accurate descriptions ensures proper simulation behavior and enables effective robot development.

## Next Steps

Return to the [Module 1 Overview](./index.md) to continue with the next module or review the concepts covered in this module.