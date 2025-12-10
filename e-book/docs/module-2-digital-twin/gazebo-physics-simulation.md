---
title: Gazebo Physics Simulation
sidebar_label: Gazebo Physics
description: Understanding physics-based simulation for accurate robot behavior in Gazebo
keywords: [gazebo, physics, simulation, robotics, collisions, dynamics]
---

# Gazebo Physics Simulation

## Introduction

Gazebo is a powerful physics-based simulation environment that enables realistic modeling of robotic systems in virtual environments. It provides accurate physics simulation, high-quality graphics, and convenient interfaces for robotics research and development. In this chapter, we'll explore how to create accurate physics simulations that can be transferred to real robots.

## Learning Objectives

By the end of this chapter, you will be able to:
- Set up and configure Gazebo simulation environments
- Create accurate physics models for robotic systems
- Configure collision and visual properties
- Simulate realistic robot behaviors and interactions
- Validate simulation accuracy for sim-to-real transfer

## Gazebo Architecture

Gazebo operates on a client-server architecture:
- **Gazebo Server**: Handles physics simulation, sensor simulation, and plugin execution
- **Gazebo Client**: Provides visualization and user interaction
- **Plugins**: Extend functionality for custom sensors, controllers, and interfaces

### Core Components:
- **Physics Engine**: Supports ODE, Bullet, Simbody, and DART
- **Sensor Simulation**: Cameras, LiDAR, IMU, force/torque sensors
- **Rendering Engine**: OpenGL-based visualization
- **ROS Integration**: Direct interfaces for ROS/ROS 2 communication

## Setting Up Gazebo Environment

### Installation
```bash
# For ROS 2 Humble on Ubuntu 22.04
sudo apt update
sudo apt install ros-humble-gazebo-*
sudo apt install gazebo
```

### Basic Launch
```bash
# Launch Gazebo server only
gzserver

# Launch Gazebo client only
gzclient

# Launch both together
gazebo
```

## Creating Physics Models

### Model Structure
Gazebo models follow a specific directory structure:
```
models/
└── my_robot/
    ├── model.config
    └── meshes/
    │   ├── link1.dae
    │   └── link2.stl
    └── model.sdf
```

### SDF Format
SDF (Simulation Description Format) is XML-based and similar to URDF but with additional simulation-specific elements:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_robot">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <box>
            <size>1.0 0.5 0.2</size>
          </box>
        </geometry>
        <!-- Surface properties for physics simulation -->
        <surface>
          <friction>
            <ode>
              <mu>1.0</mu>
              <mu2>1.0</mu2>
            </ode>
          </friction>
          <bounce>
            <restitution_coefficient>0.1</restitution_coefficient>
            <threshold>100000</threshold>
          </bounce>
          <contact>
            <ode>
              <soft_cfm>0</soft_cfm>
              <soft_erp>0.2</soft_erp>
              <kp>1e+13</kp>
              <kd>1</kd>
              <max_vel>100.0</max_vel>
              <min_depth>0.001</min_depth>
            </ode>
          </contact>
        </surface>
      </collision>

      <visual name="visual">
        <geometry>
          <box>
            <size>1.0 0.5 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.4 0.4 0.4 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>

    <!-- Joint definition -->
    <joint name="wheel_joint" type="revolute">
      <parent>chassis</parent>
      <child>wheel</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>

    <link name="wheel">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
      <collision name="wheel_collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="wheel_visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
```

## Physics Properties and Accuracy

### Mass and Inertia
Accurate mass and inertia properties are crucial for realistic simulation:
- Use CAD software to calculate exact values
- Measure physical robot when possible
- Include all components (electronics, batteries, etc.)

### Friction Parameters
Friction significantly affects robot behavior:
- **Static friction (mu)**: Prevents sliding when force is low
- **Dynamic friction**: Affects motion when sliding occurs
- Tune based on real-world contact surfaces

### Damping and Compliance
- **Damping**: Energy loss in joints and motion
- **Compliance**: Soft contact behavior for more stable simulation
- Balance stability with realism

## Collision Detection

### Collision vs Visual Geometry
- **Collision geometry**: Used for physics simulation
- **Visual geometry**: Used for rendering
- Collision geometry can be simplified for performance

### Contact Materials
Define how different materials interact:
```xml
<surface>
  <friction>
    <ode>
      <mu>0.5</mu>  <!-- Static friction coefficient -->
      <mu2>0.3</mu2> <!-- Dynamic friction coefficient -->
    </ode>
  </friction>
</surface>
```

## Environment Modeling

### World Files
World files define the simulation environment:
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Include models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define static obstacles -->
    <model name="wall_1">
      <pose>0 5 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Physics parameters -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
  </world>
</sdf>
```

### Terrain and Outdoor Simulation
- Use heightmap for complex terrain
- Configure atmospheric properties
- Add dynamic weather effects

## Sensor Simulation

### Common Sensors in Gazebo
- **Camera**: Visual perception
- **Depth Camera**: 3D point cloud data
- **LiDAR**: Range measurements
- **IMU**: Inertial measurements
- **Force/Torque**: Contact forces

### Sensor Configuration Example
```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="camera">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <frame_name>camera_frame</frame_name>
    <topic_name>camera/image_raw</topic_name>
  </plugin>
</sensor>
```

## Simulation Accuracy Considerations

### Model Fidelity
Balance accuracy with computational efficiency:
- High-fidelity models: More accurate but slower
- Simplified models: Faster but potentially less accurate
- Validate against real robot performance

### Physics Engine Selection
- **ODE**: Good balance of speed and accuracy
- **Bullet**: Better for complex contacts
- **DART**: Advanced dynamics, good for humanoid robots

### Time Step and Stability
- Smaller time steps: More accurate but slower
- Larger time steps: Faster but potentially unstable
- Balance based on simulation requirements

## Sim-to-Real Transfer Techniques

### Domain Randomization
Randomize simulation parameters to improve robustness:
- Mass and inertia variations
- Friction coefficients
- Visual appearance
- Lighting conditions

### System Identification
- Identify real robot parameters
- Match simulation to reality
- Validate with physical experiments

### Control Parameter Tuning
- Develop controllers in simulation
- Adjust for real-world differences
- Use adaptive control techniques

## Best Practices for Accurate Simulation

### 1. Validate Individual Components
- Test each joint independently
- Verify sensor outputs match expectations
- Check collision detection behavior

### 2. Calibrate Against Reality
- Measure real robot parameters
- Adjust simulation to match real behavior
- Validate with simple test cases

### 3. Gradual Complexity Increase
- Start with simple models
- Add complexity gradually
- Validate at each step

### 4. Safety Margins
- Use conservative parameters when uncertain
- Include safety factors in simulation
- Test failure scenarios

## Simulation-Based Lab Exercise: Differential Drive Robot

### Objective
Create a differential drive robot model in Gazebo and validate its kinematic behavior.

### Prerequisites
- Gazebo installed
- Basic understanding of URDF/SDF

### Steps
1. Create a differential drive robot model with two wheels
2. Configure physics properties accurately
3. Add a simple controller plugin
4. Test the robot's response to velocity commands
5. Compare simulation results with theoretical kinematics

### Expected Outcome
- Robot model with accurate physics simulation
- Proper wheel contact and friction modeling
- Kinematically accurate motion
- Foundation for more complex robot behaviors

## Domain Randomization and Synthetic Data Generation

### Domain Randomization

Domain randomization is a technique that improves the robustness of machine learning models by training them on data from randomized simulation environments. The approach helps bridge the sim-to-real gap by exposing the model to a wide variety of conditions it might encounter in reality.

#### Implementation Strategies:

1. **Visual Randomization**:
   - Randomize textures and materials
   - Vary lighting conditions
   - Change colors of objects and environments
   - Adjust camera parameters (focus, exposure, noise)

2. **Physical Randomization**:
   - Randomize friction coefficients
   - Vary mass and inertia properties
   - Adjust damping parameters
   - Modify surface properties

3. **Geometric Randomization**:
   - Change object shapes within reasonable bounds
   - Randomize object sizes
   - Adjust placement and positioning

### Example Domain Randomization in Gazebo

```xml
<world name="randomized_world">
  <!-- Randomize lighting -->
  <light name="sun" type="directional">
    <pose>0 0 10 0 0 0</pose>
    <diffuse>$(random_range 0.5 1.0) $(random_range 0.5 1.0) $(random_range 0.5 1.0) 1</diffuse>
    <specular>0.1 0.1 0.1 1</specular>
    <attenuation>
      <range>10</range>
      <constant>0.9</constant>
      <linear>0.01</linear>
      <quadratic>0.001</quadratic>
    </attenuation>
  </light>

  <!-- Randomize objects -->
  <model name="random_object">
    <pose>$(random_range -5 5) $(random_range -5 5) 0.5 0 0 $(random_range -3.14 3.14)</pose>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>$(random_range 0.5 2.0) $(random_range 0.5 2.0) $(random_range 0.5 2.0)</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>$(random_range 0.5 2.0) $(random_range 0.5 2.0) $(random_range 0.5 2.0)</size>
          </box>
        </geometry>
        <material>
          <ambient>$(random_range 0.2 0.8) $(random_range 0.2 0.8) $(random_range 0.2 0.8) 1</ambient>
          <diffuse>$(random_range 0.2 0.8) $(random_range 0.2 0.8) $(random_range 0.2 0.8) 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</world>
```

### Synthetic Data Generation

Synthetic data generation leverages simulation environments to create large, labeled datasets for training machine learning models. This is particularly valuable for robotics applications where collecting real-world data can be expensive or dangerous.

#### Benefits:
- **Cost-effective**: Generate large datasets without physical hardware
- **Controlled conditions**: Create specific scenarios for training
- **Perfect labeling**: Ground truth data available for all sensor readings
- **Safety**: Train on dangerous scenarios without risk

#### Applications:
- **Computer Vision**: Generate labeled images for object detection
- **Sensor Fusion**: Create correlated sensor data for training
- **Reinforcement Learning**: Create diverse training environments
- **Failure Modes**: Simulate rare failure scenarios for robustness

### Python Script for Data Collection

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os
from datetime import datetime

class SyntheticDataCollector(Node):
    def __init__(self):
        super().__init__('synthetic_data_collector')

        # Create directory for data collection
        self.data_dir = f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/images", exist_ok=True)
        os.makedirs(f"{self.data_dir}/lidar", exist_ok=True)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Data collection parameters
        self.bridge = CvBridge()
        self.sample_count = 0
        self.collection_active = True

        # Statistics
        self.stats = {
            'images_collected': 0,
            'lidar_samples': 0,
            'imu_samples': 0
        }

        self.get_logger().info(f'Synthetic data collection started in: {self.data_dir}')

    def image_callback(self, msg):
        if not self.collection_active:
            return

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Save image with timestamp
            image_filename = f"{self.data_dir}/images/frame_{self.sample_count:06d}.png"
            cv2.imwrite(image_filename, cv_image)

            # Store metadata
            metadata = {
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                'frame_id': msg.header.frame_id,
                'encoding': msg.encoding,
                'height': msg.height,
                'width': msg.width,
                'step': msg.step
            }

            metadata_filename = f"{self.data_dir}/images/frame_{self.sample_count:06d}_meta.json"
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.stats['images_collected'] += 1
            self.sample_count += 1

            self.get_logger().debug(f'Saved image: {image_filename}')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def lidar_callback(self, msg):
        if not self.collection_active:
            return

        try:
            # Store LiDAR data
            lidar_data = {
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                'frame_id': msg.header.frame_id,
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment,
                'time_increment': msg.time_increment,
                'scan_time': msg.scan_time,
                'range_min': msg.range_min,
                'range_max': msg.range_max,
                'ranges': list(msg.ranges),
                'intensities': list(msg.intensities) if msg.intensities else []
            }

            lidar_filename = f"{self.data_dir}/lidar/lidar_{self.stats['lidar_samples']:06d}.json"
            with open(lidar_filename, 'w') as f:
                json.dump(lidar_data, f, indent=2)

            self.stats['lidar_samples'] += 1
            self.get_logger().debug(f'Saved LiDAR data: {lidar_filename}')

        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR: {e}')

    def imu_callback(self, msg):
        if not self.collection_active:
            return

        try:
            # Store IMU data
            imu_data = {
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                'frame_id': msg.header.frame_id,
                'orientation': {
                    'x': msg.orientation.x,
                    'y': msg.orientation.y,
                    'z': msg.orientation.z,
                    'w': msg.orientation.w
                },
                'angular_velocity': {
                    'x': msg.angular_velocity.x,
                    'y': msg.angular_velocity.y,
                    'z': msg.angular_velocity.z
                },
                'linear_acceleration': {
                    'x': msg.linear_acceleration.x,
                    'y': msg.linear_acceleration.y,
                    'z': msg.linear_acceleration.z
                }
            }

            self.stats['imu_samples'] += 1

        except Exception as e:
            self.get_logger().error(f'Error processing IMU: {e}')

    def get_stats(self):
        return self.stats

def main(args=None):
    rclpy.init(args=args)
    collector = SyntheticDataCollector()

    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info('Data collection stopped by user')
        stats = collector.get_stats()
        collector.get_logger().info(f'Collection statistics: {stats}')
    finally:
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

In this chapter, we've explored Gazebo physics simulation:
- Setting up and configuring simulation environments
- Creating accurate physics models with proper mass, inertia, and friction
- Implementing collision detection and sensor simulation
- Techniques for improving sim-to-real transfer accuracy
- Domain randomization for robust model training
- Synthetic data generation for machine learning applications
- Best practices for simulation development

Accurate physics simulation is fundamental to the simulation-to-reality pipeline, allowing us to develop and validate robotic behaviors safely before real-world deployment. Domain randomization and synthetic data generation further enhance the value of simulation by enabling robust model training and large-scale data collection.

## Next Steps

Continue to the next chapter: [Unity Rendering and Human-Robot Interaction](./unity-rendering.md) to learn about visual simulation and interaction systems.