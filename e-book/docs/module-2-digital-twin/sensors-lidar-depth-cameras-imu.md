---
title: "Sensors: LiDAR, Depth Cameras, IMU"
sidebar_label: "Sensor Integration"
description: "Simulating various sensors for robotic perception systems in Gazebo and Unity"
keywords: ["sensors", "lidar", "depth camera", "imu", "perception", "robotics", "simulation"]
---

# Sensors: LiDAR, Depth Cameras, IMU

## Introduction

Sensors are the eyes and ears of robotic systems, providing crucial perception data for navigation, mapping, and interaction. In this chapter, we'll explore how to simulate various sensors in both Gazebo and Unity environments, focusing on LiDAR, depth cameras, and IMUs. Accurate sensor simulation is essential for developing and validating perception algorithms before deployment on physical robots.

## Learning Objectives

By the end of this chapter, you will be able to:
- Simulate LiDAR sensors with realistic noise and range characteristics
- Implement depth camera simulation for 3D perception
- Model IMU sensors for orientation and acceleration data
- Integrate multiple sensors for sensor fusion
- Validate sensor data accuracy for sim-to-real transfer
- Apply sensor calibration techniques in simulation

## Sensor Simulation in Robotics

### Importance of Accurate Sensor Simulation
- **Algorithm Development**: Test perception algorithms safely
- **Data Generation**: Create labeled datasets for training
- **System Integration**: Validate sensor fusion approaches
- **Edge Case Testing**: Simulate rare or dangerous scenarios

### Simulation Challenges
- **Noise Modeling**: Accurately simulate sensor noise
- **Environmental Effects**: Dust, rain, lighting conditions
- **Dynamic Range**: Handle extreme values properly
- **Computational Cost**: Balance accuracy with performance

## LiDAR Simulation

### LiDAR Fundamentals
LiDAR (Light Detection and Ranging) sensors emit laser pulses and measure the time for reflection to return, creating precise distance measurements. Key characteristics:
- **Range**: Maximum and minimum detection distance
- **Resolution**: Angular resolution and accuracy
- **Field of View**: Horizontal and vertical coverage
- **Scan Rate**: Frequency of full scans

### LiDAR in Gazebo

#### SDF Configuration
```xml
<sensor name="lidar_2d" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
        <max_angle>1.570796</max_angle>   <!-- 90 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="lidar_2d_controller" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
  </plugin>
</sensor>
```

#### Advanced LiDAR Configuration
```xml
<sensor name="lidar_3d" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>1080</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>  <!-- -180 degrees -->
        <max_angle>3.14159</max_angle>   <!-- 180 degrees -->
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.261799</min_angle>  <!-- -15 degrees -->
        <max_angle>0.261799</max_angle>   <!-- 15 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="lidar_3d_controller" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <remapping>~/out:=velodyne_points</remapping>
    </ros>
    <output_type>sensor_msgs/PointCloud2</output_type>
  </plugin>
</sensor>
```

### LiDAR Noise Modeling
```xml
<sensor name="lidar_with_noise" type="ray">
  <!-- ... scan configuration ... -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>  <!-- 1cm standard deviation -->
  </noise>
</sensor>
```

## Depth Camera Simulation

### Depth Camera Fundamentals
Depth cameras provide 2D images with depth information for each pixel, enabling 3D scene reconstruction. Key characteristics:
- **Resolution**: Image dimensions (e.g., 640x480)
- **Field of View**: Horizontal and vertical angles
- **Depth Range**: Minimum and maximum measurable distances
- **Accuracy**: Depth measurement precision

### Depth Camera in Gazebo

#### SDF Configuration
```xml
<sensor name="depth_camera" type="depth">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="depth_cam">
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
    <baseline>0.2</baseline>
    <alwaysOn>true</alwaysOn>
    <updateRate>30.0</updateRate>
    <cameraName>depth_camera</cameraName>
    <imageTopicName>/rgb/image_raw</imageTopicName>
    <depthImageTopicName>/depth/image_raw</depthImageTopicName>
    <pointCloudTopicName>/depth/points</pointCloudTopicName>
    <cameraInfoTopicName>/rgb/camera_info</cameraInfoTopicName>
    <depthImageCameraInfoTopicName>/depth/camera_info</depthImageCameraInfoTopicName>
    <frameName>depth_camera_frame</frameName>
    <pointCloudCutoff>0.5</pointCloudCutoff>
    <pointCloudCutoffMax>3.0</pointCloudCutoffMax>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
    <CxPrime>0</CxPrime>
    <Cx>0</Cx>
    <Cy>0</Cy>
    <focalLength>0</focalLength>
    <hackBaseline>0</hackBaseline>
  </plugin>
</sensor>
```

### Depth Camera in Unity

#### Unity Depth Camera Script
```csharp
using UnityEngine;
using System.Collections;

public class DepthCamera : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Camera mainCamera;
    public Shader depthShader;
    public RenderTexture depthTexture;

    [Header("Depth Parameters")]
    public float nearClip = 0.1f;
    public float farClip = 10.0f;
    [Range(0, 1)] public float depthIntensity = 0.5f;

    [Header("Output")]
    public bool saveDepthData = false;
    public string outputDirectory = "DepthData/";

    private Material depthMaterial;

    void Start()
    {
        SetupDepthCamera();
    }

    void SetupDepthCamera()
    {
        if (mainCamera == null)
            mainCamera = GetComponent<Camera>();

        if (depthShader != null)
        {
            depthMaterial = new Material(depthShader);
        }

        if (depthTexture == null)
        {
            depthTexture = new RenderTexture(640, 480, 24);
            depthTexture.name = "DepthTexture";
        }

        mainCamera.SetTargetBuffers(depthTexture.colorBuffer, depthTexture.depthBuffer);
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (depthMaterial != null)
        {
            depthMaterial.SetFloat("_NearClip", nearClip);
            depthMaterial.SetFloat("_FarClip", farClip);
            depthMaterial.SetFloat("_DepthIntensity", depthIntensity);

            Graphics.Blit(source, destination, depthMaterial);
        }
        else
        {
            Graphics.Blit(source, destination);
        }
    }

    public float[,] GetDepthData()
    {
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = depthTexture;

        Texture2D depthTex = new Texture2D(depthTexture.width, depthTexture.height, TextureFormat.RFloat, false);
        depthTex.ReadPixels(new Rect(0, 0, depthTexture.width, depthTexture.height), 0, 0);
        depthTex.Apply();

        RenderTexture.active = currentRT;

        Color[] pixels = depthTex.GetPixels();
        float[,] depthData = new float[depthTexture.width, depthTexture.height];

        for (int y = 0; y < depthTexture.height; y++)
        {
            for (int x = 0; x < depthTexture.width; x++)
            {
                depthData[x, y] = pixels[y * depthTexture.width + x].r;
            }
        }

        DestroyImmediate(depthTex);
        return depthData;
    }

    public void SaveDepthImage()
    {
        if (saveDepthData)
        {
            RenderTexture currentRT = RenderTexture.active;
            RenderTexture.active = depthTexture;

            Texture2D depthTex = new Texture2D(depthTexture.width, depthTexture.height, TextureFormat.RGB24, false);
            depthTex.ReadPixels(new Rect(0, 0, depthTexture.width, depthTexture.height), 0, 0);
            depthTex.Apply();

            byte[] bytes = depthTex.EncodeToPNG();
            string filename = System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss") + "_depth.png";
            System.IO.File.WriteAllBytes(outputDirectory + filename, bytes);

            RenderTexture.active = currentRT;
            DestroyImmediate(depthTex);
        }
    }
}
```

## IMU Simulation

### IMU Fundamentals
Inertial Measurement Units measure linear acceleration and angular velocity, providing crucial data for robot localization and control. Key measurements:
- **Linear Acceleration**: 3-axis acceleration (m/s²)
- **Angular Velocity**: 3-axis rotation rates (rad/s)
- **Orientation**: Derived from integration (optional)
- **Magnetic Field**: For absolute orientation (magnetometer)

### IMU in Gazebo

#### SDF Configuration
```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.00174533</stddev>  <!-- 0.1 deg/s in rad/s -->
          <bias_mean>0.00001</bias_mean>
          <bias_stddev>0.000001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.00174533</stddev>
          <bias_mean>0.00001</bias_mean>
          <bias_stddev>0.000001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.00174533</stddev>
          <bias_mean>0.00001</bias_mean>
          <bias_stddev>0.000001</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.0e-3</stddev>
          <bias_mean>0.01</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.0e-3</stddev>
          <bias_mean>0.01</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.0e-3</stddev>
          <bias_mean>0.01</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
    <ros>
      <remapping>~/out:=imu/data</remapping>
    </ros>
    <frame_name>imu_link</frame_name>
    <topic_name>imu/data</topic_name>
    <body_name>imu_body</body_name>
  </plugin>
</sensor>
```

### IMU Data Processing
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import numpy as np
from scipy.spatial.transform import Rotation as R

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        # Subscribe to IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publisher for processed data
        self.orientation_pub = self.create_publisher(
            Vector3,
            '/robot_orientation',
            10
        )

        # IMU state variables
        self.orientation = R.identity()
        self.angular_velocity = np.zeros(3)
        self.linear_acceleration = np.zeros(3)

        # Timing
        self.prev_time = self.get_clock().now()

        self.get_logger().info('IMU Processor initialized')

    def imu_callback(self, msg):
        # Extract measurements
        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        self.linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Get current time
        current_time = rclpy.time.Time.from_msg(msg.header.stamp)
        dt = (current_time.nanoseconds - self.prev_time.nanoseconds) / 1e9
        self.prev_time = current_time

        if dt > 0:
            # Integrate angular velocity to get orientation
            angular_speed = np.linalg.norm(self.angular_velocity)
            if angular_speed > 1e-6:  # Avoid division by zero
                axis = self.angular_velocity / angular_speed
                rotation_vector = axis * angular_speed * dt
                rotation = R.from_rotvec(rotation_vector)
                self.orientation = self.orientation * rotation

        # Publish processed orientation
        orientation_msg = Vector3()
        euler = self.orientation.as_euler('xyz')
        orientation_msg.x = euler[0]  # Roll
        orientation_msg.y = euler[1]  # Pitch
        orientation_msg.z = euler[2]  # Yaw

        self.orientation_pub.publish(orientation_msg)

def main(args=None):
    rclpy.init(args=args)
    processor = IMUProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Shutting down')
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Fusion

### Multi-Sensor Integration
Combining data from multiple sensors improves perception accuracy and robustness:
- **LiDAR + Camera**: Precise depth + rich visual information
- **IMU + Encoders**: Motion tracking + wheel odometry
- **Sensor fusion algorithms**: Kalman filters, particle filters

### Example Sensor Fusion Node
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribers for different sensors
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Publisher for fused estimate
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/fused_pose', 10
        )

        # State estimation variables
        self.position = np.zeros(3)
        self.orientation = R.identity()
        self.velocity = np.zeros(3)
        self.covariance = np.eye(6) * 0.1  # Initial uncertainty

        self.get_logger().info('Sensor Fusion Node initialized')

    def lidar_callback(self, msg):
        # Process LiDAR data for position refinement
        # This is a simplified example
        self.get_logger().debug(f'Received LiDAR scan with {len(msg.ranges)} points')

    def imu_callback(self, msg):
        # Process IMU data for orientation and acceleration
        # This is a simplified example
        self.get_logger().debug('Received IMU data')

    def odom_callback(self, msg):
        # Process odometry data for position estimate
        # This is a simplified example
        self.get_logger().debug('Received odometry data')

    def publish_fused_estimate(self):
        # Combine all sensor data into a fused estimate
        # This would implement a proper fusion algorithm
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        # Fill in position and orientation
        pose_msg.pose.pose.position.x = self.position[0]
        pose_msg.pose.pose.position.y = self.position[1]
        pose_msg.pose.pose.position.z = self.position[2]

        quat = self.orientation.as_quat()
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]

        # Fill in covariance
        for i in range(36):
            pose_msg.pose.covariance[i] = self.covariance.flatten()[i]

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        fusion_node.get_logger().info('Shutting down')
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Calibration in Simulation

### Intrinsic Calibration
- **Camera**: Focal length, principal point, distortion
- **LiDAR**: Range offset, angular accuracy
- **IMU**: Bias, scale factor, alignment

### Extrinsic Calibration
- **Sensor-to-sensor**: Relative positions and orientations
- **Sensor-to-robot**: Mounting positions and angles
- **Coordinate systems**: Frame relationships

## Simulation-Based Lab Exercise: Multi-Sensor Perception System

### Objective
Create a simulated robot equipped with LiDAR, depth camera, and IMU sensors, and implement basic perception algorithms.

### Prerequisites
- Gazebo with ROS2 interface
- Basic understanding of sensor_msgs

### Steps

#### 1. Create Multi-Sensor Robot Model
Create an SDF file with all three sensor types:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="sensor_robot">
    <!-- Robot chassis -->
    <link name="chassis">
      <pose>0 0 0.2 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.5</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.5</iyy>
          <iyz>0.0</iyz>
          <izz>0.5</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </visual>
    </link>

    <!-- LiDAR sensor -->
    <sensor name="lidar" type="ray">
      <pose>0.2 0 0.1 0 0 0</pose>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <remapping>~/out:=scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
      </plugin>
    </sensor>

    <!-- Depth camera -->
    <sensor name="depth_camera" type="depth">
      <pose>0.25 0 0.1 0 0 0</pose>
      <camera name="depth_cam">
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
      <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
        <cameraName>depth_camera</cameraName>
        <imageTopicName>/depth/image_raw</imageTopicName>
        <depthImageTopicName>/depth/image_depth</depthImageTopicName>
        <pointCloudTopicName>/depth/points</pointCloudTopicName>
        <frameName>depth_camera_frame</frameName>
      </plugin>
    </sensor>

    <!-- IMU sensor -->
    <sensor name="imu" type="imu">
      <pose>0 0 0.1 0 0 0</pose>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <stddev>0.00174533</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <stddev>0.00174533</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <stddev>0.00174533</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <stddev>1.0e-3</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <stddev>1.0e-3</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <stddev>1.0e-3</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
      <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
        <ros>
          <remapping>~/out:=imu/data</remapping>
        </ros>
        <frame_name>imu_link</frame_name>
        <topic_name>imu/data</topic_name>
      </plugin>
    </sensor>
  </model>
</sdf>
```

#### 2. Create Perception Node
Develop a ROS2 node that processes all sensor data:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np

class MultiSensorPerception(Node):
    def __init__(self):
        super().__init__('multi_sensor_perception')

        # Create subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/depth/image_depth', self.depth_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Create publishers for processed data
        self.obstacle_pub = self.create_publisher(
            LaserScan, '/obstacle_scan', 10
        )

        # Initialize bridge for image processing
        self.bridge = CvBridge()

        self.get_logger().info('Multi-Sensor Perception Node initialized')

    def lidar_callback(self, msg):
        # Process LiDAR data to detect obstacles
        # This is a simplified example
        ranges = np.array(msg.ranges)
        # Remove invalid ranges
        ranges = np.where((ranges >= msg.range_min) & (ranges <= msg.range_max), ranges, np.inf)

        # Detect obstacles within 1m
        obstacle_ranges = np.where(ranges < 1.0, ranges, np.inf)

        # Create obstacle scan message
        obstacle_msg = LaserScan()
        obstacle_msg.header = msg.header
        obstacle_msg.angle_min = msg.angle_min
        obstacle_msg.angle_max = msg.angle_max
        obstacle_msg.angle_increment = msg.angle_increment
        obstacle_msg.time_increment = msg.time_increment
        obstacle_msg.scan_time = msg.scan_time
        obstacle_msg.range_min = msg.range_min
        obstacle_msg.range_max = msg.range_max
        obstacle_msg.ranges = obstacle_ranges.tolist()

        self.obstacle_pub.publish(obstacle_msg)

    def depth_callback(self, msg):
        # Process depth image
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            # Process depth image for obstacle detection or mapping
            self.get_logger().debug(f'Depth image shape: {depth_image.shape}')
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def imu_callback(self, msg):
        # Process IMU data
        self.get_logger().debug('Processing IMU data')

def main(args=None):
    rclpy.init(args=args)
    perception_node = MultiSensorPerception()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Shutting down')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 3. Create Launch File
Create a launch file to start the simulation and perception node:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to world file
    world_file = os.path.join(
        get_package_share_directory('your_package'),
        'worlds',
        'sensor_world.sdf'
    )

    return LaunchDescription([
        # Launch Gazebo with world
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file, '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),

        # Launch perception node
        Node(
            package='your_package',
            executable='multi_sensor_perception',
            name='multi_sensor_perception',
            output='screen',
        )
    ])
```

#### 4. Test and Validate
- Run the simulation
- Verify all sensors are publishing data
- Test perception algorithms
- Validate sensor fusion results

### Expected Outcome
- Robot model with LiDAR, depth camera, and IMU
- Working perception node processing all sensor data
- Obstacle detection using multiple sensors
- Foundation for advanced perception algorithms

## Best Practices for Sensor Simulation

### 1. Accurate Noise Modeling
- Use realistic noise parameters based on real sensors
- Include bias and drift for long-term simulation
- Model environmental effects (weather, lighting)

### 2. Performance Optimization
- Use appropriate update rates for each sensor
- Simplify models when possible
- Use multi-threading for sensor processing

### 3. Validation Against Reality
- Compare simulation output with real sensor data
- Validate perception algorithms in both environments
- Adjust simulation parameters based on real-world performance

### 4. Safety Considerations
- Model sensor failures and limitations
- Include fallback behaviors for sensor loss
- Test edge cases in simulation

## Summary

In this chapter, we've explored sensor simulation for robotic systems:
- LiDAR simulation with realistic noise and range characteristics
- Depth camera simulation for 3D perception
- IMU modeling for orientation and acceleration data
- Sensor fusion techniques for combining multiple sensors
- Calibration methods for accurate simulation
- Best practices for sensor simulation development

Accurate sensor simulation is crucial for developing and validating perception algorithms that can be safely transferred to real robotic systems.

## Next Steps

Return to the [Module 2 Overview](./index.md) to continue with the next module or review the concepts covered in this module.