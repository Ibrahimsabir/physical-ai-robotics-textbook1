---
title: ROS 2 Nodes, Topics, and Services
sidebar_label: Nodes, Topics, and Services
description: Understanding the fundamental communication patterns in ROS 2
keywords: [ros2, nodes, topics, services, communication, robotics]
---

# ROS 2 Nodes, Topics, and Services

## Introduction

In this chapter, we'll explore the three fundamental communication patterns in ROS 2: nodes, topics, and services. These concepts form the backbone of ROS 2's distributed computing architecture and enable the modular design that makes robotic systems flexible and maintainable.

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain the concept of ROS 2 nodes and their role in robotic systems
- Implement publishers and subscribers using topics for asynchronous communication
- Create and use services for synchronous request-response communication
- Design robust communication patterns that follow safety-first principles

## What are ROS 2 Nodes?

A **node** is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 program. Multiple nodes are usually assembled together to form a complete robotic application. Nodes written in different programming languages can be run at the same time on the same system or on different systems and still communicate with each other.

### Key Characteristics of Nodes:
- Each node runs a specific task within the robotic system
- Nodes communicate with each other through topics, services, and actions
- Nodes can be started and stopped independently
- Nodes are designed to be modular and reusable

## Topics and Publishers/Subscribers

**Topics** enable asynchronous, many-to-many communication between nodes using a publish-subscribe pattern. This is ideal for streaming data like sensor readings, robot states, or log messages.

### Publisher-Subscriber Pattern

- **Publisher**: A node that sends data to a topic
- **Subscriber**: A node that receives data from a topic
- **Message**: The data structure exchanged between publisher and subscriber

### Example: Creating a Simple Publisher and Subscriber

Here's a Python example of a simple publisher that sends messages:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

And here's a corresponding subscriber:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services and Clients

**Services** enable synchronous, request-response communication between nodes. This is ideal for operations that need a response, such as setting parameters, triggering actions, or requesting specific information.

### Service-Client Pattern

- **Service**: A node that provides a specific function
- **Client**: A node that requests a function from a service
- **Request/Response**: Structured data for the request and response

### Example: Creating a Simple Service and Client

Service definition (add to a `srv` directory in your package):

```python
# AddTwoInts.srv
int64 a
int64 b
---
int64 sum
```

Service implementation:

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Client implementation:

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Communication Design

### 1. Topic Naming Conventions
- Use descriptive, lowercase names with underscores
- Group related topics under common prefixes
- Example: `/robot_1/sensors/lidar_scan`, `/robot_1/controls/motor_speed`

### 2. Message Design
- Keep messages lightweight but informative
- Use appropriate data types for efficiency
- Include timestamps when temporal information is important
- Design messages for future extensibility

### 3. Quality of Service (QoS) Settings
- Choose appropriate QoS profiles based on application needs
- Consider reliability, durability, and history requirements
- For safety-critical data, use reliable delivery and keep-all history

### 4. Safety Considerations
- Implement timeouts for service calls
- Design fail-safe behaviors when communication fails
- Use latching for important state information
- Monitor communication health and report issues

## Simulation-Based Lab Exercise: Create a Publisher-Subscriber Pair in Gazebo

### Objective
Create a ROS 2 publisher that publishes robot pose information and a subscriber that logs this information, all within a Gazebo simulation environment.

### Prerequisites
- ROS 2 Humble Hawksbill (or later) installed
- Gazebo Garden (or compatible version) installed
- Basic understanding of ROS 2 concepts

### Steps

#### 1. Set up the Simulation Environment
```bash
# Create a new workspace for the simulation
mkdir -p ~/ros2_simulation_ws/src
cd ~/ros2_simulation_ws

# Create a new package for the demo
colcon build
source install/setup.bash
cd src
ros2 pkg create --build-type ament_python robot_pose_demo
cd robot_pose_demo
```

#### 2. Create the Publisher Node
Create a file `robot_pose_demo/publisher_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
import math
import time

class RobotPosePublisher(Node):
    def __init__(self):
        super().__init__('robot_pose_publisher')

        # Create publisher for robot pose
        self.publisher = self.create_publisher(Pose, 'robot_pose', 10)

        # Timer to publish at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_pose)

        # Robot state variables
        self.time_offset = time.time()
        self.get_logger().info('Robot Pose Publisher started')

    def publish_pose(self):
        msg = Pose()

        # Create a circular motion pattern for simulation
        current_time = time.time() - self.time_offset
        radius = 2.0

        msg.position.x = radius * math.cos(current_time * 0.5)
        msg.position.y = radius * math.sin(current_time * 0.5)
        msg.position.z = 0.0  # Ground level

        # Simple orientation (pointing in direction of movement)
        msg.orientation.z = math.sin(current_time * 0.25)
        msg.orientation.w = math.cos(current_time * 0.25)

        self.publisher.publish(msg)
        self.get_logger().info(f'Published pose: x={msg.position.x:.2f}, y={msg.position.y:.2f}')

def main(args=None):
    rclpy.init(args=args)
    publisher = RobotPosePublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info('Shutting down')
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 3. Create the Subscriber Node
Create a file `robot_pose_demo/subscriber_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np

class RobotPoseSubscriber(Node):
    def __init__(self):
        super().__init__('robot_pose_subscriber')

        # Create subscriber for robot pose
        self.subscriber = self.create_subscription(
            Pose,
            'robot_pose',
            self.pose_callback,
            10
        )

        self.get_logger().info('Robot Pose Subscriber started')

    def pose_callback(self, msg):
        # Calculate distance from origin
        distance = (msg.position.x**2 + msg.position.y**2)**0.5

        self.get_logger().info(
            f'Received pose - Position: ({msg.position.x:.2f}, {msg.position.y:.2f}, {msg.position.z:.2f}), '
            f'Distance from origin: {distance:.2f}'
        )

        # Safety check: Stop if robot goes too far
        if distance > 5.0:
            self.get_logger().warn('Robot is moving too far from origin!')

def main(args=None):
    rclpy.init(args=args)
    subscriber = RobotPoseSubscriber()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        subscriber.get_logger().info('Shutting down')
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 4. Update setup.py for the package
Update `robot_pose_demo/setup.py`:

```python
from setuptools import find_packages, setup

package_name = 'robot_pose_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Robot pose publisher and subscriber demo',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_publisher = robot_pose_demo.publisher_node:main',
            'pose_subscriber = robot_pose_demo.subscriber_node:main',
        ],
    },
)
```

#### 5. Launch the Simulation
Create a launch file `robot_pose_demo/launch/pose_demo.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # Launch the publisher node
        Node(
            package='robot_pose_demo',
            executable='pose_publisher',
            name='pose_publisher',
            output='screen',
        ),

        # Launch the subscriber node
        Node(
            package='robot_pose_demo',
            executable='pose_subscriber',
            name='pose_subscriber',
            output='screen',
        ),
    ])
```

#### 6. Build and Run
```bash
# From the workspace root
cd ~/ros2_simulation_ws
colcon build --packages-select robot_pose_demo
source install/setup.bash

# Run the simulation
ros2 launch robot_pose_demo pose_demo.launch.py
```

#### 7. Visualization in RViz (Optional)
To visualize the robot pose in RViz:
```bash
# In a new terminal
source ~/ros2_simulation_ws/install/setup.bash
ros2 run rviz2 rviz2
```
Then add a "Pose" display and set the topic to `/robot_pose`.

### Expected Outcome
- Publisher node creates simulated robot pose data in a circular pattern
- Subscriber node receives and logs the pose information
- Distance safety check triggers warnings when robot moves too far
- Both nodes run in the simulated environment and communicate properly
- System demonstrates real-time ROS 2 communication patterns

### Extension Activities
1. Modify the motion pattern to follow a square or figure-8 path
2. Add multiple subscribers to demonstrate the many-to-many communication pattern
3. Integrate with a Gazebo simulation by publishing to Gazebo's command topics
4. Add a service call to pause/resume the robot motion

## Summary

In this chapter, we've covered the fundamental communication patterns in ROS 2:
- **Nodes**: The basic computational units of ROS 2 systems
- **Topics**: Asynchronous publish-subscribe communication for streaming data
- **Services**: Synchronous request-response communication for specific operations

These patterns enable the modular, distributed architecture that makes ROS 2 ideal for complex robotic systems. Understanding these concepts is crucial for designing robust and maintainable robotic applications.

## Next Steps

Continue to the next chapter: [Python Agents Controlling ROS 2](./python-agents-ros2.md) to learn how to create intelligent agents that control robotic systems using Python.