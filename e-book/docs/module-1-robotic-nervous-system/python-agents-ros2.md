---
title: Python Agents Controlling ROS 2
sidebar_label: Python Agents
description: Developing intelligent agents that control ROS 2 systems using Python
keywords: [ros2, python, agents, control, robotics, ai]
---

# Python Agents Controlling ROS 2

## Introduction

In this chapter, we'll explore how to develop intelligent agents using Python that can control and interact with ROS 2 systems. Python agents provide a powerful way to implement high-level control logic, decision-making algorithms, and AI-based behaviors in robotic systems.

## Learning Objectives

By the end of this chapter, you will be able to:
- Create Python nodes that act as intelligent agents in ROS 2 systems
- Implement decision-making logic for robotic control
- Design modular, reusable Python agents
- Integrate AI and machine learning components with ROS 2
- Ensure safety and reliability in agent-based control systems

## What is a Python Agent in ROS 2 Context?

A **Python agent** in ROS 2 is a node that implements higher-level logic for controlling robotic systems. These agents can:
- Monitor sensor data and make decisions
- Plan and execute complex behaviors
- Coordinate multiple robotic components
- Adapt to changing environmental conditions
- Interface with AI and machine learning models

## Basic Python Agent Structure

A Python agent in ROS 2 typically follows this structure:

```python
import rclpy
from rclpy.node import Node
import time

class RobotAgent(Node):
    def __init__(self):
        super().__init__('robot_agent')

        # Initialize agent state
        self.agent_state = 'IDLE'
        self.sensors_data = {}

        # Create subscribers for sensor data
        self.create_subscription(
            # sensor message type
            # topic name
            # callback function
            # queue size
        )

        # Create publishers for control commands
        self.cmd_publisher = self.create_publisher(
            # command message type
            # command topic
            # queue size
        )

        # Create timers for periodic behavior
        self.agent_timer = self.create_timer(
            0.1,  # seconds
            self.agent_behavior
        )

        self.get_logger().info('Robot Agent initialized')

    def sensor_callback(self, msg):
        """Process incoming sensor data"""
        self.sensors_data['sensor_name'] = msg.data
        # Process sensor data as needed

    def agent_behavior(self):
        """Main agent behavior loop"""
        if self.agent_state == 'IDLE':
            self.handle_idle_state()
        elif self.agent_state == 'ACTIVE':
            self.handle_active_state()
        # Add more states as needed

    def handle_idle_state(self):
        """Handle IDLE state behavior"""
        # Implement idle state logic
        pass

    def handle_active_state(self):
        """Handle ACTIVE state behavior"""
        # Implement active state logic
        pass

def main(args=None):
    rclpy.init(args=args)
    agent = RobotAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Agent interrupted by user')
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Example: Simple Navigation Agent

Let's implement a simple navigation agent that moves a robot to a target location while avoiding obstacles:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import math

class NavigationAgent(Node):
    def __init__(self):
        super().__init__('navigation_agent')

        # Robot state
        self.current_position = Point()
        self.current_orientation = 0.0
        self.target_position = Point()
        self.target_position.x = 5.0  # Set target x coordinate
        self.target_position.y = 5.0  # Set target y coordinate

        # Control parameters
        self.linear_speed = 0.5
        self.angular_speed = 0.5
        self.min_distance_to_target = 0.5
        self.safety_distance = 0.5

        # Create subscribers
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.laser_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Create publisher for velocity commands
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create timer for navigation behavior
        self.nav_timer = self.create_timer(0.1, self.navigate)

        self.get_logger().info('Navigation Agent initialized')

    def odom_callback(self, msg):
        """Update robot position from odometry"""
        self.current_position.x = msg.pose.pose.position.x
        self.current_position.y = msg.pose.pose.position.y

        # Extract orientation (simplified - in real implementation, use quaternions)
        orientation_q = msg.pose.pose.orientation
        # Convert quaternion to euler (simplified)
        self.current_orientation = 2 * math.atan2(orientation_q.z, orientation_q.w)

    def laser_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        # Check for obstacles in front of the robot
        front_scan = msg.ranges[len(msg.ranges)//2 - 10:len(msg.ranges)//2 + 10]
        self.min_front_distance = min(front_scan) if front_scan else float('inf')

    def navigate(self):
        """Main navigation behavior"""
        # Calculate distance to target
        dist_to_target = math.sqrt(
            (self.target_position.x - self.current_position.x)**2 +
            (self.target_position.y - self.current_position.y)**2
        )

        # Check if we reached the target
        if dist_to_target < self.min_distance_to_target:
            self.stop_robot()
            self.get_logger().info('Target reached!')
            return

        # Check for obstacles
        if self.min_front_distance < self.safety_distance:
            self.avoid_obstacle()
            return

        # Navigate toward target
        self.move_towards_target()

    def move_towards_target(self):
        """Move robot towards the target position"""
        cmd_vel = Twist()

        # Calculate angle to target
        angle_to_target = math.atan2(
            self.target_position.y - self.current_position.y,
            self.target_position.x - self.current_position.x
        )

        # Calculate angle difference
        angle_diff = angle_to_target - self.current_orientation

        # Normalize angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Set angular velocity proportional to angle error
        cmd_vel.angular.z = self.angular_speed * angle_diff

        # Set linear velocity based on distance (slow down when close)
        cmd_vel.linear.x = min(self.linear_speed, dist_to_target * 0.5)

        self.cmd_publisher.publish(cmd_vel)

    def avoid_obstacle(self):
        """Implement obstacle avoidance behavior"""
        cmd_vel = Twist()
        # Turn away from obstacles
        cmd_vel.angular.z = self.angular_speed
        cmd_vel.linear.x = 0.0  # Stop forward movement
        self.cmd_publisher.publish(cmd_vel)

    def stop_robot(self):
        """Stop the robot"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_publisher.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    agent = NavigationAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Navigation Agent stopped by user')
    finally:
        agent.stop_robot()
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Agent Patterns

### 1. State Machine Agent

For complex behaviors, implement agents using state machines:

```python
from enum import Enum

class RobotState(Enum):
    IDLE = 1
    NAVIGATING = 2
    AVOIDING_OBSTACLE = 3
    REACHED_GOAL = 4
    EMERGENCY_STOP = 5

class StateMachineAgent(Node):
    def __init__(self):
        super().__init__('state_machine_agent')
        self.current_state = RobotState.IDLE
        self.state_timer = self.create_timer(0.1, self.state_machine)

    def state_machine(self):
        """State machine implementation"""
        if self.current_state == RobotState.IDLE:
            self.handle_idle()
        elif self.current_state == RobotState.NAVIGATING:
            self.handle_navigating()
        elif self.current_state == RobotState.AVOIDING_OBSTACLE:
            self.handle_avoiding_obstacle()
        elif self.current_state == RobotState.REACHED_GOAL:
            self.handle_reached_goal()
        elif self.current_state == RobotState.EMERGENCY_STOP:
            self.handle_emergency_stop()

    def handle_idle(self):
        """Handle IDLE state"""
        # Check if new goal is available
        if self.new_goal_available():
            self.current_state = RobotState.NAVIGATING

    def handle_navigating(self):
        """Handle NAVIGATING state"""
        # Check for obstacles
        if self.obstacle_detected():
            self.current_state = RobotState.AVOIDING_OBSTACLE
        # Check if goal reached
        elif self.goal_reached():
            self.current_state = RobotState.REACHED_GOAL

    def handle_avoiding_obstacle(self):
        """Handle AVOIDING_OBSTACLE state"""
        # Check if obstacle cleared
        if not self.obstacle_detected():
            self.current_state = RobotState.NAVIGATING

    def handle_reached_goal(self):
        """Handle REACHED_GOAL state"""
        # Wait for new goal or timeout
        if self.timeout_reached() or self.new_goal_available():
            self.current_state = RobotState.IDLE

    def handle_emergency_stop(self):
        """Handle EMERGENCY_STOP state"""
        # Only return to other states when emergency is cleared
        if self.emergency_cleared():
            self.current_state = RobotState.IDLE
```

### 2. Behavior Tree Agent

For more complex decision-making, implement behavior trees:

```python
class BehaviorNode:
    def __init__(self):
        self.status = 'IDLE'

    def tick(self):
        """Execute the behavior and return status"""
        pass

class SequenceNode(BehaviorNode):
    def __init__(self, children):
        super().__init__()
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status != 'SUCCESS':
                return status
        return 'SUCCESS'

class SelectorNode(BehaviorNode):
    def __init__(self, children):
        super().__init__()
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status == 'SUCCESS':
                return status
        return 'FAILURE'
```

## Safety-First Agent Design

When designing Python agents for robotic systems, safety must be the primary concern:

### 1. Fail-Safe Behaviors
```python
def emergency_stop(self):
    """Always available emergency stop function"""
    cmd_vel = Twist()
    cmd_vel.linear.x = 0.0
    cmd_vel.angular.z = 0.0
    self.cmd_publisher.publish(cmd_vel)
```

### 2. Timeout Handling
```python
def __init__(self):
    # ... other initialization
    self.last_sensor_update = self.get_clock().now()
    self.sensor_timeout = rclpy.Duration(seconds=1.0)

def sensor_callback(self, msg):
    self.last_sensor_update = self.get_clock().now()
    # Process sensor data

def check_sensor_timeout(self):
    """Check if sensor data is too old"""
    current_time = self.get_clock().now()
    if (current_time - self.last_sensor_update) > self.sensor_timeout:
        self.get_logger().error('Sensor timeout detected - activating emergency stop')
        self.emergency_stop()
        return True
    return False
```

### 3. Validation and Error Handling
```python
def validate_command(self, cmd_vel):
    """Validate velocity commands before sending"""
    # Check limits
    max_linear = 1.0  # m/s
    max_angular = 1.0  # rad/s

    cmd_vel.linear.x = max(min(cmd_vel.linear.x, max_linear), -max_linear)
    cmd_vel.angular.z = max(min(cmd_vel.angular.z, max_angular), -max_angular)

    return cmd_vel
```

## Hands-on Exercise: Create a Python Agent

### Objective
Create a Python agent that monitors sensor data and implements a simple reactive behavior.

### Steps
1. Create a new Python agent node
2. Subscribe to sensor data (e.g., laser scan or camera)
3. Implement a reactive behavior based on sensor input
4. Add safety checks and emergency stop functionality
5. Test the agent in simulation

### Expected Outcome
- Agent responds appropriately to sensor input
- Safety mechanisms prevent dangerous behaviors
- Agent follows modular design principles
- Code follows Python best practices

## Summary

In this chapter, we've explored how to create intelligent Python agents that control ROS 2 systems:
- Basic agent structure and patterns
- State machine and behavior tree implementations
- Safety-first design principles
- Practical examples of navigation and control agents

Python agents provide a powerful way to implement high-level control logic and AI-based behaviors in robotic systems. By following safety-first design principles and modular architecture, we can create robust and reliable robotic applications.

## Next Steps

Continue to the next chapter: [URDF for Humanoid Robots](./urdf-humanoid-robots.md) to learn how to create robot descriptions for humanoid platforms.