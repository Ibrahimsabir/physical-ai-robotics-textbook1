---
title: Multi-Step Robotic Actions
sidebar_label: Multi-Step Actions
description: Sequencing complex robotic behaviors and managing multi-step action execution
keywords: [multi-step, action sequencing, behavior trees, robotics, planning, execution]
---

# Multi-Step Robotic Actions

## Introduction

Complex robotic tasks require the coordination of multiple primitive actions into coherent sequences. Multi-step action execution is fundamental to creating autonomous robots that can perform sophisticated tasks like assembly, cleaning, or assistance. This chapter explores how to design, implement, and execute complex behavioral sequences in robotic systems.

## Learning Objectives

By the end of this chapter, you will be able to:
- Design multi-step action sequences for complex robotic tasks
- Implement behavior trees for complex task orchestration
- Manage action execution with proper error handling and recovery
- Create flexible and reusable action compositions
- Implement state management for long-running tasks
- Design safe and reliable multi-step action systems

## Multi-Step Action Fundamentals

### Sequential vs Concurrent Actions

Robotic tasks can be decomposed into:
- **Sequential Actions**: Each step must complete before the next begins
- **Concurrent Actions**: Multiple actions execute in parallel
- **Conditional Actions**: Execution depends on specific conditions
- **Looping Actions**: Repetitive execution until a condition is met

### Action Primitives vs Composite Actions

**Action Primitives**: Basic building blocks (move, grasp, detect)
**Composite Actions**: Higher-level behaviors composed of primitives

```python
# Example of composite action
def deliver_item_to_person(person_name, item_name):
    """
    Composite action: Deliver item to person
    Composed of: detect_person, approach_person, grasp_item, move_to_person, place_item
    """
    # 1. Detect person
    person_location = detect_person(person_name)
    if not person_location:
        raise Exception(f"Could not find {person_name}")

    # 2. Approach person
    approach_location(person_location)

    # 3. Grasp item
    grasp_item(item_name)

    # 4. Move to person
    move_to_location(person_location)

    # 5. Place item
    place_item(item_name)
```

## Behavior Trees for Action Sequencing

Behavior trees provide a powerful framework for structuring complex robotic behaviors. They offer hierarchical composition, clear execution flow, and robust error handling.

### Behavior Tree Components

```python
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Optional

class NodeStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"

class BehaviorTreeNode(ABC):
    def __init__(self, name: str = ""):
        self.name = name
        self.status = NodeStatus.RUNNING

    @abstractmethod
    def tick(self) -> NodeStatus:
        """Execute one cycle of the behavior"""
        pass

    def reset(self):
        """Reset node state"""
        self.status = NodeStatus.RUNNING

class ActionNode(BehaviorTreeNode):
    """Leaf node that performs a specific action"""
    def __init__(self, name: str, action_func):
        super().__init__(name)
        self.action_func = action_func

    def tick(self) -> NodeStatus:
        try:
            result = self.action_func()
            return NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        except Exception as e:
            print(f"Action {self.name} failed: {e}")
            return NodeStatus.FAILURE

class ConditionNode(BehaviorTreeNode):
    """Leaf node that checks a condition"""
    def __init__(self, name: str, condition_func):
        super().__init__(name)
        self.condition_func = condition_func

    def tick(self) -> NodeStatus:
        try:
            result = self.condition_func()
            return NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        except Exception as e:
            print(f"Condition {self.name} failed: {e}")
            return NodeStatus.FAILURE

class SequenceNode(BehaviorTreeNode):
    """Runs children in sequence; fails if any child fails"""
    def __init__(self, name: str, children: List[BehaviorTreeNode]):
        super().__init__(name)
        self.children = children
        self.current_child_idx = 0

    def tick(self) -> NodeStatus:
        while self.current_child_idx < len(self.children):
            child_status = self.children[self.current_child_idx].tick()

            if child_status == NodeStatus.FAILURE:
                # Reset for next execution
                self.current_child_idx = 0
                return NodeStatus.FAILURE
            elif child_status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            else:  # SUCCESS
                self.current_child_idx += 1

        # All children succeeded
        self.current_child_idx = 0
        return NodeStatus.SUCCESS

    def reset(self):
        super().reset()
        self.current_child_idx = 0
        for child in self.children:
            child.reset()

class SelectorNode(BehaviorTreeNode):
    """Runs children in sequence; succeeds if any child succeeds"""
    def __init__(self, name: str, children: List[BehaviorTreeNode]):
        super().__init__(name)
        self.children = children
        self.current_child_idx = 0

    def tick(self) -> NodeStatus:
        while self.current_child_idx < len(self.children):
            child_status = self.children[self.current_child_idx].tick()

            if child_status == NodeStatus.SUCCESS:
                # Reset for next execution
                self.current_child_idx = 0
                return NodeStatus.SUCCESS
            elif child_status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            else:  # FAILURE
                self.current_child_idx += 1

        # All children failed
        self.current_child_idx = 0
        return NodeStatus.FAILURE

    def reset(self):
        super().reset()
        self.current_child_idx = 0
        for child in self.children:
            child.reset()

class DecoratorNode(BehaviorTreeNode):
    """Modifies behavior of a single child node"""
    def __init__(self, name: str, child: BehaviorTreeNode):
        super().__init__(name)
        self.child = child

    def tick(self) -> NodeStatus:
        return self.child.tick()

    def reset(self):
        super().reset()
        self.child.reset()
```

### Inverter Decorator
```python
class InverterNode(DecoratorNode):
    """Inverts the result of the child node"""
    def tick(self) -> NodeStatus:
        child_status = self.child.tick()

        if child_status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
        elif child_status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS
        else:
            return NodeStatus.RUNNING
```

### Retry Until Success Decorator
```python
class RetryUntilSuccessNode(DecoratorNode):
    """Keeps retrying the child until it succeeds"""
    def __init__(self, name: str, child: BehaviorTreeNode, max_attempts: int = 10):
        super().__init__(name, child)
        self.max_attempts = max_attempts
        self.attempts = 0

    def tick(self) -> NodeStatus:
        while self.attempts < self.max_attempts:
            child_status = self.child.tick()

            if child_status == NodeStatus.SUCCESS:
                self.attempts = 0  # Reset for next execution
                return NodeStatus.SUCCESS
            elif child_status == NodeStatus.FAILURE:
                self.attempts += 1
                self.child.reset()  # Reset child for next attempt
                if self.attempts >= self.max_attempts:
                    self.attempts = 0
                    return NodeStatus.FAILURE
            else:  # RUNNING
                return NodeStatus.RUNNING

        return NodeStatus.FAILURE

    def reset(self):
        super().reset()
        self.attempts = 0
        self.child.reset()
```

## Complex Task Example: Fetch and Carry

Let's build a complex behavior tree for a fetch-and-carry task:

```python
class FetchAndCarryBehaviorTree:
    def __init__(self, robot_interface):
        self.robot = robot_interface

        # Build the behavior tree
        self.root = self.build_tree()

    def build_tree(self) -> BehaviorTreeNode:
        """Build the complete behavior tree for fetch and carry task"""
        return SequenceNode("FetchAndCarry", [
            # 1. Detect target object
            self.create_detect_object_node(),

            # 2. Navigate to object
            self.create_navigate_to_object_node(),

            # 3. Grasp object
            self.create_grasp_object_node(),

            # 4. Navigate to destination
            self.create_navigate_to_destination_node(),

            # 5. Place object
            self.create_place_object_node()
        ])

    def create_detect_object_node(self) -> BehaviorTreeNode:
        """Create node to detect the target object"""
        def detect_action():
            target_obj = self.robot.get_task_parameters().get('target_object')
            detected = self.robot.detect_object(target_obj)
            if detected:
                self.robot.set_current_object(detected)
                return True
            return False

        return RetryUntilSuccessNode(
            "DetectObject",
            ActionNode("DetectObject", detect_action),
            max_attempts=5
        )

    def create_navigate_to_object_node(self) -> BehaviorTreeNode:
        """Create node to navigate to the detected object"""
        def navigate_action():
            current_obj = self.robot.get_current_object()
            if not current_obj:
                return False

            return self.robot.navigate_to(current_obj['position'])

        return ActionNode("NavigateToObject", navigate_action)

    def create_grasp_object_node(self) -> BehaviorTreeNode:
        """Create node to grasp the object"""
        def grasp_action():
            current_obj = self.robot.get_current_object()
            if not current_obj:
                return False

            # Check if object is graspable
            if not current_obj.get('graspable', False):
                return False

            return self.robot.grasp_object(current_obj['name'])

        return RetryUntilSuccessNode(
            "GraspObject",
            ActionNode("GraspObject", grasp_action),
            max_attempts=3
        )

    def create_navigate_to_destination_node(self) -> BehaviorTreeNode:
        """Create node to navigate to destination"""
        def navigate_action():
            destination = self.robot.get_task_parameters().get('destination')
            if not destination:
                return False

            return self.robot.navigate_to(destination)

        return ActionNode("NavigateToDestination", navigate_action)

    def create_place_object_node(self) -> BehaviorTreeNode:
        """Create node to place the object"""
        def place_action():
            current_obj = self.robot.get_current_object()
            if not current_obj:
                return False

            return self.robot.place_object(current_obj['name'])

        return ActionNode("PlaceObject", place_action)

    def tick(self) -> NodeStatus:
        """Execute one cycle of the behavior tree"""
        return self.root.tick()

    def reset(self):
        """Reset the entire tree"""
        self.root.reset()
```

## State Management for Long-Running Tasks

Complex multi-step tasks often need to maintain state across multiple execution cycles:

```python
class TaskStateManager:
    def __init__(self):
        self.task_state = {}
        self.task_history = []
        self.max_history = 50  # Limit history size

    def set_state(self, key: str, value):
        """Set a state variable"""
        self.task_state[key] = value

    def get_state(self, key: str, default=None):
        """Get a state variable"""
        return self.task_state.get(key, default)

    def update_history(self, action: str, result: str, timestamp: float):
        """Update execution history"""
        entry = {
            'action': action,
            'result': result,
            'timestamp': timestamp,
            'state_snapshot': self.task_state.copy()
        }

        self.task_history.append(entry)

        # Limit history size
        if len(self.task_history) > self.max_history:
            self.task_history.pop(0)

    def get_current_context(self) -> dict:
        """Get current context for decision making"""
        return {
            'current_state': self.task_state,
            'recent_history': self.task_history[-5:],  # Last 5 actions
            'execution_time': len(self.task_history)
        }

class StatefulActionNode(ActionNode):
    """Action node that maintains state"""
    def __init__(self, name: str, action_func, state_manager: TaskStateManager):
        super().__init__(name, action_func)
        self.state_manager = state_manager
        self.start_time = None

    def tick(self) -> NodeStatus:
        if self.start_time is None:
            self.start_time = time.time()
            self.state_manager.set_state(f"{self.name}_start_time", self.start_time)

        try:
            result = self.action_func(self.state_manager)

            # Update history
            elapsed = time.time() - self.start_time
            self.state_manager.update_history(
                self.name,
                "SUCCESS" if result else "FAILURE",
                time.time()
            )

            return NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        except Exception as e:
            self.state_manager.update_history(
                self.name,
                f"ERROR: {str(e)}",
                time.time()
            )
            print(f"Action {self.name} failed: {e}")
            return NodeStatus.FAILURE
```

## Error Handling and Recovery

Robust multi-step action systems must handle failures gracefully:

```python
class ResilientSequenceNode(SequenceNode):
    """Sequence node with error handling and recovery"""
    def __init__(self, name: str, children: List[BehaviorTreeNode], recovery_plan: Optional[List[BehaviorTreeNode]] = None):
        super().__init__(name, children)
        self.recovery_plan = recovery_plan or []
        self.failed_child_idx = -1
        self.executing_recovery = False

    def tick(self) -> NodeStatus:
        if self.executing_recovery:
            return self.execute_recovery()

        while self.current_child_idx < len(self.children):
            child_status = self.children[self.current_child_idx].tick()

            if child_status == NodeStatus.FAILURE:
                self.failed_child_idx = self.current_child_idx
                if self.recovery_plan:
                    self.executing_recovery = True
                    self.current_child_idx = 0  # Reset for recovery plan
                    return self.execute_recovery()
                else:
                    # No recovery plan, fail the sequence
                    self.current_child_idx = 0
                    return NodeStatus.FAILURE
            elif child_status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            else:  # SUCCESS
                self.current_child_idx += 1

        # All children succeeded
        self.current_child_idx = 0
        return NodeStatus.SUCCESS

    def execute_recovery(self) -> NodeStatus:
        """Execute recovery plan"""
        while self.current_child_idx < len(self.recovery_plan):
            recovery_status = self.recovery_plan[self.current_child_idx].tick()

            if recovery_status == NodeStatus.FAILURE:
                # Recovery failed, reset everything
                self.reset()
                return NodeStatus.FAILURE
            elif recovery_status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            else:  # SUCCESS
                self.current_child_idx += 1

        # Recovery succeeded, attempt to continue original task
        self.executing_recovery = False
        self.current_child_idx = self.failed_child_idx
        # Reset the failed child to retry it
        self.children[self.current_child_idx].reset()

        return self.tick()  # Continue with original sequence

    def reset(self):
        super().reset()
        self.failed_child_idx = -1
        self.executing_recovery = False
        for recovery_node in self.recovery_plan:
            recovery_node.reset()
```

## Concurrency in Multi-Step Actions

Some actions can run concurrently to improve efficiency:

```python
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class ConcurrentActionNode(BehaviorTreeNode):
    """Node that executes multiple actions concurrently"""
    def __init__(self, name: str, actions: List[callable], concurrency_policy: str = "all_success"):
        super().__init__(name)
        self.actions = actions
        self.concurrency_policy = concurrency_policy  # "all_success", "any_success", "majority"
        self.results = [None] * len(actions)
        self.completed = [False] * len(actions)
        self.executor = ThreadPoolExecutor(max_workers=len(actions))

    def tick(self) -> NodeStatus:
        # Submit all actions that haven't been started
        futures = []
        for i, (action, completed) in enumerate(zip(self.actions, self.completed)):
            if not completed:
                future = self.executor.submit(self._execute_action, i, action)
                futures.append((i, future))

        # Check for completed actions
        all_completed = True
        for i, future in futures:
            if future.done():
                self.results[i], self.completed[i] = future.result()
            else:
                all_completed = False

        if all_completed:
            # Evaluate results based on policy
            success_count = sum(1 for result in self.results if result is True)

            if self.concurrency_policy == "all_success":
                return NodeStatus.SUCCESS if success_count == len(self.actions) else NodeStatus.FAILURE
            elif self.concurrency_policy == "any_success":
                return NodeStatus.SUCCESS if success_count > 0 else NodeStatus.FAILURE
            elif self.concurrency_policy == "majority":
                return NodeStatus.SUCCESS if success_count > len(self.actions) // 2 else NodeStatus.FAILURE

        return NodeStatus.RUNNING

    def _execute_action(self, idx: int, action: callable):
        """Execute individual action"""
        try:
            result = action()
            return result, True
        except Exception as e:
            print(f"Action {idx} failed: {e}")
            return False, True

    def reset(self):
        super().reset()
        self.results = [None] * len(self.actions)
        self.completed = [False] * len(self.actions)
```

## ROS 2 Integration for Multi-Step Actions

Integrating multi-step actions with ROS 2 requires proper action server implementation:

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time

from custom_action_interfaces.action import MultiStepTask  # Custom action definition

class MultiStepActionServer(Node):
    def __init__(self):
        super().__init__('multi_step_action_server')

        # Create action server with reentrant callback group for concurrency
        self._action_server = ActionServer(
            self,
            MultiStepTask,
            'execute_multi_step_task',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            cancel_callback=self.cancel_callback
        )

        # Task execution components
        self.behavior_tree = None
        self.task_active = False
        self.cancel_requested = False

        self.get_logger().info("Multi-step action server initialized")

    def execute_callback(self, goal_handle):
        """Execute the multi-step task"""
        self.get_logger().info(f'Executing multi-step task: {goal_handle.request.task_description}')

        # Build behavior tree based on goal
        self.behavior_tree = self.build_behavior_tree_from_goal(goal_handle.request)
        self.task_active = True
        self.cancel_requested = False

        # Execute the task
        feedback_msg = MultiStepTask.Feedback()
        result_msg = MultiStepTask.Result()

        while self.task_active and not self.cancel_requested:
            try:
                # Execute one tick of the behavior tree
                status = self.behavior_tree.tick()

                # Update feedback
                feedback_msg.current_step = self.get_current_step_info()
                feedback_msg.progress_percentage = self.calculate_progress()

                goal_handle.publish_feedback(feedback_msg)

                if status == NodeStatus.SUCCESS:
                    result_msg.success = True
                    result_msg.message = "Task completed successfully"
                    goal_handle.succeed()
                    break
                elif status == NodeStatus.FAILURE:
                    result_msg.success = False
                    result_msg.message = "Task failed during execution"
                    goal_handle.abort()
                    break
                else:  # RUNNING
                    # Small delay to prevent busy waiting
                    time.sleep(0.1)

            except Exception as e:
                self.get_logger().error(f'Error during task execution: {e}')
                result_msg.success = False
                result_msg.message = f"Execution error: {str(e)}"
                goal_handle.abort()
                break

        self.task_active = False
        return result_msg

    def cancel_callback(self, goal_handle):
        """Handle cancellation request"""
        self.get_logger().info('Received cancel request')
        self.cancel_requested = True
        return CancelResponse.ACCEPT

    def build_behavior_tree_from_goal(self, goal) -> BehaviorTreeNode:
        """Build behavior tree from goal specification"""
        # Parse the goal and build appropriate behavior tree
        # This is a simplified example
        if goal.task_type == "fetch_and_carry":
            return FetchAndCarryBehaviorTree(self).root
        elif goal.task_type == "room_cleaning":
            return RoomCleaningBehaviorTree(self).root
        else:
            # Default sequence for custom tasks
            return self.build_custom_sequence(goal.steps)

    def get_current_step_info(self) -> str:
        """Get information about current step"""
        # In a real implementation, this would query the behavior tree
        return "Executing task..."

    def calculate_progress(self) -> float:
        """Calculate task progress percentage"""
        # In a real implementation, this would track actual progress
        return 50.0  # Placeholder

def main(args=None):
    rclpy.init(args=args)

    action_server = MultiStepActionServer()

    # Use multi-threaded executor to handle concurrent callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(action_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        action_server.get_logger().info("Shutting down")
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Simulation-Based Lab Exercise: Multi-Step Task Execution

### Objective
Create a complete multi-step action execution system that can perform complex tasks like fetching an object and delivering it to a specified location.

### Prerequisites
- ROS 2 installation
- Gazebo simulation environment
- Basic understanding of action servers

### Steps

#### 1. Create Custom Action Definition
First, create a custom action definition for multi-step tasks:

```# MultiStepTask.action
# Define the task to be executed
string task_description
string task_type
TaskStep[] steps

---
# Result of the task execution
bool success
string message
float32 completion_percentage

---
# Feedback during execution
string current_step
float32 progress_percentage
bool is_executing
```

```# TaskStep.action
string action_name
KeyValue[] parameters
bool is_optional

# KeyValue.msg
string key
string value
```

#### 2. Implement the Multi-Step Executor
Create the complete multi-step action executor with behavior trees:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from custom_action_interfaces.action import MultiStepTask

class AdvancedMultiStepExecutor(Node):
    def __init__(self):
        super().__init__('advanced_multi_step_executor')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Action server
        self._action_server = ActionServer(
            self,
            MultiStepTask,
            'advanced_multi_step_task',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup()
        )

        # Robot state
        self.current_pose = Pose()
        self.scan_data = None
        self.task_queue = []
        self.active_task = None

        self.get_logger().info("Advanced Multi-Step Executor initialized")

    def scan_callback(self, msg):
        """Update scan data for navigation"""
        self.scan_data = msg

    def execute_callback(self, goal_handle):
        """Execute complex multi-step task"""
        self.get_logger().info(f"Starting task: {goal_handle.request.task_description}")

        # Build and execute behavior tree
        behavior_tree = self.build_behavior_tree(goal_handle.request)

        feedback_msg = MultiStepTask.Feedback()
        result_msg = MultiStepTask.Result()

        # Execute the behavior tree
        success = self.execute_behavior_tree(behavior_tree, goal_handle, feedback_msg)

        if success:
            result_msg.success = True
            result_msg.message = "Task completed successfully"
            result_msg.completion_percentage = 100.0
            goal_handle.succeed()
        else:
            result_msg.success = False
            result_msg.message = "Task failed during execution"
            result_msg.completion_percentage = self.get_last_progress()
            goal_handle.abort()

        return result_msg

    def build_behavior_tree(self, goal) -> BehaviorTreeNode:
        """Build behavior tree from goal specification"""
        # For this example, we'll create a fetch-and-deliver task
        if goal.task_type == "fetch_and_deliver":
            return self.create_fetch_and_deliver_tree(goal)
        elif goal.task_type == "clean_room":
            return self.create_clean_room_tree(goal)
        else:
            return self.create_generic_task_tree(goal)

    def create_fetch_and_deliver_tree(self, goal) -> BehaviorTreeNode:
        """Create behavior tree for fetch and deliver task"""
        # This would be a complex tree with:
        # 1. Navigate to pickup location
        # 2. Detect and grasp object
        # 3. Navigate to delivery location
        # 4. Place object
        # 5. Return to home position

        # For simplicity, returning a sequence of simulated actions
        return SequenceNode("FetchAndDeliver", [
            ActionNode("NavigateToPickup", lambda: self.simulate_navigation(1.0, 1.0)),
            ActionNode("DetectObject", lambda: self.simulate_object_detection()),
            ActionNode("GraspObject", lambda: self.simulate_grasping()),
            ActionNode("NavigateToDelivery", lambda: self.simulate_navigation(3.0, 2.0)),
            ActionNode("PlaceObject", lambda: self.simulate_placement())
        ])

    def execute_behavior_tree(self, tree: BehaviorTreeNode, goal_handle, feedback_msg) -> bool:
        """Execute behavior tree with feedback and cancellation"""
        self.active_task = tree

        while rclpy.ok():
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                tree.reset()
                return False

            # Execute one tick
            status = tree.tick()

            # Update feedback
            feedback_msg.current_step = self.get_current_step_description(tree)
            feedback_msg.progress_percentage = self.calculate_tree_progress(tree)
            feedback_msg.is_executing = True

            goal_handle.publish_feedback(feedback_msg)

            if status == NodeStatus.SUCCESS:
                return True
            elif status == NodeStatus.FAILURE:
                return False
            else:  # RUNNING
                # Small delay to prevent busy waiting
                self.get_clock().sleep_for(Duration(seconds=0.1))

        return False

    def simulate_navigation(self, x: float, y: float) -> bool:
        """Simulate navigation to coordinates"""
        self.get_logger().info(f"Navigating to ({x}, {y})")

        # Simulate navigation
        cmd = Twist()
        cmd.linear.x = 0.2  # Move forward
        self.cmd_vel_pub.publish(cmd)

        # Simulate reaching destination
        self.get_clock().sleep_for(Duration(seconds=2.0))

        # Update current pose (simulated)
        self.current_pose.position.x = x
        self.current_pose.position.y = y

        return True

    def simulate_object_detection(self) -> bool:
        """Simulate object detection"""
        self.get_logger().info("Detecting object...")
        self.get_clock().sleep_for(Duration(seconds=1.0))
        return True

    def simulate_grasping(self) -> bool:
        """Simulate object grasping"""
        self.get_logger().info("Grasping object...")
        self.get_clock().sleep_for(Duration(seconds=1.0))
        return True

    def simulate_placement(self) -> bool:
        """Simulate object placement"""
        self.get_logger().info("Placing object...")
        self.get_clock().sleep_for(Duration(seconds=1.0))
        return True

    def get_current_step_description(self, tree) -> str:
        """Get description of current step"""
        # In a real implementation, this would introspect the tree
        return "Executing multi-step task"

    def calculate_tree_progress(self, tree) -> float:
        """Calculate progress of tree execution"""
        # In a real implementation, this would calculate actual progress
        return 50.0  # Placeholder

def main(args=None):
    rclpy.init(args=args)

    executor_node = AdvancedMultiStepExecutor()

    try:
        rclpy.spin(executor_node)
    except KeyboardInterrupt:
        executor_node.get_logger().info("Shutting down")
    finally:
        executor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 3. Test the System
Test the multi-step action execution system with various complex tasks:
- Fetch and deliver objects
- Clean a room by navigating to multiple locations
- Perform inspection rounds

#### 4. Validate Safety Features
- Test error recovery mechanisms
- Verify proper cancellation handling
- Check that failed actions don't compromise robot safety

### Expected Outcome
- Complete multi-step action execution system
- Behavior tree implementation for complex task orchestration
- Proper state management for long-running tasks
- Error handling and recovery mechanisms
- Integration with ROS 2 action server framework

## Safety-First Design in Multi-Step Actions

### Core Safety Principles

When designing multi-step robotic actions, safety must be the primary consideration at every level:

#### 1. Action-Level Safety
- **Precondition Checks**: Verify all conditions are met before executing any action
- **Postcondition Validation**: Confirm expected outcomes after action completion
- **Safe Transitions**: Ensure smooth transitions between consecutive actions
- **Emergency Stops**: Implement immediate stop capabilities for any action

#### 2. Sequence-Level Safety
- **Dependency Validation**: Verify that each step's requirements are met
- **Rollback Procedures**: Define safe states to return to if a step fails
- **Timeout Protection**: Prevent infinite execution of any single action
- **State Monitoring**: Continuously monitor robot and environment state

#### 3. System-Level Safety
- **Fail-Safe Defaults**: Ensure the system moves to a safe state upon any failure
- **Human Override**: Provide mechanisms for human intervention
- **Risk Assessment**: Evaluate potential hazards before task execution
- **Safe Abort Procedures**: Define safe ways to terminate any ongoing task

### Safety Implementation Example

```python
class SafeMultiStepExecutor:
    def __init__(self):
        self.safety_monitor = SafetyMonitor()
        self.emergency_stop = EmergencyStopSystem()
        self.state_validator = StateValidator()

    def execute_safe_sequence(self, behavior_tree: BehaviorTreeNode) -> bool:
        """Execute sequence with safety monitoring"""
        # Validate initial state
        if not self.state_validator.validate_initial_state():
            self.safety_monitor.log_violation("Initial state validation failed")
            return False

        # Start safety monitoring
        self.safety_monitor.start_monitoring()

        try:
            while not self.emergency_stop.is_triggered():
                # Check safety conditions before each tick
                if not self.safety_monitor.are_conditions_safe():
                    self.emergency_stop.trigger()
                    return False

                status = behavior_tree.tick()

                if status == NodeStatus.FAILURE:
                    # Attempt safe recovery
                    if self.attempt_safe_recovery(behavior_tree):
                        continue
                    else:
                        self.emergency_stop.trigger()
                        return False
                elif status == NodeStatus.SUCCESS:
                    # Validate final state
                    if self.state_validator.validate_final_state():
                        return True
                    else:
                        self.safety_monitor.log_violation("Final state validation failed")
                        return False
                # If RUNNING, continue to next iteration

        except Exception as e:
            self.safety_monitor.log_violation(f"Unexpected error: {e}")
            self.emergency_stop.trigger()
            return False
        finally:
            self.safety_monitor.stop_monitoring()

    def attempt_safe_recovery(self, tree: BehaviorTreeNode) -> bool:
        """Attempt to recover safely from a failure"""
        try:
            # Move to a safe intermediate state
            self.move_to_safe_state()

            # Reset the failing node
            tree.reset()

            # Resume execution
            return True
        except:
            return False

    def move_to_safe_state(self):
        """Move robot to a predefined safe state"""
        # Example: stop all motion, retract manipulator, etc.
        self.stop_all_motion()
        self.retract_manipulator_if_extended()

    def stop_all_motion(self):
        """Emergency stop all robot motion"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def retract_manipulator_if_extended(self):
        """Retract manipulator to safe position if it's extended"""
        # Implementation would move manipulator to safe configuration
        pass
```

### Risk Assessment Framework

#### Hazard Identification
- **Kinematic Hazards**: Unintended collisions due to motion
- **Dynamic Hazards**: Forces exceeding safe limits
- **Environmental Hazards**: Changes in surroundings during execution
- **System Hazards**: Component failures during operation

#### Risk Mitigation Strategies
1. **Redundancy**: Multiple sensors for critical safety functions
2. **Diversity**: Different approaches for critical safety checks
3. **Isolation**: Separate safety systems from main control
4. **Testing**: Extensive testing in simulation before deployment

### Safety-First Best Practices

#### 1. Modularity
- Break complex tasks into reusable components
- Create libraries of common action primitives
- Use composition to build complex behaviors

#### 2. Safety (Enhanced)
- **Zero-Trust Validation**: Validate everything, even internal communications
- **Defense in Depth**: Multiple layers of safety checks
- **Graceful Degradation**: Maintain basic safety even when components fail
- **Fail-Safe by Default**: System assumes unsafe state until proven otherwise

#### 3. Monitoring
- Track progress and performance metrics
- Log all action executions for debugging
- Provide real-time feedback to operators
- Monitor for safety parameter drifts

#### 4. Flexibility
- Design for varying environmental conditions
- Allow runtime parameter adjustment
- Support different robot configurations
- Enable dynamic safety threshold adjustment

## Summary

In this chapter, we've explored multi-step robotic actions:
- Behavior trees for complex task orchestration
- State management for long-running tasks
- Error handling and recovery mechanisms
- Concurrency for improved efficiency
- ROS 2 integration with action servers
- Best practices for robust multi-step action systems

Multi-step action execution enables robots to perform complex, coordinated tasks that require multiple sequential and concurrent operations, forming the foundation for sophisticated autonomous behaviors.

## Next Steps

Return to the [Module 4 Overview](./index.md) to complete this module or review the concepts covered in this chapter.