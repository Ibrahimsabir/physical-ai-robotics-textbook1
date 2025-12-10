---
title: LLM Cognitive Planning
sidebar_label: LLM Cognitive Planning
description: Using Large Language Models for high-level task planning and cognitive control in robotic systems
keywords: [llm, large language model, cognitive planning, robotics, ai, task planning, gpt, claude]
---

# LLM Cognitive Planning

## Introduction

Large Language Models (LLMs) have emerged as powerful tools for cognitive planning in robotic systems. These models excel at understanding natural language commands, decomposing complex tasks into executable steps, and reasoning about the world in ways that enable sophisticated robotic behaviors. In this chapter, we'll explore how to integrate LLMs with robotic systems for high-level cognitive planning and task orchestration.

## Learning Objectives

By the end of this chapter, you will be able to:
- Integrate LLMs with robotic systems for cognitive planning
- Decompose complex tasks into executable robotic actions
- Implement safety checks for LLM-generated plans
- Design context-aware planning systems
- Handle ambiguous or impossible commands gracefully
- Create robust human-LLM-robot interaction systems

## LLMs in Robotics Context

### Advantages of LLMs for Cognitive Planning
- **Natural Language Understanding**: Interpret complex natural language commands
- **World Knowledge**: Leverage extensive world knowledge for planning
- **Reasoning Capabilities**: Perform multi-step reasoning about actions
- **Flexibility**: Handle novel or underspecified commands
- **Abstraction**: Bridge high-level goals with low-level actions

### Challenges in Robotics
- **Reliability**: LLMs can produce hallucinations or incorrect plans
- **Latency**: Cloud-based models may introduce delays
- **Safety**: Ensuring plans are safe before execution
- **Grounding**: Connecting abstract plans to concrete robot capabilities
- **Real-time Constraints**: Meeting timing requirements for interactive systems

## Setting Up LLM Integration

### Popular LLM Options for Robotics
1. **OpenAI GPT Models**: Well-documented, strong reasoning
2. **Anthropic Claude**: Emphasis on safety and helpfulness
3. **Open Source Models**: Hugging Face ecosystem (Llama, Mistral, etc.)
4. **Specialized Robotics Models**: Models fine-tuned for robotic tasks

### Basic LLM Integration
```python
import openai
import rospy
import json
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from typing import Dict, List, Any

class LLMCognitivePlanner:
    def __init__(self):
        rospy.init_node('llm_cognitive_planner')

        # Initialize LLM client (example with OpenAI)
        openai.api_key = rospy.get_param('~openai_api_key', 'your-key-here')

        # Publishers and subscribers
        self.plan_pub = rospy.Publisher('/high_level_plan', String, queue_size=10)
        self.task_sub = rospy.Subscriber('/natural_language_task', String, self.task_callback)

        # Robot state information
        self.robot_capabilities = self.get_robot_capabilities()
        self.environment_context = self.get_environment_context()

        rospy.loginfo("LLM Cognitive Planner initialized")

    def get_robot_capabilities(self) -> Dict[str, Any]:
        """Get robot's current capabilities and limitations"""
        return {
            'movement': {
                'types': ['translate', 'rotate'],
                'max_speed': 0.5,  # m/s
                'precision': 0.05  # m
            },
            'manipulation': {
                'reachable_area': {'min_x': -1.0, 'max_x': 1.0, 'min_y': -1.0, 'max_y': 1.0},
                'gripper': True,
                'max_load': 2.0  # kg
            },
            'sensors': ['camera', 'lidar', 'imu'],
            'current_pose': {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        }

    def get_environment_context(self) -> Dict[str, Any]:
        """Get current environment context"""
        return {
            'objects': self.get_visible_objects(),
            'safe_zones': ['starting_area', 'charging_station'],
            'restricted_areas': ['exit_door', 'lab_equipment'],
            'landmarks': ['desk', 'shelf', 'charger']
        }

    def get_visible_objects(self) -> List[Dict[str, Any]]:
        """Get objects currently in robot's field of view"""
        # In a real system, this would come from perception system
        return [
            {'name': 'red_block', 'type': 'block', 'position': {'x': 0.5, 'y': 0.3}},
            {'name': 'blue_box', 'type': 'container', 'position': {'x': -0.2, 'y': 0.8}},
            {'name': 'green_cylinder', 'type': 'cylinder', 'position': {'x': 0.1, 'y': -0.4}}
        ]

    def task_callback(self, msg: String):
        """Handle incoming natural language task"""
        try:
            rospy.loginfo(f"Received task: {msg.data}")

            # Generate plan using LLM
            plan = self.generate_plan(msg.data)

            # Validate and execute plan
            if self.validate_plan(plan):
                self.execute_plan(plan)
            else:
                rospy.logerr("Generated plan failed validation")
                self.request_clarification(msg.data)

        except Exception as e:
            rospy.logerr(f"Error processing task: {e}")
            self.handle_error(e)

    def generate_plan(self, natural_language_task: str) -> Dict[str, Any]:
        """Generate a plan using LLM"""
        system_prompt = f"""
        You are a helpful assistant that converts natural language robotic tasks into executable plans.
        The robot has the following capabilities: {json.dumps(self.robot_capabilities)}
        Current environment context: {json.dumps(self.environment_context)}

        Respond with a JSON object containing:
        {{
            "task_description": "Brief description of the original task",
            "plan_steps": [
                {{
                    "action": "action_name",
                    "parameters": {{"param_name": "param_value"}},
                    "description": "Human-readable description of the step"
                }}
            ],
            "estimated_duration": "Estimated time in seconds",
            "potential_risks": ["list", "of", "potential", "risks"]
        }}

        Available actions:
        - move_to(x, y): Move to position (meters)
        - rotate(theta): Rotate to angle (radians)
        - pick_up(object_name): Pick up an object
        - place(object_name, x, y): Place object at position
        - look_at(x, y): Turn camera toward position
        - wait(duration): Wait for duration in seconds
        - approach(object_name): Approach an object
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or gpt-4 for more complex tasks
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": natural_language_task}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=1000
            )

            # Extract and parse the plan
            plan_text = response.choices[0].message.content

            # Find JSON in response (in case LLM adds extra text)
            json_start = plan_text.find('{')
            json_end = plan_text.rfind('}') + 1
            json_str = plan_text[json_start:json_end]

            plan = json.loads(json_str)
            rospy.loginfo(f"Generated plan: {plan}")

            return plan

        except json.JSONDecodeError as e:
            rospy.logerr(f"Could not parse LLM response as JSON: {e}")
            return self.create_simple_plan(natural_language_task)
        except Exception as e:
            rospy.logerr(f"Error calling LLM: {e}")
            return self.create_simple_plan(natural_language_task)

    def create_simple_plan(self, task: str) -> Dict[str, Any]:
        """Create a simple plan when LLM fails"""
        # Simple fallback logic
        if 'move to' in task.lower():
            # Extract coordinates if possible
            return {
                "task_description": task,
                "plan_steps": [{"action": "move_to", "parameters": {"x": 1.0, "y": 1.0}}],
                "estimated_duration": 10.0,
                "potential_risks": ["navigation risk"]
            }

        return {
            "task_description": task,
            "plan_steps": [{"action": "unknown", "parameters": {}, "description": "Unable to parse task"}],
            "estimated_duration": 0.0,
            "potential_risks": ["task not understood"]
        }

    def validate_plan(self, plan: Dict[str, Any]) -> bool:
        """Validate the generated plan for safety and feasibility"""
        if not isinstance(plan, dict):
            rospy.logerr("Plan is not a dictionary")
            return False

        if 'plan_steps' not in plan or not isinstance(plan['plan_steps'], list):
            rospy.logerr("Plan does not contain valid steps")
            return False

        # Check each step for validity
        for step in plan['plan_steps']:
            if 'action' not in step:
                rospy.logerr(f"Step missing action: {step}")
                return False

            # Validate action parameters
            if not self.is_action_valid(step):
                rospy.logerr(f"Invalid action: {step}")
                return False

            # Check for safety
            if not self.is_action_safe(step):
                rospy.logerr(f"Unsafe action: {step}")
                return False

        return True

    def is_action_valid(self, step: Dict[str, Any]) -> bool:
        """Check if action is valid for robot"""
        action = step['action']
        params = step.get('parameters', {})

        # Validate based on robot capabilities
        if action == 'move_to':
            x = params.get('x', 0)
            y = params.get('y', 0)

            # Check if position is reachable
            if abs(x) > 5.0 or abs(y) > 5.0:  # Assuming 5m workspace
                return False

        elif action == 'pick_up':
            obj_name = params.get('object_name')
            if not obj_name:
                return False

            # Check if object exists and is reachable
            if not self.is_object_reachable(obj_name):
                return False

        return True

    def is_action_safe(self, step: Dict[str, Any]) -> bool:
        """Check if action is safe to execute"""
        action = step['action']
        params = step.get('parameters', {})

        # Safety checks
        if action == 'move_to':
            x = params.get('x', 0)
            y = params.get('y', 0)

            # Check if position is in restricted area
            if self.is_restricted_area(x, y):
                return False

        elif action == 'rotate':
            # Check for potential collisions during rotation
            pass

        return True

    def is_object_reachable(self, object_name: str) -> bool:
        """Check if object is currently reachable"""
        for obj in self.get_visible_objects():
            if obj['name'] == object_name:
                # Check if within reach
                current_pos = self.robot_capabilities['current_pose']
                obj_pos = obj['position']

                distance = ((current_pos['x'] - obj_pos['x'])**2 +
                           (current_pos['y'] - obj_pos['y'])**2)**0.5

                return distance < 1.0  # 1 meter reach

        return False

    def is_restricted_area(self, x: float, y: float) -> bool:
        """Check if coordinates are in a restricted area"""
        restricted_areas = self.environment_context.get('restricted_areas', [])
        # In a real system, this would check against a map
        # For now, return False as a placeholder
        return False

    def execute_plan(self, plan: Dict[str, Any]):
        """Execute the validated plan"""
        rospy.loginfo(f"Executing plan: {plan['task_description']}")

        for i, step in enumerate(plan['plan_steps']):
            rospy.loginfo(f"Executing step {i+1}: {step['description']}")

            success = self.execute_single_step(step)

            if not success:
                rospy.logerr(f"Step {i+1} failed, aborting plan")
                self.abort_plan(plan)
                return

            rospy.loginfo(f"Step {i+1} completed successfully")

        rospy.loginfo("Plan completed successfully")

    def execute_single_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single plan step"""
        action = step['action']
        params = step.get('parameters', {})

        # Create and publish command based on action
        command_msg = String()

        if action == 'move_to':
            # Create movement command
            command_msg.data = f"MOVE_TO:{params['x']},{params['y']}"
        elif action == 'rotate':
            command_msg.data = f"ROTATE:{params['theta']}"
        elif action == 'pick_up':
            command_msg.data = f"PICK_UP:{params['object_name']}"
        elif action == 'place':
            command_msg.data = f"PLACE:{params['object_name']},{params['x']},{params['y']}"
        elif action == 'look_at':
            command_msg.data = f"LOOK_AT:{params['x']},{params['y']}"
        elif action == 'wait':
            command_msg.data = f"WAIT:{params['duration']}"
        elif action == 'approach':
            command_msg.data = f"APPROACH:{params['object_name']}"
        else:
            rospy.logerr(f"Unknown action: {action}")
            return False

        # Publish command
        self.plan_pub.publish(command_msg)

        # Wait for execution to complete (simplified)
        rospy.sleep(params.get('duration', 1.0))

        return True

    def request_clarification(self, original_task: str):
        """Request clarification for ambiguous tasks"""
        rospy.logwarn(f"Task requires clarification: {original_task}")
        # In a real system, this might trigger audio feedback
        # self.audio_player.play("I need clarification on this task.")

    def handle_error(self, error: Exception):
        """Handle errors during plan generation or execution"""
        rospy.logerr(f"Cognitive planning error: {error}")

        error_msg = String()
        error_msg.data = f"PLANNING_ERROR:{str(error)}"
        self.plan_pub.publish(error_msg)

def main():
    try:
        planner = LLMCognitivePlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("LLM Cognitive Planner node terminated")

if __name__ == '__main__':
    main()
```

## Advanced Planning Concepts

### Hierarchical Task Networks
```python
class HierarchicalPlanner:
    def __init__(self):
        self.primitive_actions = {
            'move_to': self.execute_move_to,
            'rotate': self.execute_rotate,
            'pick_up': self.execute_pick_up,
            'place': self.execute_place
        }

        self.composite_tasks = {
            'assemble_item': [
                {'action': 'move_to', 'params': {'x': 1.0, 'y': 1.0}},
                {'action': 'pick_up', 'params': {'object_name': 'part_a'}},
                {'action': 'move_to', 'params': {'x': 2.0, 'y': 1.0}},
                {'action': 'place', 'params': {'object_name': 'part_a', 'x': 2.0, 'y': 1.0}},
            ],
            'deliver_item': [
                {'action': 'approach', 'params': {'object_name': 'item_to_deliver'}},
                {'action': 'pick_up', 'params': {'object_name': 'item_to_deliver'}},
                {'action': 'move_to', 'params': {'x': 3.0, 'y': 2.0}},  # delivery location
                {'action': 'place', 'params': {'object_name': 'item_to_deliver', 'x': 3.0, 'y': 2.0}},
            ]
        }

    def decompose_task(self, task_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose high-level task into primitive actions"""
        if task_name in self.composite_tasks:
            # Instantiate composite task with parameters
            instantiated_steps = []
            for step in self.composite_tasks[task_name]:
                instantiated_step = self.instantiate_step(step, params)
                instantiated_steps.append(instantiated_step)
            return instantiated_steps

        # If it's already a primitive action
        if task_name in self.primitive_actions:
            return [{'action': task_name, 'parameters': params}]

        raise ValueError(f"Unknown task: {task_name}")

    def instantiate_step(self, step: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Instantiate step with specific parameters"""
        instantiated = step.copy()

        # Replace parameter placeholders with actual values
        for key, value in instantiated['params'].items():
            if isinstance(value, str) and value.startswith('$'):
                param_name = value[1:]  # Remove '$' prefix
                if param_name in params:
                    instantiated['params'][key] = params[param_name]

        return instantiated
```

### Context-Aware Planning
```python
class ContextAwarePlanner:
    def __init__(self):
        self.context_history = []
        self.user_preferences = {}
        self.robot_state_history = []

    def update_context(self, new_context: Dict[str, Any]):
        """Update planner context with new information"""
        self.context_history.append({
            'timestamp': rospy.get_rostime(),
            'context': new_context
        })

        # Keep only recent context (last 10 items)
        if len(self.context_history) > 10:
            self.context_history = self.context_history[-10:]

    def get_relevant_context(self, current_task: str) -> str:
        """Extract relevant context for current task"""
        relevant_items = []

        for item in self.context_history:
            # Simple keyword matching (in practice, use semantic similarity)
            if any(keyword in current_task.lower() for keyword in ['object', 'move', 'grasp']):
                relevant_items.append(item['context'])

        return json.dumps(relevant_items[-3:])  # Return last 3 relevant contexts

    def refine_plan_with_context(self, plan: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Refine plan based on context history"""
        # Example: If user frequently asks for slow movements, adjust plan
        slow_movement_requested = self.has_recent_slow_request(task)

        if slow_movement_requested:
            for step in plan.get('plan_steps', []):
                if step['action'] in ['move_to', 'rotate']:
                    # Add slow movement modifier
                    step['parameters']['speed'] = 'slow'

        return plan

    def has_recent_slow_request(self, current_task: str) -> bool:
        """Check if user recently requested slow movements"""
        for item in self.context_history[-5:]:  # Check last 5 interactions
            context = item['context']
            if 'slow' in str(context).lower() or 'carefully' in str(context).lower():
                return True
        return False
```

## Safety Mechanisms for LLM Planning

### Plan Validation and Filtering
```python
class SafeLLMPlanner:
    def __init__(self):
        # Define restricted actions and dangerous patterns
        self.restricted_actions = [
            'goto_unmapped_area',
            'enter_restricted_zone',
            'perform_self_diagnostics',
            'modify_system_settings'
        ]

        self.dangerous_patterns = [
            r'.*destroy.*',
            r'.*break.*',
            r'.*damage.*',
            r'.*harm.*',
            r'.*unsafe.*',
            r'.*dangerous.*'
        ]

    def filter_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Filter dangerous or inappropriate actions from plan"""
        filtered_steps = []

        for step in plan['plan_steps']:
            action = step['action']

            # Check for restricted actions
            if action in self.restricted_actions:
                rospy.logwarn(f"Skipping restricted action: {action}")
                continue

            # Check for dangerous patterns in parameters
            params_str = json.dumps(step.get('parameters', {}))
            is_dangerous = any(
                re.search(pattern, params_str, re.IGNORECASE)
                for pattern in self.dangerous_patterns
            )

            if is_dangerous:
                rospy.logwarn(f"Skipping dangerous action: {action}")
                continue

            # Add safety checks to each step
            step_with_safety = self.add_safety_checks(step)
            filtered_steps.append(step_with_safety)

        plan['plan_steps'] = filtered_steps
        return plan

    def add_safety_checks(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Add safety checks to plan step"""
        action = step['action']

        # Add safety parameters
        if action == 'move_to':
            # Add collision checking
            step['safety_checks'] = ['collision_check', 'kinematic_feasibility']

        elif action == 'pick_up':
            # Add grasp validation
            step['safety_checks'] = ['object_validation', 'grasp_feasibility']

        return step
```

## Multi-Modal Integration

### Combining Vision and LLM Planning
```python
class MultiModalPlanner:
    def __init__(self):
        self.perception_system = PerceptionInterface()
        self.llm_planner = LLMCognitivePlanner()
        self.action_executor = ActionExecutor()

    def generate_vision_guided_plan(self, natural_task: str) -> Dict[str, Any]:
        """Generate plan guided by current visual input"""
        # Get current visual observations
        visual_observations = self.perception_system.get_current_scene()

        # Enhance natural task with visual context
        enhanced_task = self.enhance_task_with_vision(natural_task, visual_observations)

        # Generate plan with enhanced context
        plan = self.llm_planner.generate_plan(enhanced_task)

        return plan

    def enhance_task_with_vision(self, task: str, observations: Dict[str, Any]) -> str:
        """Enhance natural language task with visual context"""
        object_descriptions = []
        for obj in observations.get('objects', []):
            desc = f"{obj['name']} at position ({obj['position']['x']:.2f}, {obj['position']['y']:.2f})"
            object_descriptions.append(desc)

        enhanced_prompt = f"""
        Original task: {task}

        Current scene observations:
        Objects detected: {', '.join(object_descriptions)}
        Environment: {observations.get('environment', 'indoor')}

        Please generate a plan considering the current visual context.
        """

        return enhanced_prompt
```

## Simulation-Based Lab Exercise: LLM-Powered Task Planner

### Objective
Create an LLM-powered cognitive planning system that can interpret natural language commands and generate executable robotic plans in simulation.

### Prerequisites
- ROS 2 installation
- OpenAI API key (or equivalent for other LLM provider)
- Gazebo simulation environment

### Steps

#### 1. Set Up LLM Integration
Create the LLM cognitive planner node with proper error handling and safety checks.

#### 2. Integrate with Simulation
Connect the planner to a simulated robot environment:
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import json

class SimulationLLMPlanner(Node):
    def __init__(self):
        super().__init__('simulation_llm_planner')

        # Publishers for simulated robot
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.task_sub = self.create_subscription(
            String, '/natural_language_task', self.process_task, 10
        )

        # Robot state
        self.current_position = {'x': 0.0, 'y': 0.0, 'theta': 0.0}

        self.get_logger().info("Simulation LLM Planner initialized")

    def process_task(self, msg):
        """Process natural language task in simulation"""
        task = msg.data
        self.get_logger().info(f"Processing task: {task}")

        # In simulation, we can simulate the LLM response
        simulated_plan = self.simulate_llm_response(task)

        # Execute the plan in simulation
        self.execute_simulated_plan(simulated_plan)

    def simulate_llm_response(self, task):
        """Simulate LLM response for demonstration"""
        # This simulates what the LLM would return
        if "move forward" in task.lower():
            return {
                "task_description": task,
                "plan_steps": [
                    {"action": "move_to", "parameters": {"x": 1.0, "y": 0.0}, "description": "Move forward 1 meter"}
                ],
                "estimated_duration": 5.0,
                "potential_risks": []
            }
        elif "turn left" in task.lower():
            return {
                "task_description": task,
                "plan_steps": [
                    {"action": "rotate", "parameters": {"theta": 1.57}, "description": "Turn 90 degrees left"}
                ],
                "estimated_duration": 3.0,
                "potential_risks": []
            }
        else:
            return {
                "task_description": task,
                "plan_steps": [
                    {"action": "unknown", "parameters": {}, "description": "Task not understood"}
                ],
                "estimated_duration": 0.0,
                "potential_risks": ["task not understood"]
            }

    def execute_simulated_plan(self, plan):
        """Execute plan in simulated environment"""
        for step in plan['plan_steps']:
            action = step['action']
            params = step.get('parameters', {})

            self.get_logger().info(f"Executing: {step['description']}")

            if action == 'move_to':
                self.simulate_move_to(params['x'], params['y'])
            elif action == 'rotate':
                self.simulate_rotation(params['theta'])
            # Add more actions as needed

            # Small delay between steps
            self.get_clock().sleep_for(Duration(seconds=1))

    def simulate_move_to(self, target_x, target_y):
        """Simulate movement to target position"""
        # In a real simulation, this would interface with Gazebo
        self.get_logger().info(f"Simulating movement to ({target_x}, {target_y})")

        # Publish command to simulated robot
        cmd = Twist()
        cmd.linear.x = 0.2  # Move at 0.2 m/s
        self.cmd_vel_pub.publish(cmd)

        # Simulate time delay
        self.get_clock().sleep_for(Duration(seconds=abs(target_x)/0.2))

def main(args=None):
    rclpy.init(args=args)
    planner = SimulationLLMPlanner()

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.get_logger().info("Shutting down")
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 3. Test Various Commands
Test the system with various natural language commands:
- "Move forward 2 meters"
- "Turn left and move to the red block"
- "Pick up the blue cube and place it on the table"

#### 4. Validate Safety Features
- Test with potentially dangerous commands
- Verify that safety checks prevent unsafe plans
- Test error handling for misunderstood commands

### Expected Outcome
- LLM-powered cognitive planner that interprets natural language
- Safe execution of generated plans in simulation
- Proper error handling for ambiguous commands
- Complete integration with robotic control system

## Best Practices for LLM Integration

### 1. Context Management
- Provide rich context to the LLM about robot capabilities
- Include environmental information in prompts
- Maintain conversation history for context awareness

### 2. Validation and Safety
- Always validate LLM outputs before execution
- Implement multiple layers of safety checks
- Have human-in-the-loop for critical decisions

### 3. Error Handling
- Gracefully handle LLM API failures
- Provide fallback mechanisms for critical functions
- Implement timeouts for LLM responses

### 4. Performance Optimization
- Cache common responses to reduce API calls
- Use local models when possible for low-latency responses
- Batch similar requests when appropriate

## Summary

In this chapter, we've explored LLM cognitive planning:
- Integrating Large Language Models with robotic systems
- Generating executable plans from natural language commands
- Implementing safety checks for LLM-generated plans
- Creating context-aware planning systems
- Combining vision and language for better planning
- Best practices for LLM integration in robotics

LLM cognitive planning bridges the gap between high-level human intentions and low-level robotic actions, enabling more natural and flexible human-robot interaction.

## Next Steps

Continue to the next chapter: [Multi-Step Robotic Actions](./multi-step-robotic-actions.md) to learn how to sequence complex robotic behaviors.