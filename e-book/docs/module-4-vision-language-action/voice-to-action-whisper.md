---
title: Voice-to-Action with Whisper
sidebar_label: Voice-to-Action
description: Implementing speech recognition and command parsing using Whisper for robotic systems
keywords: [whisper, speech recognition, voice control, robotics, natural language processing]
---

# Voice-to-Action with Whisper

## Introduction

Speech recognition provides a natural and intuitive interface for commanding robotic systems. OpenAI's Whisper model represents a breakthrough in automatic speech recognition, offering multilingual capabilities and robust performance across various acoustic conditions. In this chapter, we'll explore how to integrate Whisper with robotic systems to create voice-controlled robots.

## Learning Objectives

By the end of this chapter, you will be able to:
- Set up and configure Whisper for robotic voice control
- Process speech commands in real-time
- Parse voice commands into structured robotic actions
- Handle ambiguous or unclear voice commands safely
- Integrate voice recognition with ROS 2 systems
- Implement error handling and fallback behaviors

## Whisper in Robotics Context

### Advantages of Whisper for Robotics
- **Multilingual Support**: Understand commands in multiple languages
- **Robustness**: Performs well in various acoustic conditions
- **Open Source**: Free to use and customize
- **Context Learning**: Can adapt to specific command vocabularies

### Challenges in Robotics
- **Real-time Processing**: Need for low-latency response
- **Ambiguous Commands**: Interpreting vague or underspecified requests
- **Safety**: Ensuring safe interpretation of commands
- **Environmental Noise**: Filtering out background sounds

## Setting Up Whisper for Robotics

### Installation
```bash
# Install Whisper and related dependencies
pip install openai-whisper
pip install torch torchvision torchaudio
pip install sounddevice numpy

# Additional dependencies for audio processing
pip install pyaudio speech_recognition
```

### Basic Whisper Configuration
```python
import whisper
import torch
import numpy as np
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class VoiceToActionNode:
    def __init__(self, model_size="base"):
        rospy.init_node('voice_to_action')

        # Load Whisper model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"Using device: {self.device}")

        self.model = whisper.load_model(model_size).to(self.device)

        # Publisher for parsed commands
        self.command_pub = rospy.Publisher('/parsed_commands', String, queue_size=10)

        # Publisher for robot control
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscriber for voice input
        self.voice_input_sub = rospy.Subscriber('/voice_input', String, self.voice_callback)

        # Configuration
        self.language = "english"  # Set expected language
        self.temperature = 0.0     # Lower temperature for more consistent results

        rospy.loginfo("Voice-to-Action node initialized")

    def voice_callback(self, msg):
        """Process voice input and convert to action"""
        try:
            # Transcribe the voice command
            transcription = self.transcribe_audio(msg.data)

            # Parse the command
            parsed_command = self.parse_command(transcription)

            # Execute or publish the command
            self.execute_command(parsed_command)

        except Exception as e:
            rospy.logerr(f"Error processing voice command: {e}")
            self.handle_error(e)

    def transcribe_audio(self, audio_path):
        """Transcribe audio file using Whisper"""
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            temperature=self.temperature,
            verbose=False
        )
        return result["text"].strip()

    def parse_command(self, text):
        """Parse natural language command into structured action"""
        text = text.lower()

        # Define command patterns
        command_patterns = {
            'move_forward': ['move forward', 'go forward', 'forward', 'move straight', 'go straight'],
            'move_backward': ['move backward', 'go backward', 'backward', 'reverse'],
            'turn_left': ['turn left', 'rotate left', 'go left'],
            'turn_right': ['turn right', 'rotate right', 'go right'],
            'stop': ['stop', 'halt', 'pause', 'freeze'],
            'follow': ['follow me', 'follow', 'come with me', 'come behind me'],
            'approach': ['approach', 'come to me', 'come here', 'move to me'],
            'grasp': ['grasp', 'pick up', 'take', 'grab'],
            'drop': ['drop', 'put down', 'release']
        }

        # Find matching command
        for action, patterns in command_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    return {
                        'action': action,
                        'confidence': 1.0,  # Placeholder confidence
                        'original_text': text,
                        'parsed_command': pattern
                    }

        # If no pattern matches, return unrecognized
        return {
            'action': 'unrecognized',
            'confidence': 0.0,
            'original_text': text,
            'parsed_command': None
        }

    def execute_command(self, parsed_command):
        """Execute the parsed command"""
        action = parsed_command['action']

        # Safety check: only execute recognized commands with high confidence
        if parsed_command['confidence'] < 0.8 or action == 'unrecognized':
            rospy.logwarn(f"Unrecognized or low-confidence command: {parsed_command['original_text']}")
            self.request_confirmation(parsed_command)
            return

        rospy.loginfo(f"Executing command: {action}")

        # Create Twist message for robot movement
        twist_cmd = Twist()

        if action == 'move_forward':
            twist_cmd.linear.x = 0.5  # Move forward at 0.5 m/s
        elif action == 'move_backward':
            twist_cmd.linear.x = -0.5  # Move backward
        elif action == 'turn_left':
            twist_cmd.angular.z = 0.5  # Turn left
        elif action == 'turn_right':
            twist_cmd.angular.z = -0.5  # Turn right
        elif action == 'stop':
            # Velocities remain zero (stopping)
            pass
        elif action in ['follow', 'approach']:
            # More complex behaviors would go here
            rospy.loginfo(f"Complex action '{action}' requires additional processing")
            return
        elif action in ['grasp', 'drop']:
            # Send to manipulation subsystem
            rospy.loginfo(f"Manipulation action '{action}' sent to gripper controller")
            return

        # Publish the command
        self.cmd_vel_pub.publish(twist_cmd)

        # Also publish parsed command for other nodes
        self.command_pub.publish(f"EXECUTE:{action}")

    def request_confirmation(self, parsed_command):
        """Request user confirmation for uncertain commands"""
        rospy.logwarn(f"Uncertain command received: '{parsed_command['original_text']}'")
        rospy.logwarn("Please confirm by repeating the command clearly or say 'cancel'")

        # In a real system, you might play an audio prompt
        # self.audio_player.play("I didn't understand. Please repeat your command.")

    def handle_error(self, error):
        """Handle errors during voice processing"""
        rospy.logerr(f"Voice processing error: {error}")

        # Publish error message
        error_msg = String()
        error_msg.data = f"ERROR: {str(error)}"
        self.command_pub.publish(error_msg)

def main():
    try:
        node = VoiceToActionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Voice-to-Action node terminated")

if __name__ == '__main__':
    main()
```

## Real-time Audio Processing

### Audio Input Node
```python
import rospy
import pyaudio
import wave
import threading
import time
from std_msgs.msg import String

class AudioInputNode:
    def __init__(self):
        rospy.init_node('audio_input')

        # Audio configuration
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper works well at 16kHz
        self.chunk = 1024
        self.record_seconds = 5  # Record 5-second chunks

        # Publisher for audio data
        self.audio_pub = rospy.Publisher('/voice_input', String, queue_size=10)

        # Audio processing thread
        self.recording = False
        self.audio_thread = None

        # Start audio recording
        self.start_recording()

        rospy.loginfo("Audio Input node initialized")

    def start_recording(self):
        """Start continuous audio recording"""
        self.recording = True
        self.audio_thread = threading.Thread(target=self.continuous_record)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def continuous_record(self):
        """Continuously record audio and publish when activity detected"""
        audio = pyaudio.PyAudio()

        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        silence_threshold = 500  # Adjust based on environment
        silent_chunks = 0
        max_silent_chunks = 20  # About 1.25 seconds of silence

        try:
            while self.recording:
                data = stream.read(self.chunk)
                frames.append(data)

                # Check for audio activity (simple energy-based VAD)
                audio_data = np.frombuffer(data, dtype=np.int16)
                energy = np.mean(np.abs(audio_data))

                if energy < silence_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0  # Reset counter when speech detected

                # If we have accumulated speech and then silence, process the chunk
                if len(frames) > 5 and silent_chunks > max_silent_chunks:
                    if len(frames) > 10:  # Make sure we have substantial speech
                        self.save_and_publish_audio(frames[:])

                    frames = []  # Reset for next chunk
                    silent_chunks = 0

                # Prevent buffer from growing indefinitely
                if len(frames) > 100:  # About 6.4 seconds at 16kHz
                    frames = frames[-50:]  # Keep only recent frames

        except Exception as e:
            rospy.logerr(f"Audio recording error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def save_and_publish_audio(self, frames):
        """Save audio frames to temporary file and publish path"""
        try:
            # Create temporary filename
            temp_filename = f"/tmp/voice_input_{int(time.time())}.wav"

            # Save audio to file
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Publish the filename
            audio_msg = String()
            audio_msg.data = temp_filename
            self.audio_pub.publish(audio_msg)

            rospy.logdebug(f"Audio saved and published: {temp_filename}")

        except Exception as e:
            rospy.logerr(f"Error saving audio: {e}")

    def shutdown(self):
        """Clean shutdown of audio recording"""
        self.recording = False
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)

def main():
    node = AudioInputNode()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Audio Input node terminated")
    finally:
        node.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Command Parsing

### Intent Recognition with Context
```python
import re
from typing import Dict, List, Tuple

class CommandParser:
    def __init__(self):
        # Define command templates with parameters
        self.command_templates = [
            {
                'pattern': r'(?:move|go|drive)\s+(?P<direction>forward|backward|left|right|ahead)\s*(?:(?:for|by)?\s*(?P<distance>\d+(?:\.\d+)?)\s*(?P<unit>meters|m|cm|centimeters|units)?)?',
                'action': 'MOVE',
                'required_args': ['direction']
            },
            {
                'pattern': r'(?:turn|rotate|pivot)\s+(?P<direction>left|right)\s*(?:(?:by|for)?\s*(?P<angle>\d+(?:\.\d+)?)\s*(?P<unit>degrees|°|radians)?)?',
                'action': 'TURN',
                'required_args': ['direction']
            },
            {
                'pattern': r'(?:pick up|grasp|take|grab|lift)\s+(?P<object>.+?)(?:\s+from\s+(?P<location>.+?))?$',
                'action': 'PICK_UP',
                'required_args': ['object']
            },
            {
                'pattern': r'(?:put down|place|drop|release)\s+(?P<object>.+?)\s+(?:on|at|in)\s+(?P<location>.+)$',
                'action': 'PLACE',
                'required_args': ['object', 'location']
            },
            {
                'pattern': r'(?:bring|fetch|get)\s+(?P<object>.+?)\s+(?:for|to)\s+(?P<recipient>.+)$',
                'action': 'FETCH',
                'required_args': ['object', 'recipient']
            }
        ]

    def parse(self, text: str) -> Dict:
        """Parse natural language command into structured action"""
        text = text.strip().lower()

        for template in self.command_templates:
            match = re.search(template['pattern'], text, re.IGNORECASE)
            if match:
                # Extract matched groups
                params = match.groupdict()

                # Validate required arguments
                missing_args = []
                for arg in template['required_args']:
                    if not params.get(arg):
                        missing_args.append(arg)

                if missing_args:
                    return {
                        'action': 'INCOMPLETE_COMMAND',
                        'intent': template['action'],
                        'missing_args': missing_args,
                        'original_text': text,
                        'confidence': 0.7
                    }

                # Return structured command
                return {
                    'action': template['action'],
                    'params': params,
                    'original_text': text,
                    'confidence': 0.9
                }

        # If no template matches, return unrecognized
        return {
            'action': 'UNRECOGNIZED',
            'original_text': text,
            'confidence': 0.0
        }

# Example usage in voice processing
def enhanced_voice_processing(text: str) -> Dict:
    """Enhanced voice command processing with context awareness"""
    parser = CommandParser()
    parsed_result = parser.parse(text)

    # Additional context processing could go here
    # For example, disambiguating based on robot state or environment

    return parsed_result
```

## Safety Mechanisms for Voice Control

### Command Validation
```python
class SafeVoiceController:
    def __init__(self):
        # Define safe zones and restricted areas
        self.safe_zones = []  # Will be populated from map
        self.restricted_actions = ['jump', 'fly', 'self_destruct']

        # Speed and motion limits
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 0.5  # rad/s
        self.max_duration = 10.0  # seconds for any single command

        # Emergency stop phrases
        self.emergency_phrases = [
            'emergency stop',
            'stop immediately',
            'safety stop',
            'cease all motion'
        ]

    def validate_command(self, parsed_command: Dict) -> Tuple[bool, str]:
        """Validate command for safety before execution"""

        # Check for emergency stop
        if parsed_command['action'] == 'UNRECOGNIZED':
            for phrase in self.emergency_phrases:
                if phrase in parsed_command['original_text'].lower():
                    return True, "Emergency command validated"

        # Check for restricted actions
        if parsed_command['action'] in self.restricted_actions:
            return False, f"Action '{parsed_command['action']}' is restricted for safety reasons"

        # Validate movement parameters
        if parsed_command['action'] == 'MOVE':
            params = parsed_command.get('params', {})
            distance = params.get('distance')

            if distance and float(distance) > 5.0:  # 5 meters max in one command
                return False, "Movement distance exceeds safe limits"

        # Validate turn parameters
        if parsed_command['action'] == 'TURN':
            params = parsed_command.get('params', {})
            angle = params.get('angle')

            if angle and float(angle) > 180:  # 180 degrees max turn
                return False, "Turn angle exceeds safe limits"

        # Check if destination is in safe zone (would need map integration)
        # This is a simplified check - in practice, you'd check against a map
        if parsed_command['action'] in ['MOVE', 'GOTO'] and not self.is_destination_safe(parsed_command):
            return False, "Destination appears to be in an unsafe area"

        return True, "Command is safe to execute"

    def is_destination_safe(self, command: Dict) -> bool:
        """Check if destination is in a safe area (simplified implementation)"""
        # In a real system, this would check against a map of safe zones
        # For now, return True as a placeholder
        return True

    def execute_safe_command(self, parsed_command: Dict):
        """Execute command with safety checks"""
        is_safe, reason = self.validate_command(parsed_command)

        if not is_safe:
            rospy.logerr(f"Unsafe command blocked: {reason}")
            self.alert_user(reason)
            return False

        # Execute the command
        try:
            self.execute_command_with_limits(parsed_command)
            return True
        except Exception as e:
            rospy.logerr(f"Error executing command: {e}")
            return False

    def alert_user(self, message: str):
        """Alert user about safety issue"""
        # In a real system, this might trigger audio feedback
        rospy.logwarn(f"Safety alert: {message}")

    def execute_command_with_limits(self, command: Dict):
        """Execute command with built-in safety limits"""
        # Implementation would apply speed/acceleration limits
        # and monitor for safety violations during execution
        pass
```

## Integration with ROS 2

### Launch File for Voice System
```xml
<launch>
  <!-- Audio input node -->
  <node pkg="voice_control" exec="audio_input_node" name="audio_input" output="screen">
    <param name="sample_rate" value="16000"/>
    <param name="channels" value="1"/>
  </node>

  <!-- Whisper processing node -->
  <node pkg="voice_control" exec="whisper_node" name="whisper_processor" output="screen">
    <param name="model_size" value="base"/>
    <param name="language" value="english"/>
  </node>

  <!-- Command execution node -->
  <node pkg="voice_control" exec="command_executor" name="command_executor" output="screen">
    <param name="max_linear_speed" value="0.5"/>
    <param name="max_angular_speed" value="0.5"/>
  </node>

  <!-- Safety supervisor -->
  <node pkg="voice_control" exec="safety_supervisor" name="safety_supervisor" output="screen">
    <param name="enable_emergency_stop" value="true"/>
  </node>
</launch>
```

## Simulation-Based Lab Exercise: Voice-Controlled Robot

### Objective
Create a complete voice-controlled robot system that can receive voice commands and execute corresponding actions in simulation.

### Prerequisites
- Gazebo simulation environment
- ROS 2 installation
- Whisper model installed

### Steps

#### 1. Create the Complete Voice Control System
Implement the nodes described in the examples above:
- Audio input node
- Whisper processing node
- Command execution node
- Safety supervisor

#### 2. Set Up Simulation Environment
```bash
# Launch Gazebo with a simple environment
gazebo --verbose worlds/empty.world

# Launch your robot model
ros2 launch your_robot_description view_robot.launch.py
```

#### 3. Integrate with Robot Control
Connect the voice control system to your simulated robot's command interface:
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import threading

class VoiceControlledRobot(Node):
    def __init__(self):
        super().__init__('voice_controlled_robot')

        # Robot command publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Voice command subscriber
        self.voice_cmd_sub = self.create_subscription(
            String, '/parsed_commands', self.voice_command_callback, 10
        )

        # Robot state
        self.current_twist = Twist()
        self.command_active = False

        self.get_logger().info('Voice Controlled Robot node started')

    def voice_command_callback(self, msg):
        """Handle parsed voice commands"""
        command_str = msg.data

        if command_str.startswith('EXECUTE:'):
            action = command_str.split(':', 1)[1]
            self.execute_action(action)

    def execute_action(self, action):
        """Execute specific action on robot"""
        twist = Twist()

        if action == 'move_forward':
            twist.linear.x = 0.3
            self.get_logger().info('Moving forward')
        elif action == 'move_backward':
            twist.linear.x = -0.3
            self.get_logger().info('Moving backward')
        elif action == 'turn_left':
            twist.angular.z = 0.5
            self.get_logger().info('Turning left')
        elif action == 'turn_right':
            twist.angular.z = -0.5
            self.get_logger().info('Turning right')
        elif action == 'stop':
            # Velocities remain zero
            self.get_logger().info('Stopping')
        else:
            self.get_logger().warn(f'Unknown action: {action}')
            return

        # Publish command
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    robot = VoiceControlledRobot()

    try:
        rclpy.spin(robot)
    except KeyboardInterrupt:
        robot.get_logger().info('Shutting down')
    finally:
        robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 4. Test Voice Commands
Test various voice commands:
- "Move forward"
- "Turn left"
- "Go backward slowly"
- "Stop"

#### 5. Validate Safety Features
- Test emergency stop commands
- Verify speed limits
- Test invalid command handling

### Expected Outcome
- Robot responds to voice commands in simulation
- Safety mechanisms prevent dangerous commands
- System handles ambiguous or unclear commands gracefully
- Complete voice-controlled robot system operational

## Best Practices for Voice Control

### 1. Clear Command Vocabulary
- Define a specific set of supported commands
- Train users on the exact phrasing
- Provide feedback for recognized commands

### 2. Error Handling
- Handle unclear or ambiguous commands safely
- Provide audio feedback for command recognition
- Implement timeout for command execution

### 3. Privacy Considerations
- Process audio locally when possible
- Minimize data transmission
- Consider privacy regulations in design

### 4. Accessibility
- Support multiple languages
- Provide alternative input methods
- Consider users with speech impediments

## Summary

In this chapter, we've explored voice-to-action systems using Whisper:
- Setting up Whisper for robotic voice control
- Processing speech commands in real-time
- Parsing voice commands into structured robotic actions
- Implementing safety mechanisms for voice control
- Integrating voice recognition with ROS 2 systems
- Best practices for voice-controlled robots

Voice-to-action systems provide a natural interface for human-robot interaction, enabling intuitive control of robotic systems through spoken commands.

## Next Steps

Continue to the next chapter: [LLM Cognitive Planning](./llm-cognitive-planning.md) to learn how to use Large Language Models for high-level task planning.