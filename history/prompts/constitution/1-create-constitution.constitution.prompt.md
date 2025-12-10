---
id: 1
title: Create Constitution
stage: constitution
date: 2025-12-08
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: User
command: /sp.constitution
labels: [constitution,book,physical-ai,humanoid-robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

"I am writing a book on Physical AI and Humanoid Robotics. I want you to generate a complete, structured constitution file that will guide the entire book's writing style, tone, depth, and chapter structure. The book must be based on the following core content and cover the full quarter curriculum in a clear, professional, educational manner.

Core Content to Follow:

Physical AI & Humanoid Robotics: Embodied intelligence, AI systems in the physical world, connecting digital brains to physical bodies.

Overall Goal: Teach students how to design, control, simulate, and deploy humanoid robots using ROS 2, Gazebo, Unity, NVIDIA Isaac, and Vision-Language-Action systems.

Quarter Overview: AI in the physical world, digital twins, humanoid interaction, simulation, navigation, and autonomous robots.

Include the following modules:

Module 1: The Robotic Nervous System (ROS 2)

ROS 2 nodes, topics, services

Python agents controlling ROS 2

URDF for humanoid robots

Module 2: The Digital Twin (Gazebo & Unity)

Physics simulation, collisions, environments

Rendering in Unity, human-robot interaction

Sensors: LiDAR, depth cameras, IMU

Module 3: The AI-Robot Brain (NVIDIA Isaac)

Isaac Sim for photorealistic simulation

Synthetic data generation

Isaac ROS: VSLAM, navigation

Nav2 path planning for bipedal humanoids

Module 4: Vision-Language-Action (VLA)

Voice-to-Action using Whisper

LLM cognitive planning

Multi-step robotic action sequencing

Capstone Project

Autonomous Humanoid robot

Receives voice command

Plans path

Navigates obstacles

Identifies object via CV

Manipulates object"

## Response snapshot

A comprehensive constitution file for the Physical AI and Humanoid Robotics book has been created, containing:

- Core Principles including Embodied Intelligence First, Simulation-to-Reality Pipeline, ROS 2 Ecosystem Integration, Multi-Modal Perception Integration, Safety-First Design, and Modular Architecture
- Educational Standards for book content and approach
- Development Workflow standards for implementation and testing
- Governance section outlining how the constitution is maintained

The constitution is version 1.0.0 and was ratified on 2025-12-08.

## Outcome

- ✅ Impact: Created initial constitution file for Physical AI and Humanoid Robotics book
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md, created history/prompts/constitution/1-create-constitution.constitution.prompt.md
- 🔁 Next prompts: Continue with book development following the established constitution
- 🧠 Reflection: The constitution establishes a solid foundation for the book's content and development approach, focusing on practical implementation with ROS 2, simulation environments, and modern AI techniques.

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin developing book chapters following the established principles