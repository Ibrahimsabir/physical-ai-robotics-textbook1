---
title: Isaac Sim for Photorealistic Simulation
sidebar_label: Isaac Sim Photorealistic
description: Creating photorealistic simulation environments using NVIDIA Isaac Sim for robotic applications
keywords: [nvidia, isaac, simulation, photorealistic, omniverse, robotics, rendering]
---

# Isaac Sim for Photorealistic Simulation

## Introduction

NVIDIA Isaac Sim is a powerful robotics simulation environment built on the Omniverse platform. It provides photorealistic rendering capabilities that enable the creation of high-fidelity simulation environments for training and testing robotic systems. This chapter explores how to set up and use Isaac Sim for creating realistic robotic simulation scenarios.

## Learning Objectives

By the end of this chapter, you will be able to:
- Install and configure NVIDIA Isaac Sim
- Create photorealistic environments for robotics
- Configure advanced rendering and lighting
- Implement sensor simulation with realistic physics
- Generate synthetic data for AI model training
- Apply domain randomization techniques
- Optimize simulation performance

## Isaac Sim Architecture

### Core Components
- **Omniverse USD**: Universal Scene Description for 3D scenes
- **PhysX Physics Engine**: Realistic physics simulation
- **RTX Renderer**: Photorealistic rendering with ray tracing
- **ROS/ROS2 Bridge**: Integration with robotics frameworks
- **Isaac Extensions**: Specialized robotics tools and features

### System Requirements
- **GPU**: NVIDIA RTX series (RTX 3060 or higher recommended)
- **VRAM**: 8GB or more for complex scenes
- **CPU**: Multi-core processor (Intel i7 or AMD Ryzen equivalent)
- **RAM**: 16GB or more
- **OS**: Windows 10/11 or Ubuntu 20.04+

## Installing Isaac Sim

### Prerequisites
```bash
# Ensure NVIDIA drivers are up to date
# Install Omniverse Launcher from NVIDIA Developer website
# Install CUDA toolkit (version compatible with Isaac Sim)
```

### Installation Process
```bash
# 1. Download Omniverse Launcher
# 2. Install Isaac Sim through the launcher
# 3. Install Isaac ROS packages if using ROS integration:
sudo apt update
sudo apt install ros-humble-isaac-sim-bridge
```

## Setting Up Your First Isaac Sim Environment

### Basic Scene Creation
```python
# Example Python script to create a basic scene in Isaac Sim
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import numpy as np

def setup_basic_scene():
    # Initialize Isaac Sim world
    world = World(stage_units_in_meters=1.0)

    # Add a simple robot
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("Could not find Isaac Sim assets. Please enable Isaac Sim Kit in Nucleus.")
        return None

    # Add a simple cuboid as a basic robot
    world.scene.add(
        prim_path="/World/Robot",
        name="simple_robot",
        position=np.array([0, 0, 0.5]),
        orientation=np.array([0, 0, 0, 1]),
        scale=np.array([0.3, 0.3, 0.3])
    )

    # Add ground plane
    world.scene.add_ground_plane("/World/Ground", static_friction=0.5, dynamic_friction=0.5)

    return world

# Run the simulation
world = setup_basic_scene()
if world:
    world.reset()

    for i in range(1000):
        world.step(render=True)

        if i % 100 == 0:
            print(f"Simulation step: {i}")
```

## Photorealistic Environment Creation

### USD Scene Structure
Isaac Sim uses Universal Scene Description (USD) for scene representation:
- **/World**: Root of the scene hierarchy
- **/World/Robots**: Robot instances
- **/World/Environments**: Environmental objects
- **/World/Lights**: Lighting setup
- **/World/Sensors**: Sensor configurations

### Creating Complex Environments
```python
import omni
from pxr import Gf, Sdf, UsdGeom, UsdShade
import carb

def create_photorealistic_office():
    """Create a photorealistic office environment"""
    stage = omni.usd.get_context().get_stage()

    # Create office room
    room_prim = stage.DefinePrim("/World/OfficeRoom", "Xform")

    # Add walls
    add_wall(room_prim, "Wall_Left", [-5, 0, 2.5], [0.2, 10, 5], [0.8, 0.8, 0.8])
    add_wall(room_prim, "Wall_Right", [5, 0, 2.5], [0.2, 10, 5], [0.8, 0.8, 0.8])
    add_wall(room_prim, "Wall_Front", [0, -5, 2.5], [10, 0.2, 5], [0.8, 0.8, 0.8])
    add_wall(room_prim, "Wall_Back", [0, 5, 2.5], [10, 0.2, 5], [0.8, 0.8, 0.8])

    # Add ceiling
    add_ceiling(room_prim, "Ceiling", [0, 0, 5], [10, 10, 0.2], [0.9, 0.9, 0.9])

    # Add floor
    add_floor(room_prim, "Floor", [0, 0, 0], [10, 10, 0.2], [0.6, 0.6, 0.6])

    # Add furniture
    add_desk(room_prim, "Desk", [-2, -2, 0.5])
    add_chair(room_prim, "Chair", [-1.5, -1.5, 0.5])
    add_bookshelf(room_prim, "Bookshelf", [3, -4, 1.0])

    # Add lighting
    add_lighting(stage)

def add_wall(parent_prim, name, position, size, color):
    """Add a wall to the scene"""
    wall_path = f"{parent_prim.GetPath()}/{name}"
    wall = UsdGeom.Cube.Define(parent_prim.GetStage(), wall_path)
    wall.GetSizeAttr().Set(max(size))
    wall.GetXformOp().SetTranslate(Gf.Vec3f(*position))

    # Apply material
    add_material(wall, f"{wall_path}_Material", color)

def add_material(prim, material_name, color):
    """Add a material to a primitive"""
    stage = prim.GetStage()
    material_path = Sdf.Path(f"/World/Materials/{material_name}")
    material = UsdShade.Material.Define(stage, material_path)

    # Create PBR shader
    shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)

    # Bind material to prim
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(material)

def add_lighting(stage):
    """Add photorealistic lighting to the scene"""
    # Add dome light for ambient lighting
    dome_light = UsdGeom.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.GetIntensityAttr().Set(1.0)
    dome_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    # Add directional light to simulate sun
    sun_light = UsdGeom.DistantLight.Define(stage, "/World/SunLight")
    sun_light.GetIntensityAttr().Set(500.0)
    sun_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.9))
    sun_light.GetDirectionAttr().Set(Gf.Vec3f(-1, -1, -1))
```

## Advanced Rendering Features

### RTX Rendering Settings
```python
def configure_rtx_rendering():
    """Configure advanced RTX rendering settings"""
    # Access rendering settings
    settings = carb.settings.get_settings()

    # Enable RTX rendering
    settings.set("/rtx/renderMode", "RayTracedLightmapped")

    # Configure ray tracing quality
    settings.set("/rtx/raytracing/enable", True)
    settings.set("/rtx/raytracing/maxBounces", 8)
    settings.set("/rtx/raytracing/maxDiffuseBounces", 4)
    settings.set("/rtx/raytracing/maxSpecularBounces", 4)
    settings.set("/rtx/raytracing/maxTransmissionBounces", 8)

    # Configure denoising
    settings.set("/rtx/denoise/enable", True)
    settings.set("/rtx/denoise/enableDlss", True)

    # Configure path tracing
    settings.set("/rtx/pathtracing/enable", True)
    settings.set("/rtx/pathtracing/maxSamples", 256)
```

### Physically-Based Materials
```python
def create_pbr_materials():
    """Create physically-based materials for realistic rendering"""
    stage = omni.usd.get_context().get_stage()

    # Metal material
    metal_mat = UsdShade.Material.Define(stage, "/World/Materials/Metal")
    metal_shader = UsdShade.Shader.Define(stage, "/World/Materials/Metal/Surface")
    metal_shader.CreateIdAttr("UsdPreviewSurface")

    metal_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.7, 0.7, 0.8))
    metal_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.95)
    metal_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
    metal_shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))

    metal_mat.CreateSurfaceOutput().ConnectToSource(metal_shader.ConnectableAPI(), "surface")

    # Plastic material
    plastic_mat = UsdShade.Material.Define(stage, "/World/Materials/Plastic")
    plastic_shader = UsdShade.Shader.Define(stage, "/World/Materials/Plastic/Surface")
    plastic_shader.CreateIdAttr("UsdPreviewSurface")

    plastic_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.2, 0.6, 0.8))
    plastic_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    plastic_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)

    plastic_mat.CreateSurfaceOutput().ConnectToSource(plastic_shader.ConnectableAPI(), "surface")
```

## Sensor Simulation

### Camera Sensor Configuration
```python
from omni.isaac.sensor import Camera

def setup_photorealistic_camera(robot_prim_path, position, orientation):
    """Set up a photorealistic camera for the robot"""
    # Create camera
    camera = Camera(
        prim_path=f"{robot_prim_path}/camera",
        position=position,
        frequency=30,  # Hz
        resolution=(1280, 720)
    )

    # Configure advanced camera properties
    camera_config = camera.prim.GetAttribute("sensor:camera:projectionType").Set("perspective")
    camera_config = camera.prim.GetAttribute("sensor:camera:focalLength").Set(24.0)  # mm
    camera_config = camera.prim.GetAttribute("sensor:camera:horizontalAperture").Set(36.0)  # mm

    # Enable physically-based camera effects
    camera.enable_distortion(True)
    camera.set_distortion_params(k1=-0.1, k2=0.05, p1=0.0, p2=0.0)

    return camera

def capture_photorealistic_images(camera, num_images=10):
    """Capture photorealistic images with advanced effects"""
    images = []

    for i in range(num_images):
        # Step the world to update sensor data
        omni.timeline.get_timeline_interface().play()

        # Get image data
        image = camera.get_rgb()
        depth = camera.get_depth()
        segmentation = camera.get_semantic_segmentation()

        # Apply post-processing effects
        processed_image = apply_post_processing(image, depth, segmentation)

        images.append(processed_image)

        # Move camera slightly for variety
        current_pos = camera.get_world_pos()
        camera.set_world_pos([
            current_pos[0] + np.random.uniform(-0.01, 0.01),
            current_pos[1] + np.random.uniform(-0.01, 0.01),
            current_pos[2] + np.random.uniform(-0.01, 0.01)
        ])

    return images

def apply_post_processing(rgb_image, depth_image, segmentation):
    """Apply photorealistic post-processing effects"""
    # Simulate lens effects
    rgb_image = add_lens_flare(rgb_image)
    rgb_image = add_chromatic_aberration(rgb_image)

    # Add realistic noise patterns
    rgb_image = add_realistic_noise(rgb_image)

    # Simulate depth of field
    rgb_image = add_depth_of_field(rgb_image, depth_image)

    return {
        'rgb': rgb_image,
        'depth': depth_image,
        'segmentation': segmentation
    }
```

## Synthetic Data Generation

### Domain Randomization
```python
import random
import colorsys

class DomainRandomizer:
    def __init__(self):
        self.light_properties = {
            'intensity_range': (100, 1000),
            'color_temperature_range': (3000, 8000),  # Kelvin
            'position_jitter': 0.5
        }

        self.material_properties = {
            'albedo_range': (0.1, 1.0),
            'roughness_range': (0.0, 1.0),
            'metallic_range': (0.0, 1.0)
        }

    def randomize_lighting(self, light_prim):
        """Randomize lighting properties"""
        intensity = random.uniform(*self.light_properties['intensity_range'])
        color_temp = random.uniform(*self.light_properties['color_temperature_range'])

        # Convert color temperature to RGB
        color_rgb = self.color_temperature_to_rgb(color_temp)

        light_prim.GetIntensityAttr().Set(intensity)
        light_prim.GetColorAttr().Set(color_rgb)

        # Add position jitter
        current_pos = light_prim.GetXformOp().Get()
        jitter = [
            random.uniform(-self.light_properties['position_jitter'], self.light_properties['position_jitter'])
            for _ in range(3)
        ]
        new_pos = [current_pos[i] + jitter[i] for i in range(3)]
        light_prim.GetXformOp().Set(Gf.Vec3f(*new_pos))

    def randomize_materials(self, material_prim):
        """Randomize material properties"""
        # Get shader
        surface_output = material_prim.GetSurfaceOutput()
        shader_prim = surface_output.GetConnectedSource()[0].GetPrim()

        # Randomize material properties
        albedo = random.uniform(*self.material_properties['albedo_range'])
        roughness = random.uniform(*self.material_properties['roughness_range'])
        metallic = random.uniform(*self.material_properties['metallic_range'])

        # Apply randomization
        shader_prim.GetInput('diffuseColor').Set((
            random.uniform(0.0, albedo),
            random.uniform(0.0, albedo),
            random.uniform(0.0, albedo)
        ))
        shader_prim.GetInput('roughness').Set(roughness)
        shader_prim.GetInput('metallic').Set(metallic)

    def color_temperature_to_rgb(self, kelvin):
        """Convert color temperature in Kelvin to RGB"""
        temp = kelvin / 100

        if temp <= 66:
            red = 255
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            red = max(0, min(255, red))

        if temp <= 66:
            green = temp
            green = 99.4708025861 * math.log(green) - 161.1195681661
        else:
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)
        green = max(0, min(255, green))

        if temp >= 66:
            blue = 255
        elif temp <= 19:
            blue = 0
        else:
            blue = temp - 10
            blue = 138.5177312231 * math.log(blue) - 305.0447927307
            blue = max(0, min(255, blue))

        return (red/255.0, green/255.0, blue/255.0)
```

### Synthetic Dataset Generation Pipeline
```python
def generate_synthetic_dataset(robot, environment, num_scenes=1000):
    """Generate a synthetic dataset with randomized environments"""
    randomizer = DomainRandomizer()
    dataset = []

    for scene_idx in range(num_scenes):
        print(f"Generating scene {scene_idx+1}/{num_scenes}")

        # Randomize environment
        randomize_scene(environment, randomizer)

        # Randomize robot position and orientation
        randomize_robot_pose(robot)

        # Capture sensor data
        sensor_data = capture_sensor_data(robot)

        # Create annotation
        annotation = create_annotation(robot, environment, sensor_data)

        # Save data
        sample = {
            'scene_id': scene_idx,
            'sensor_data': sensor_data,
            'annotation': annotation,
            'environment_config': get_environment_config(environment),
            'robot_config': get_robot_config(robot)
        }

        dataset.append(sample)

        # Save sample to disk
        save_sample_to_disk(sample, f"./dataset/sample_{scene_idx:06d}")

    return dataset

def randomize_scene(environment, randomizer):
    """Randomize the scene properties"""
    # Randomize lighting
    lights = get_all_lights_in_scene(environment)
    for light in lights:
        randomizer.randomize_lighting(light)

    # Randomize materials
    materials = get_all_materials_in_scene(environment)
    for material in materials:
        randomizer.randomize_materials(material)

    # Randomize object positions and properties
    objects = get_all_objects_in_scene(environment)
    for obj in objects:
        randomize_object_properties(obj)

def capture_sensor_data(robot):
    """Capture synchronized sensor data from the robot"""
    # Get data from all sensors
    rgb_image = robot.camera.get_rgb()
    depth_image = robot.camera.get_depth()
    segmentation = robot.camera.get_semantic_segmentation()

    # Get robot state
    robot_pose = robot.get_world_pose()
    robot_velocity = robot.get_linear_velocity()

    # Get other sensor data if available
    imu_data = robot.imu.get_data() if hasattr(robot, 'imu') else None
    lidar_data = robot.lidar.get_point_cloud() if hasattr(robot, 'lidar') else None

    return {
        'rgb': rgb_image,
        'depth': depth_image,
        'segmentation': segmentation,
        'robot_pose': robot_pose,
        'robot_velocity': robot_velocity,
        'imu': imu_data,
        'lidar': lidar_data
    }

def create_annotation(robot, environment, sensor_data):
    """Create annotation for the captured data"""
    # Detect objects in scene
    objects = detect_objects_in_environment(environment)

    # Create bounding boxes for detected objects
    bboxes = []
    for obj in objects:
        bbox = get_2d_bbox_from_3d_object(obj, robot.camera)
        bboxes.append({
            'class': obj.class_name,
            'bbox_2d': bbox,
            'bbox_3d': obj.bounding_box,
            'pose': obj.pose
        })

    # Create semantic segmentation masks
    seg_masks = create_instance_masks(sensor_data['segmentation'], objects)

    return {
        'objects': bboxes,
        'instance_masks': seg_masks,
        'poses': [obj.pose for obj in objects],
        'classes': [obj.class_name for obj in objects]
    }
```

## Performance Optimization

### Scene Optimization Techniques
```python
def optimize_scene_performance():
    """Optimize scene for better simulation performance"""
    settings = carb.settings.get_settings()

    # Physics optimization
    settings.set("/physics/worker_thread_count", 4)  # Adjust based on CPU cores
    settings.set("/physics/substeps", 1)
    settings.set("/physics/solver_position_iteration_count", 4)
    settings.set("/physics/solver_velocity_iteration_count", 1)

    # Rendering optimization
    settings.set("/app/renderer/enabled", True)
    settings.set("/app/renderer/resolution/width", 1280)
    settings.set("/app/renderer/resolution/height", 720)
    settings.set("/app/renderer/max_render_width", 3840)
    settings.set("/app/renderer/max_render_height", 2160)

    # LOD (Level of Detail) settings
    settings.set("/app/lodScale", 1.0)
    settings.set("/app/lodBias", 0.0)

    # Caching settings
    settings.set("/app/sceneCacheSize", 512 * 1024 * 1024)  # 512 MB

def use_proxy_shapes_for_complex_models():
    """Use proxy shapes for complex models during simulation"""
    # For complex models, use simplified proxy shapes for physics
    # while keeping detailed models for rendering
    stage = omni.usd.get_context().get_stage()

    # Example: Create a simplified collision mesh for a complex robot
    complex_robot_mesh = stage.GetPrimAtPath("/World/Robot/complex_model")
    collision_proxy = UsdGeom.Mesh.Define(stage, "/World/Robot/collision_proxy")

    # Set collision proxy to be used for physics but not rendering
    collision_proxy.GetPurposeAttr().Set("proxy")

    # Set detailed mesh for rendering
    complex_robot_mesh.GetPurposeAttr().Set("render")
```

## Isaac Sim with ROS/ROS2 Integration

### Setting up Isaac ROS Bridge
```python
# Example launch file for Isaac Sim ROS bridge
"""
<launch>
  <!-- Isaac Sim Bridge -->
  <node
    pkg="isaac_ros_bridges"
    exec="isaac_sim_bridge"
    name="isaac_sim_bridge"
    output="screen">
    <param name="config_file" value="$(find-pkg-share my_robot_isaac_sim)/config/isaac_sim_bridge.yaml"/>
  </node>

  <!-- Camera publisher -->
  <node
    pkg="isaac_ros_stereo_image_proc"
    exec="isaac_ros_zed2_rectify_node"
    name="camera_rectifier"
    output="screen">
    <param name="input_image_topic" value="/camera/image_raw"/>
    <param name="output_image_topic" value="/camera/image_rect_color"/>
  </node>

  <!-- Perception pipeline -->
  <node
    pkg="isaac_ros_detect_net"
    exec="isaac_ros_detect_net"
    name="object_detector"
    output="screen">
    <param name="camera_topic" value="/camera/image_rect_color"/>
    <param name="tensorrt_engine_file_path" value="$(var model_path)/detect_net.trt"/>
  </node>
</launch>
"""
```

## Simulation-Based Lab Exercise: Photorealistic Warehouse Environment

### Objective
Create a photorealistic warehouse environment in Isaac Sim with realistic lighting, materials, and sensor simulation.

### Prerequisites
- NVIDIA Isaac Sim installed
- Compatible GPU with RTX capabilities
- Basic understanding of USD and Omniverse

### Steps

#### 1. Create the Warehouse Environment
```python
def create_photorealistic_warehouse():
    """Create a detailed warehouse environment"""
    stage = omni.usd.get_context().get_stage()

    # Create warehouse structure
    warehouse_size = [50, 30, 10]  # x, y, z in meters
    create_warehouse_structure(stage, warehouse_size)

    # Add industrial lighting
    add_industrial_lighting(stage, warehouse_size)

    # Add warehouse equipment
    add_racking_systems(stage)
    add_pallets(stage)
    add_conveyor_belts(stage)

    # Add environmental details
    add_cranes(stage)
    add_vehicles(stage)
    add_signage(stage)

    # Configure materials for photorealism
    apply_photorealistic_materials(stage)

def create_warehouse_structure(stage, size):
    """Create the basic warehouse structure"""
    # Create floor
    floor = UsdGeom.Cube.Define(stage, "/World/Warehouse/Floor")
    floor.GetSizeAttr().Set(1.0)
    floor.GetXformOp().SetTranslate(Gf.Vec3f(0, 0, -0.1))
    floor.GetXformOp().SetScale(Gf.Vec3f(size[0], size[1], 0.2))

    # Create walls
    wall_thickness = 0.5
    create_wall(stage, "/World/Warehouse/Wall_Left",
                [-size[0]/2 - wall_thickness/2, 0, size[2]/2],
                [wall_thickness, size[1], size[2]])
    create_wall(stage, "/World/Warehouse/Wall_Right",
                [size[0]/2 + wall_thickness/2, 0, size[2]/2],
                [wall_thickness, size[1], size[2]])
    create_wall(stage, "/World/Warehouse/Wall_Front",
                [0, -size[1]/2 - wall_thickness/2, size[2]/2],
                [size[0], wall_thickness, size[2]])
    create_wall(stage, "/World/Warehouse/Wall_Back",
                [0, size[1]/2 + wall_thickness/2, size[2]/2],
                [size[0], wall_thickness, size[2]])

    # Create ceiling
    ceiling = UsdGeom.Cube.Define(stage, "/World/Warehouse/Ceiling")
    ceiling.GetSizeAttr().Set(1.0)
    ceiling.GetXformOp().SetTranslate(Gf.Vec3f(0, 0, size[2]))
    ceiling.GetXformOp().SetScale(Gf.Vec3f(size[0], size[1], 0.2))

def add_industrial_lighting(stage, warehouse_size):
    """Add realistic industrial lighting"""
    # Add fluorescent tube lights along the ceiling
    spacing = 5  # meters between lights
    num_lights_x = int(warehouse_size[0] / spacing)
    num_lights_y = int(warehouse_size[1] / spacing)

    for i in range(num_lights_x):
        for j in range(num_lights_y):
            x_pos = -warehouse_size[0]/2 + (i + 0.5) * spacing
            y_pos = -warehouse_size[1]/2 + (j + 0.5) * spacing
            z_pos = warehouse_size[2] - 0.5  # Just below ceiling

            create_fluorescent_light(stage,
                                   f"/World/Warehouse/Lighting/Light_{i}_{j}",
                                   [x_pos, y_pos, z_pos])

def create_fluorescent_light(stage, path, position):
    """Create a fluorescent light"""
    light = UsdGeom.Cylinder.Define(stage, path)
    light.GetRadiusAttr().Set(0.05)  # 5cm diameter
    light.GetHeightAttr().Set(1.2)   # 1.2m length
    light.GetXformOp().SetTranslate(Gf.Vec3f(*position))

    # Add emissive material to simulate light
    add_emissive_material(light, path + "_Mat", [2.0, 2.0, 1.8])

def add_emissive_material(prim, mat_name, emission_color):
    """Add emissive material to simulate light emission"""
    stage = prim.GetStage()
    material_path = Sdf.Path(f"/World/Materials/{mat_name}")
    material = UsdShade.Material.Define(stage, material_path)

    shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")

    # Set to pure emission
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(emission_color)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set([0, 0, 0])
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Bool).Set(False)

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(material)
```

#### 2. Integrate with a Robot
Add a robot to the environment and configure its sensors:

```python
def setup_robot_in_warehouse(robot_usd_path):
    """Set up a robot in the warehouse environment"""
    # Load robot model
    add_reference_to_stage(
        usd_path=robot_usd_path,
        prim_path="/World/Robot"
    )

    # Position robot in warehouse
    robot_prim = stage.GetPrimAtPath("/World/Robot")
    robot_prim.GetXformOp().SetTranslate(Gf.Vec3f(0, 0, 0.5))  # 0.5m above ground

    # Add sensors to robot
    setup_robot_sensors("/World/Robot")

    # Configure robot for warehouse navigation
    configure_navigation_system("/World/Robot")
```

#### 3. Test Sensor Simulation
Verify that sensors are working properly in the photorealistic environment:
- Camera capturing realistic images
- LiDAR producing accurate point clouds
- IMU providing realistic inertial measurements

#### 4. Generate Synthetic Data
Run domain randomization to generate a diverse dataset for AI training.

### Expected Outcome
- Photorealistic warehouse environment with realistic lighting
- Properly configured robot with accurate sensor simulation
- Synthetic dataset suitable for AI model training
- Foundation for complex warehouse automation scenarios

## Best Practices for Photorealistic Simulation

### 1. Material Accuracy
- Use physically-based materials with correct properties
- Match real-world material properties as closely as possible
- Validate material appearance under different lighting conditions

### 2. Lighting Design
- Use realistic light intensities and color temperatures
- Include environmental lighting effects (sky, reflections)
- Consider time-of-day variations for outdoor scenarios

### 3. Performance vs. Quality
- Balance photorealism with simulation performance
- Use appropriate LOD for distant objects
- Optimize complex scenes for real-time simulation

### 4. Validation
- Compare simulation results with real-world data
- Validate sensor models against real sensors
- Test edge cases to ensure robustness

## Summary

In this chapter, we've explored Isaac Sim for photorealistic simulation:
- Installing and configuring Isaac Sim
- Creating photorealistic environments with advanced materials and lighting
- Setting up sensor simulation with realistic physics
- Generating synthetic data for AI model training
- Applying domain randomization techniques
- Optimizing performance for real-time simulation
- Integrating with ROS/ROS2 systems

Isaac Sim provides a powerful platform for creating high-fidelity simulation environments that enable safe and efficient development of robotic systems before real-world deployment.

## Next Steps

Continue to the next chapter: [Synthetic Data Generation](./synthetic-data-generation.md) to learn about generating training data for AI models.