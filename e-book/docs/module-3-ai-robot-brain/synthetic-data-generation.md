---
title: Synthetic Data Generation for Robotics
sidebar_label: Synthetic Data Generation
description: Generating synthetic training data using NVIDIA Isaac Sim for AI model development in robotics
keywords: [synthetic data, data generation, ai training, computer vision, robotics, nvidia, isaac]
---

# Synthetic Data Generation for Robotics

## Introduction

Synthetic data generation is a transformative approach in robotics and AI development that allows us to create large, diverse, and perfectly labeled datasets in simulation environments. With NVIDIA Isaac Sim, we can generate photorealistic synthetic data that closely approximates real-world conditions, enabling the training of robust AI models without the need for expensive and time-consuming real-world data collection.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the principles and benefits of synthetic data generation for robotics
- Set up Isaac Sim for synthetic data generation workflows
- Implement domain randomization techniques to improve model robustness
- Generate various types of synthetic data (images, point clouds, sensor data)
- Create diverse training scenarios with randomized parameters
- Validate synthetic-to-real transfer effectiveness
- Optimize synthetic data generation pipelines for performance

## Why Synthetic Data for Robotics?

### Advantages Over Real Data Collection

1. **Cost-Effective**: No need for physical robots, sensors, or manual data labeling
2. **Safety**: Train on dangerous scenarios without risk
3. **Control**: Exact control over environmental conditions and parameters
4. **Scalability**: Generate unlimited amounts of diverse data
5. **Ground Truth**: Perfect annotations for all data points
6. **Edge Cases**: Easily create rare or dangerous scenarios
7. **Privacy**: No privacy concerns with synthetic environments

### Applications in Robotics
- **Perception**: Object detection, segmentation, pose estimation
- **Navigation**: Training navigation and path planning models
- **Manipulation**: Grasping and manipulation skill learning
- **Simulation-to-Reality Transfer**: Bridging sim-to-real gap
- **Safety Validation**: Testing in diverse, challenging scenarios

## Isaac Sim Synthetic Data Pipeline

### Core Components
1. **Scene Randomization**: Environment, lighting, and object variation
2. **Material Variation**: Appearance and texture randomization
3. **Sensor Simulation**: Accurate simulation of real sensors
4. **Annotation Generation**: Automatic ground truth creation
5. **Data Export**: Formatted output for ML training

### Synthetic Data Generation Architecture
```python
# Isaac Sim synthetic data generation pipeline
import omni
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.synthetic_utils.annotation import AnnotationParser
import numpy as np
import cv2
import json
from PIL import Image
import os

class SyntheticDataManager:
    def __init__(self, output_dir="./synthetic_data", num_samples=1000):
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.sd_helper = None
        self.annotation_parser = None

        # Create output directories
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)
        os.makedirs(f"{output_dir}/masks", exist_ok=True)

        # Initialize Isaac Sim helpers
        self.setup_synthetic_pipeline()

    def setup_synthetic_pipeline(self):
        """Initialize Isaac Sim synthetic data pipeline"""
        # Initialize the synthetic data helper
        self.sd_helper = SyntheticDataHelper()
        self.annotation_parser = AnnotationParser()

        # Configure camera settings for synthetic data capture
        self.configure_camera_settings()

        # Set up annotation types
        self.annotation_types = [
            "bounding_box_2d_tight",
            "instance_segmentation",
            "semantic_segmentation",
            "depth",
            "normal"
        ]

    def configure_camera_settings(self):
        """Configure camera for optimal synthetic data capture"""
        # Set up camera intrinsics
        self.camera_resolution = (1280, 720)
        self.camera_fov = 60  # degrees

        # Configure for photorealistic capture
        self.sd_helper.set_camera_resolution(self.camera_resolution)

    def generate_sample(self, sample_id):
        """Generate a single synthetic data sample"""
        # Randomize scene parameters
        self.randomize_scene()

        # Capture sensor data
        sensor_data = self.capture_sensor_data()

        # Generate annotations
        annotations = self.generate_annotations(sensor_data)

        # Save data
        self.save_sample(sample_id, sensor_data, annotations)

        return sample_id

    def randomize_scene(self):
        """Randomize scene parameters for domain randomization"""
        # Randomize lighting
        self.randomize_lighting()

        # Randomize materials
        self.randomize_materials()

        # Randomize object positions
        self.randomize_object_positions()

        # Randomize environmental conditions
        self.randomize_environment()

    def randomize_lighting(self):
        """Randomize lighting conditions"""
        stage = omni.usd.get_context().get_stage()

        # Get all lights in the scene
        lights = [prim for prim in stage.TraverseAll() if prim.GetTypeName() == "DistantLight" or prim.GetTypeName() == "SphereLight"]

        for light in lights:
            # Randomize intensity (between 500 and 1500)
            intensity = np.random.uniform(500, 1500)
            light.GetAttribute("inputs:intensity").Set(intensity)

            # Randomize color temperature (between 3000K and 8000K)
            color_temp = np.random.uniform(3000, 8000)
            color_rgb = self.color_temperature_to_rgb(color_temp)
            light.GetAttribute("inputs:color").Set(color_rgb)

            # Randomize position (small jitter)
            current_pos = light.GetAttribute("xformOp:translate").Get()
            if current_pos:
                jitter = np.random.uniform(-0.5, 0.5, size=3)
                new_pos = [current_pos[i] + jitter[i] for i in range(3)]
                light.GetAttribute("xformOp:translate").Set(new_pos)

    def randomize_materials(self):
        """Randomize material properties"""
        stage = omni.usd.get_context().get_stage()

        # Get all materials in the scene
        materials = [prim for prim in stage.TraverseAll() if prim.GetTypeName() == "Material"]

        for material in materials:
            # Find the shader inside the material
            for child in material.GetChildren():
                if child.GetTypeName() == "Shader":
                    shader = child
                    break

            if shader:
                # Randomize diffuse/albedo
                albedo_range = (0.1, 1.0)
                albedo = np.random.uniform(albedo_range[0], albedo_range[1], size=3)
                shader.GetAttribute("inputs:diffuse_color_constant").Set(albedo)

                # Randomize roughness
                roughness = np.random.uniform(0.1, 0.9)
                shader.GetAttribute("inputs:roughness_constant").Set(roughness)

                # Randomize metallic
                metallic = np.random.uniform(0.0, 0.9)
                shader.GetAttribute("inputs:metallic_constant").Set(metallic)

    def randomize_object_positions(self):
        """Randomize object positions and orientations"""
        stage = omni.usd.get_context().get_stage()

        # Get all mesh objects in the scene
        meshes = [prim for prim in stage.TraverseAll() if prim.GetTypeName() == "Mesh"]

        for mesh in meshes:
            # Skip if it's part of the static environment (floor, walls, etc.)
            path = mesh.GetPath().pathString
            if "floor" in path.lower() or "wall" in path.lower() or "environment" in path.lower():
                continue

            # Randomize position
            current_pos = mesh.GetAttribute("xformOp:translate").Get()
            if current_pos:
                # Randomize position within a defined area
                rand_x = np.random.uniform(-5.0, 5.0)
                rand_y = np.random.uniform(-5.0, 5.0)
                rand_z = np.random.uniform(0.1, 2.0)  # Keep above ground

                mesh.GetAttribute("xformOp:translate").Set(Gf.Vec3f(rand_x, rand_y, rand_z))

            # Randomize orientation
            rand_rot_x = np.random.uniform(-np.pi, np.pi)
            rand_rot_y = np.random.uniform(-np.pi, np.pi)
            rand_rot_z = np.random.uniform(-np.pi, np.pi)

            mesh.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(
                np.degrees(rand_rot_x),
                np.degrees(rand_rot_y),
                np.degrees(rand_rot_z)
            ))

    def randomize_environment(self):
        """Randomize environmental conditions"""
        # Randomize environmental factors
        stage = omni.usd.get_context().get_stage()

        # Add environmental effects like fog or haze
        # This would depend on specific USD stage setup
        pass

    def capture_sensor_data(self):
        """Capture sensor data from the current scene"""
        # Capture RGB image
        rgb_data = self.sd_helper.get_rgb()

        # Capture depth data
        depth_data = self.sd_helper.get_depth()

        # Capture segmentation data
        seg_data = self.sd_helper.get_semantic_segmentation()

        # Capture instance segmentation
        instance_seg_data = self.sd_helper.get_instance_segmentation()

        # Capture normal map
        normal_data = self.sd_helper.get_normals()

        return {
            'rgb': rgb_data,
            'depth': depth_data,
            'semantic_segmentation': seg_data,
            'instance_segmentation': instance_seg_data,
            'normals': normal_data
        }

    def generate_annotations(self, sensor_data):
        """Generate annotations for the captured sensor data"""
        # Generate bounding boxes from segmentation
        bboxes = self.generate_bounding_boxes(sensor_data['semantic_segmentation'])

        # Generate pose annotations
        poses = self.generate_pose_annotations(sensor_data)

        # Generate object classification labels
        labels = self.generate_classification_labels(sensor_data['semantic_segmentation'])

        return {
            'bounding_boxes': bboxes,
            'poses': poses,
            'labels': labels,
            'sensor_data_shape': sensor_data['rgb'].shape
        }

    def generate_bounding_boxes(self, segmentation_data):
        """Generate 2D bounding boxes from segmentation data"""
        bboxes = []

        # Find unique objects in segmentation
        unique_ids = np.unique(segmentation_data)

        for obj_id in unique_ids:
            if obj_id == 0:  # Skip background
                continue

            # Create mask for this object
            mask = (segmentation_data == obj_id).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 5:  # Skip very small contours
                    continue

                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Add bounding box information
                bboxes.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'object_id': int(obj_id)
                })

        return bboxes

    def generate_pose_annotations(self, sensor_data):
        """Generate pose annotations for objects"""
        # This would typically involve matching segmented objects to known 3D models
        # and calculating 6D poses using the Isaac Sim USD stage information
        poses = []

        # In a real implementation, this would access the USD stage
        # to get ground truth poses of objects
        return poses

    def generate_classification_labels(self, segmentation_data):
        """Generate classification labels from segmentation"""
        # Map segmentation IDs to class names
        # This mapping would be defined based on your specific objects
        label_mapping = self.get_label_mapping()

        labels = {}
        unique_ids = np.unique(segmentation_data)

        for obj_id in unique_ids:
            if obj_id in label_mapping:
                labels[int(obj_id)] = label_mapping[obj_id]

        return labels

    def get_label_mapping(self):
        """Get mapping from segmentation IDs to class names"""
        # This mapping should correspond to your semantic segmentation setup
        # In practice, this would be defined in your USD scene
        return {
            1: "robot",
            2: "box",
            3: "table",
            4: "chair",
            5: "wall",
            6: "floor",
            # Add more mappings as needed
        }

    def save_sample(self, sample_id, sensor_data, annotations):
        """Save synthetic data sample to disk"""
        # Save RGB image
        rgb_img = Image.fromarray(sensor_data['rgb'])
        rgb_img.save(f"{self.output_dir}/images/rgb_{sample_id:06d}.png")

        # Save depth image
        depth_img = Image.fromarray(sensor_data['depth'])
        depth_img.save(f"{self.output_dir}/images/depth_{sample_id:06d}.png")

        # Save semantic segmentation mask
        seg_img = Image.fromarray(sensor_data['semantic_segmentation'])
        seg_img.save(f"{self.output_dir}/masks/seg_{sample_id:06d}.png")

        # Save instance segmentation mask
        inst_seg_img = Image.fromarray(sensor_data['instance_segmentation'])
        inst_seg_img.save(f"{self.output_dir}/masks/inst_seg_{sample_id:06d}.png")

        # Save annotations as JSON
        annotations['sample_id'] = sample_id
        with open(f"{self.output_dir}/annotations/annot_{sample_id:06d}.json", 'w') as f:
            json.dump(annotations, f, indent=2)

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

    def generate_dataset(self):
        """Generate the complete synthetic dataset"""
        print(f"Generating {self.num_samples} synthetic samples...")

        for i in range(self.num_samples):
            sample_id = self.generate_sample(i)
            if i % 100 == 0:
                print(f"Generated {i}/{self.num_samples} samples")

        print(f"Dataset generation complete! Saved to {self.output_dir}")

# Example usage
if __name__ == "__main__":
    # Initialize the synthetic data manager
    synth_gen = SyntheticDataManager(
        output_dir="./synthetic_robotics_dataset",
        num_samples=5000  # Generate 5000 samples
    )

    # Generate the dataset
    synth_gen.generate_dataset()
```

## Domain Randomization Techniques

Domain randomization is a critical technique for improving the robustness of models trained on synthetic data, making them more transferable to real-world scenarios.

### Basic Domain Randomization
```python
class DomainRandomizer:
    def __init__(self):
        self.randomization_settings = {
            # Lighting randomization
            'light_intensity_range': (500, 1500),
            'light_color_temperature_range': (3000, 8000),  # Kelvin
            'light_position_jitter': 1.0,  # meters

            # Material randomization
            'albedo_range': (0.1, 1.0),
            'roughness_range': (0.05, 0.95),
            'metallic_range': (0.0, 0.9),

            # Environmental randomization
            'background_texture_probability': 0.7,
            'object_texture_probability': 0.8,
            'environmental_effects_probability': 0.3,

            # Camera randomization
            'camera_position_jitter': 0.1,  # meters
            'camera_rotation_jitter': 0.1,  # radians
        }

    def randomize_scene(self, stage):
        """Apply domain randomization to the scene"""
        # Randomize lighting
        self.randomize_lighting(stage)

        # Randomize materials
        self.randomize_materials(stage)

        # Randomize environmental conditions
        self.randomize_environment(stage)

        # Randomize camera parameters
        self.randomize_camera_parameters()

    def randomize_lighting(self, stage):
        """Randomize lighting in the scene"""
        lights = [prim for prim in stage.TraverseAll()
                 if prim.GetTypeName() in ["DistantLight", "SphereLight", "DiskLight"]]

        for light in lights:
            # Randomize intensity
            intensity = np.random.uniform(
                self.randomization_settings['light_intensity_range'][0],
                self.randomization_settings['light_intensity_range'][1]
            )
            light.GetAttribute("inputs:intensity").Set(intensity)

            # Randomize color temperature
            color_temp = np.random.uniform(
                self.randomization_settings['light_color_temperature_range'][0],
                self.randomization_settings['light_color_temperature_range'][1]
            )
            color_rgb = self.kelvin_to_rgb(color_temp)
            light.GetAttribute("inputs:color").Set(color_rgb)

            # Randomize position
            current_pos = light.GetAttribute("xformOp:translate").Get()
            if current_pos:
                jitter = np.random.uniform(
                    -self.randomization_settings['light_position_jitter'],
                    self.randomization_settings['light_position_jitter'],
                    size=3
                )
                new_pos = [current_pos[i] + jitter[i] for i in range(3)]
                light.GetAttribute("xformOp:translate").Set(new_pos)

    def randomize_materials(self, stage):
        """Randomize materials in the scene"""
        materials = [prim for prim in stage.TraverseAll() if prim.GetTypeName() == "Material"]

        for material in materials:
            shader = self.get_shader_for_material(material)
            if shader:
                # Randomize material properties
                self.apply_random_material_properties(shader)

    def apply_random_material_properties(self, shader):
        """Apply randomized material properties"""
        # Randomize albedo/diffuse
        albedo = np.random.uniform(
            self.randomization_settings['albedo_range'][0],
            self.randomization_settings['albedo_range'][1],
            size=3
        )
        shader.GetAttribute("inputs:diffuse_color_constant").Set(albedo)

        # Randomize roughness
        roughness = np.random.uniform(
            self.randomization_settings['roughness_range'][0],
            self.randomization_settings['roughness_range'][1]
        )
        shader.GetAttribute("inputs:roughness_constant").Set(roughness)

        # Randomize metallic
        metallic = np.random.uniform(
            self.randomization_settings['metallic_range'][0],
            self.randomization_settings['metallic_range'][1]
        )
        shader.GetAttribute("inputs:metallic_constant").Set(metallic)

    def get_shader_for_material(self, material):
        """Find the shader associated with a material"""
        for child in material.GetChildren():
            if child.GetTypeName() == "Shader":
                return child
        return None

    def kelvin_to_rgb(self, kelvin):
        """Convert Kelvin temperature to RGB color"""
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

### Advanced Domain Randomization
```python
class AdvancedDomainRandomizer(DomainRandomizer):
    def __init__(self):
        super().__init__()

        # Add advanced randomization settings
        self.advanced_settings = {
            # Texture randomization
            'texture_change_probability': 0.6,
            'texture_scale_range': (0.5, 2.0),
            'texture_rotation_range': (0, 2 * math.pi),

            # Geometric randomization
            'object_scale_jitter': 0.1,  # 10% scale variation
            'object_position_jitter': 0.05,  # 5cm position variation

            # Atmospheric effects
            'fog_density_range': (0.0, 0.05),
            'fog_color_temperature_range': (2000, 12000),

            # Camera effects
            'motion_blur_probability': 0.1,
            'chromatic_aberration_probability': 0.15,
            'lens_distortion_probability': 0.2,
        }

    def randomize_advanced_features(self, stage):
        """Apply advanced domain randomization features"""
        # Apply texture randomization
        self.randomize_textures(stage)

        # Apply geometric randomization
        self.randomize_geometry(stage)

        # Apply atmospheric effects
        self.randomize_atmosphere(stage)

        # Apply camera effects
        self.apply_camera_effects()

    def randomize_textures(self, stage):
        """Randomize textures on objects"""
        if np.random.random() > self.advanced_settings['texture_change_probability']:
            return

        # Get all materials with textures
        materials = [prim for prim in stage.TraverseAll() if prim.GetTypeName() == "Material"]

        for material in materials:
            shader = self.get_shader_for_material(material)
            if shader:
                # Randomize texture scale
                scale_factor = np.random.uniform(
                    self.advanced_settings['texture_scale_range'][0],
                    self.advanced_settings['texture_scale_range'][1]
                )

                # Randomize texture rotation
                rotation = np.random.uniform(
                    self.advanced_settings['texture_rotation_range'][0],
                    self.advanced_settings['texture_rotation_range'][1]
                )

    def randomize_geometry(self, stage):
        """Randomize object geometry parameters"""
        # Get all mesh objects
        meshes = [prim for prim in stage.TraverseAll() if prim.GetTypeName() == "Mesh"]

        for mesh in meshes:
            # Skip environment objects
            path = mesh.GetPath().pathString
            if any(skip in path.lower() for skip in ["floor", "wall", "environment", "ground"]):
                continue

            # Randomize scale
            current_scale = mesh.GetAttribute("xformOp:scale").Get()
            if not current_scale:
                current_scale = Gf.Vec3f(1, 1, 1)

            scale_jitter = np.random.uniform(
                1 - self.advanced_settings['object_scale_jitter'],
                1 + self.advanced_settings['object_scale_jitter'],
                size=3
            )

            new_scale = Gf.Vec3f(
                current_scale[0] * scale_jitter[0],
                current_scale[1] * scale_jitter[1],
                current_scale[2] * scale_jitter[2]
            )
            mesh.GetAttribute("xformOp:scale").Set(new_scale)

    def randomize_atmosphere(self, stage):
        """Randomize atmospheric conditions"""
        # This would typically involve setting up volumetric effects
        # in the rendering pipeline
        fog_density = np.random.uniform(
            self.advanced_settings['fog_density_range'][0],
            self.advanced_settings['fog_density_range'][1]
        )

        # Apply fog based on renderer capabilities
        # Implementation would depend on specific rendering setup
        pass
```

## Synthetic Data for Different Modalities

### 1. RGB Image Synthesis
```python
def synthesize_rgb_images(robot_config, environment_config, num_samples=1000):
    """Generate synthetic RGB images with various conditions"""
    rgb_samples = []

    for i in range(num_samples):
        # Randomize scene
        randomize_scene_for_rgb(robot_config, environment_config)

        # Capture image
        rgb_image = capture_rgb_image()

        # Apply post-processing effects
        rgb_processed = apply_camera_effects(rgb_image)

        rgb_samples.append({
            'image': rgb_processed,
            'metadata': get_scene_metadata(),
            'sample_id': i
        })

    return rgb_samples

def apply_camera_effects(image):
    """Apply realistic camera effects to synthetic images"""
    # Apply lens distortion
    image = add_lens_distortion(image)

    # Add chromatic aberration
    image = add_chromatic_aberration(image)

    # Add motion blur
    image = add_motion_blur(image)

    # Add realistic noise
    image = add_realistic_noise(image)

    # Adjust color balance
    image = adjust_color_balance(image)

    return image
```

### 2. Depth Map Synthesis
```python
def synthesize_depth_maps(robot_config, environment_config, num_samples=1000):
    """Generate synthetic depth maps with realistic noise patterns"""
    depth_samples = []

    for i in range(num_samples):
        # Randomize scene
        randomize_scene_for_depth(robot_config, environment_config)

        # Capture depth
        depth_map = capture_depth_map()

        # Add realistic depth noise
        depth_noisy = add_depth_noise(depth_map)

        depth_samples.append({
            'depth': depth_noisy,
            'metadata': get_depth_metadata(),
            'sample_id': i
        })

    return depth_samples

def add_depth_noise(depth_map):
    """Add realistic noise patterns to depth maps"""
    # Add Gaussian noise based on distance
    base_noise = np.random.normal(0, 0.001, depth_map.shape)  # 1mm at 1m

    # Scale noise with distance (noise increases with distance)
    distance_scaled_noise = base_noise * (depth_map / 10.0)  # Scale with distance

    # Add quantization effects
    depth_quantized = np.round((depth_map + distance_scaled_noise) * 1000) / 1000.0

    return depth_quantized
```

### 3. Point Cloud Synthesis
```python
def synthesize_point_clouds(lidar_config, environment_config, num_samples=500):
    """Generate synthetic point clouds from LiDAR simulation"""
    pc_samples = []

    for i in range(num_samples):
        # Randomize environment
        randomize_environment_for_lidar(environment_config)

        # Generate point cloud
        point_cloud = simulate_lidar_scan(lidar_config)

        # Add realistic LiDAR noise
        noisy_pc = add_lidar_noise(point_cloud, lidar_config)

        pc_samples.append({
            'point_cloud': noisy_pc,
            'metadata': get_lidar_metadata(lidar_config),
            'sample_id': i
        })

    return pc_samples

def add_lidar_noise(point_cloud, lidar_config):
    """Add realistic noise to simulated LiDAR point clouds"""
    # Add range noise (distance-dependent)
    range_noise_std = lidar_config['range_accuracy']
    range_noise = np.random.normal(0, range_noise_std, point_cloud.shape[0])

    # Add angular noise
    angular_noise_std = lidar_config['angular_accuracy']
    angular_noise_azimuth = np.random.normal(0, angular_noise_std, point_cloud.shape[0])
    angular_noise_elevation = np.random.normal(0, angular_noise_std, point_cloud.shape[0])

    # Apply noise to points
    points_with_noise = point_cloud.copy()

    # Convert to spherical coordinates to apply angular noise
    for j, point in enumerate(points_with_noise):
        x, y, z = point
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) if r != 0 else 0  # elevation
        phi = np.arctan2(y, x)    # azimuth

        # Apply noise
        r_noisy = r + range_noise[j]
        theta_noisy = theta + angular_noise_elevation[j]
        phi_noisy = phi + angular_noise_azimuth[j]

        # Convert back to Cartesian
        points_with_noise[j][0] = r_noisy * np.sin(theta_noisy) * np.cos(phi_noisy)
        points_with_noise[j][1] = r_noisy * np.sin(theta_noisy) * np.sin(phi_noisy)
        points_with_noise[j][2] = r_noisy * np.cos(theta_noisy)

    return points_with_noise
```

## Data Quality Assessment

### Synthetic-to-Real Similarity Metrics
```python
from scipy import linalg
import torch
import torch.nn.functional as F

class DataQualityAssessor:
    def __init__(self):
        self.feature_extractor = self.load_feature_extractor()

    def load_feature_extractor(self):
        """Load a pre-trained feature extractor for domain comparison"""
        # In practice, you'd load a pre-trained CNN like ResNet
        # that can extract meaningful features for comparison
        pass

    def assess_quality(self, synthetic_data, real_data):
        """Assess the quality of synthetic data compared to real data"""
        # Extract features from both datasets
        synth_features = self.extract_features(synthetic_data)
        real_features = self.extract_features(real_data)

        # Calculate various quality metrics
        metrics = {
            'fid_score': self.calculate_fid(synth_features, real_features),
            'mmd_score': self.calculate_mmd(synth_features, real_features),
            'kl_divergence': self.calculate_kl_divergence(synth_features, real_features),
            'domain_gap': self.estimate_domain_gap(synthetic_data, real_data)
        }

        return metrics

    def calculate_fid(self, synth_features, real_features):
        """Calculate Fréchet Inception Distance"""
        # Calculate mean and covariance for both sets
        mu1, sigma1 = self.calculate_statistics(synth_features)
        mu2, sigma2 = self.calculate_statistics(real_features)

        # Calculate FID
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(fid)

    def calculate_statistics(self, features):
        """Calculate mean and covariance of features"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_mmd(self, synth_features, real_features):
        """Calculate Maximum Mean Discrepancy"""
        # Implementation of MMD calculation
        # This is a simplified version - in practice, use kernel-based MMD
        mean_diff = np.mean(synth_features, axis=0) - np.mean(real_features, axis=0)
        mmd = np.sqrt(np.sum(mean_diff ** 2))
        return mmd

    def estimate_domain_gap(self, synthetic_data, real_data):
        """Estimate the domain gap between synthetic and real data"""
        # Train a domain classifier to distinguish between domains
        # The accuracy of this classifier indicates the domain gap
        # Lower accuracy means smaller domain gap (more similar domains)
        pass
```

## Isaac Sim Synthetic Data Generation Lab Exercise

### Objective
Create a complete synthetic data generation pipeline that produces photorealistic images with accurate annotations for object detection training.

### Prerequisites
- NVIDIA Isaac Sim installed with Omniverse
- Python environment with required packages
- Basic understanding of USD and Isaac Sim

### Steps

#### 1. Set Up the Environment
Create a warehouse scene with various objects:
```python
# Create warehouse scene with randomizable objects
def create_warehouse_scene():
    """Create a warehouse scene for synthetic data generation"""
    stage = omni.usd.get_context().get_stage()

    # Create warehouse structure
    create_warehouse_structure(stage)

    # Add randomizable objects
    add_randomizable_objects(stage)

    # Set up lighting
    setup_warehouse_lighting(stage)

    return stage

def create_warehouse_structure(stage):
    """Create basic warehouse structure"""
    # Floor
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.GetSizeAttr().Set(1.0)
    floor.GetXformOp().SetTranslate(Gf.Vec3f(0, 0, -0.1))
    floor.GetXformOp().SetScale(Gf.Vec3f(20, 20, 0.2))

    # Walls
    wall_height = 5.0
    wall_thickness = 0.5

    # Left wall
    left_wall = UsdGeom.Cube.Define(stage, "/World/Walls/LeftWall")
    left_wall.GetSizeAttr().Set(1.0)
    left_wall.GetXformOp().SetTranslate(Gf.Vec3f(-10 - wall_thickness/2, 0, wall_height/2))
    left_wall.GetXformOp().SetScale(Gf.Vec3f(wall_thickness, 20, wall_height))

    # Right wall
    right_wall = UsdGeom.Cube.Define(stage, "/World/Walls/RightWall")
    right_wall.GetSizeAttr().Set(1.0)
    right_wall.GetXformOp().SetTranslate(Gf.Vec3f(10 + wall_thickness/2, 0, wall_height/2))
    right_wall.GetXformOp().SetScale(Gf.Vec3f(wall_thickness, 20, wall_height))

    # Front wall
    front_wall = UsdGeom.Cube.Define(stage, "/World/Walls/FrontWall")
    front_wall.GetSizeAttr().Set(1.0)
    front_wall.GetXformOp().SetTranslate(Gf.Vec3f(0, -10 - wall_thickness/2, wall_height/2))
    front_wall.GetXformOp().SetScale(Gf.Vec3f(20, wall_thickness, wall_height))

    # Back wall
    back_wall = UsdGeom.Cube.Define(stage, "/World/Walls/BackWall")
    back_wall.GetSizeAttr().Set(1.0)
    back_wall.GetXformOp().SetTranslate(Gf.Vec3f(0, 10 + wall_thickness/2, wall_height/2))
    back_wall.GetXformOp().SetScale(Gf.Vec3f(20, wall_thickness, wall_height))

def add_randomizable_objects(stage):
    """Add objects that can be randomized"""
    # Add boxes of different sizes and colors
    for i in range(20):
        box = UsdGeom.Cube.Define(stage, f"/World/Objects/Box_{i}")
        box.GetSizeAttr().Set(1.0)

        # Random position within warehouse
        x = np.random.uniform(-8, 8)
        y = np.random.uniform(-8, 8)
        z = 0.5  # Half a meter above ground

        box.GetXformOp().SetTranslate(Gf.Vec3f(x, y, z))

        # Random scale
        scale_x = np.random.uniform(0.5, 1.5)
        scale_y = np.random.uniform(0.5, 1.5)
        scale_z = np.random.uniform(0.5, 2.0)

        box.GetXformOp().SetScale(Gf.Vec3f(scale_x, scale_y, scale_z))

def setup_warehouse_lighting(stage):
    """Set up warehouse-appropriate lighting"""
    # Add overhead lights
    for i in range(5):
        for j in range(5):
            x = -8 + i * 4
            y = -8 + j * 4
            z = 4.5  # Hang from ceiling

            light = UsdLux.SphereLight.Define(stage, f"/World/Lights/Light_{i}_{j}")
            light.CreateIntensityAttr(500)
            light.GetXformOp().SetTranslate(Gf.Vec3f(x, y, z))
```

#### 2. Implement Domain Randomization
Add the domain randomization code to vary scene parameters:

```python
class WarehouseDomainRandomizer:
    def __init__(self):
        self.object_types = ["box", "cylinder", "capsule", "cone"]
        self.material_colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
        ]

    def randomize_warehouse(self, stage):
        """Randomize warehouse scene for domain randomization"""
        # Randomize object properties
        self.randomize_objects(stage)

        # Randomize lighting
        self.randomize_lighting(stage)

        # Randomize materials
        self.randomize_materials(stage)

        # Randomize camera position
        self.randomize_camera()

    def randomize_objects(self, stage):
        """Randomize object properties"""
        objects = [prim for prim in stage.TraverseAll() if "Objects" in str(prim.GetPath())]

        for obj in objects:
            # Randomize position
            x = np.random.uniform(-9, 9)
            y = np.random.uniform(-9, 9)
            z = np.random.uniform(0.3, 1.5)
            obj.GetAttribute("xformOp:translate").Set(Gf.Vec3f(x, y, z))

            # Randomize rotation
            rot_x = np.random.uniform(-np.pi, np.pi)
            rot_y = np.random.uniform(-np.pi, np.pi)
            rot_z = np.random.uniform(-np.pi, np.pi)
            obj.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(
                np.degrees(rot_x),
                np.degrees(rot_y),
                np.degrees(rot_z)
            ))

    def randomize_lighting(self, stage):
        """Randomize lighting conditions"""
        lights = [prim for prim in stage.TraverseAll() if "Lights" in str(prim.GetPath())]

        for light in lights:
            # Randomize intensity
            intensity = np.random.uniform(300, 800)
            light.GetAttribute("inputs:intensity").Set(intensity)

            # Randomize color temperature
            color_temp = np.random.uniform(4000, 7000)
            color_rgb = self.kelvin_to_rgb(color_temp)
            light.GetAttribute("inputs:color").Set(color_rgb)
```

#### 3. Generate the Dataset
Run the data generation pipeline:

```python
def run_synthetic_generation_lab():
    """Run the complete synthetic data generation lab"""
    # Initialize Isaac Sim
    simulation_app = omni.simulation.SimulationApp({"headless": False})

    try:
        # Create warehouse scene
        stage = create_warehouse_scene()

        # Initialize domain randomizer
        randomizer = WarehouseDomainRandomizer()

        # Initialize synthetic data generator
        synth_gen = SyntheticDataManager(
            output_dir="./warehouse_synthetic_dataset",
            num_samples=2000
        )

        # Generate dataset with domain randomization
        for i in range(2000):
            # Randomize scene
            randomizer.randomize_warehouse(stage)

            # Generate sample
            synth_gen.generate_sample(i)

            if i % 200 == 0:
                print(f"Generated {i}/2000 samples")

        print("Synthetic dataset generation complete!")

    finally:
        simulation_app.close()

if __name__ == "__main__":
    run_synthetic_generation_lab()
```

#### 4. Validate the Generated Data
Test the quality of the generated synthetic data:

```python
def validate_synthetic_data():
    """Validate the quality of generated synthetic data"""
    # Load a subset of generated data
    synth_data = load_generated_data_subset("./warehouse_synthetic_dataset", num_samples=100)

    # Assess quality
    assessor = DataQualityAssessor()
    quality_metrics = assessor.assess_quality(synth_data, real_data_reference)

    print("Synthetic Data Quality Report:")
    print(f"FID Score: {quality_metrics['fid_score']}")
    print(f"MMD Score: {quality_metrics['mmd_score']}")
    print(f"KL Divergence: {quality_metrics['kl_divergence']}")

    # Visualize sample images
    visualize_samples(synth_data[:5])

def load_generated_data_subset(dataset_path, num_samples):
    """Load a subset of generated data for validation"""
    # Implementation to load data subset
    pass

def visualize_samples(samples):
    """Visualize generated samples"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(samples), figsize=(15, 3))
    if len(samples) == 1:
        axes = [axes]

    for i, sample in enumerate(samples):
        axes[i].imshow(sample['image'])
        axes[i].set_title(f"Sample {sample['sample_id']}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
```

### Expected Outcome
- Complete synthetic data generation pipeline for warehouse robotics
- Domain randomization implementation to improve model robustness
- 2000+ synthetic images with accurate annotations
- Quality assessment of generated data
- Ready-to-use dataset for training perception models

## Best Practices for Synthetic Data Generation

### 1. Validation
- Always validate synthetic-to-real transfer effectiveness
- Compare synthetic and real data distributions
- Test model performance on both synthetic and real data

### 2. Quality Control
- Monitor synthetic data quality metrics
- Check for artifacts or unrealistic elements
- Ensure diversity in generated data

### 3. Efficiency
- Optimize scene complexity for generation speed
- Use appropriate rendering settings for quality/speed trade-off
- Implement caching for repeated elements

### 4. Documentation
- Document randomization parameters
- Track data generation settings
- Record quality metrics for reproducibility

## Summary

In this chapter, we've explored synthetic data generation for robotics:
- Understanding the principles and benefits of synthetic data
- Setting up Isaac Sim for synthetic data workflows
- Implementing domain randomization techniques
- Generating various types of synthetic sensor data
- Creating diverse training scenarios
- Validating synthetic-to-real transfer effectiveness
- Best practices for synthetic data generation

Synthetic data generation is a powerful technique that enables the creation of large, diverse, and perfectly labeled datasets for training robust AI models in robotics applications, significantly reducing the need for expensive real-world data collection.

## Next Steps

Continue to the next chapter: [Isaac ROS: VS-LAM Navigation](./isaac-ros-vslam-navigation.md) to learn about visual SLAM and navigation techniques using Isaac ROS.