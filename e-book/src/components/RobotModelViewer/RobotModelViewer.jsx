import React, { useState, useEffect, useRef } from 'react';
import clsx from 'clsx';
import styles from './RobotModelViewer.module.css';

// 3D Robot Model Viewer Component using CSS 3D transforms
const RobotModelViewer = ({
  title = '3D Robot Model Viewer',
  description = 'Interactive 3D visualization of humanoid robot models',
  robotType = 'humanoid',
  initialPosition = { x: 0, y: 0, z: 0 },
  initialRotation = { x: 0, y: 0, z: 0 }
}) => {
  const [isRotating, setIsRotating] = useState(false);
  const [rotation, setRotation] = useState(initialRotation);
  const [position, setPosition] = useState(initialPosition);
  const [viewMode, setViewMode] = useState('isometric'); // isometric, front, side, top
  const [animationType, setAnimationType] = useState('none'); // none, walk, gesture, balance
  const containerRef = useRef(null);

  // Animation loop for continuous rotation
  useEffect(() => {
    let animationFrame;

    if (isRotating) {
      const animate = () => {
        setRotation(prev => ({
          ...prev,
          y: prev.y + 1 // Rotate around Y axis
        }));
        animationFrame = requestAnimationFrame(animate);
      };
      animate();
    }

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [isRotating]);

  // Animation based on animationType
  useEffect(() => {
    if (animationType === 'walk') {
      const walkInterval = setInterval(() => {
        setRotation(prev => ({
          ...prev,
          x: Math.sin(Date.now() / 300) * 5, // Subtle forward/back tilt
          z: Math.sin(Date.now() / 200) * 2  // Subtle side tilt
        }));
      }, 50);

      return () => clearInterval(walkInterval);
    } else if (animationType === 'gesture') {
      const gestureInterval = setInterval(() => {
        setRotation(prev => ({
          ...prev,
          z: Math.sin(Date.now() / 500) * 15 // Arm gesture
        }));
      }, 50);

      return () => clearInterval(gestureInterval);
    } else if (animationType === 'balance') {
      const balanceInterval = setInterval(() => {
        setRotation(prev => ({
          ...prev,
          x: (Math.random() - 0.5) * 3, // Random small tilt
          z: (Math.random() - 0.5) * 3
        }));
      }, 100);

      return () => clearInterval(balanceInterval);
    }
  }, [animationType]);

  const handleRotationChange = (axis, delta) => {
    setRotation(prev => ({
      ...prev,
      [axis]: prev[axis] + delta
    }));
  };

  const handlePositionChange = (axis, delta) => {
    setPosition(prev => ({
      ...prev,
      [axis]: prev[axis] + delta
    }));
  };

  const resetView = () => {
    setRotation(initialRotation);
    setPosition(initialPosition);
    setAnimationType('none');
  };

  // Render different robot types
  const renderRobotModel = () => {
    switch(robotType) {
      case 'humanoid':
        return <HumanoidRobot rotation={rotation} />;
      case 'wheeled':
        return <WheeledRobot rotation={rotation} />;
      case 'quadruped':
        return <QuadrupedRobot rotation={rotation} />;
      default:
        return <HumanoidRobot rotation={rotation} />;
    }
  };

  // Get CSS transform based on view mode
  const getTransformForViewMode = () => {
    switch(viewMode) {
      case 'front':
        return `rotateX(-10deg) rotateY(0deg) rotateZ(0deg) translateZ(0)`;
      case 'side':
        return `rotateX(-10deg) rotateY(90deg) rotateZ(0deg) translateZ(0)`;
      case 'top':
        return `rotateX(-90deg) rotateY(0deg) rotateZ(0deg) translateZ(0)`;
      case 'isometric':
      default:
        return `rotateX(${rotation.x}deg) rotateY(${rotation.y}deg) rotateZ(${rotation.z}deg) translateZ(0)`;
    }
  };

  return (
    <div className={clsx('margin-vert--md', styles.robotViewerContainer)}>
      <div className={styles.viewerHeader}>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>

      <div className={styles.viewerControls}>
        <div className={styles.controlGroup}>
          <label>View Mode:</label>
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value)}
            className={styles.controlSelect}
          >
            <option value="isometric">Isometric</option>
            <option value="front">Front</option>
            <option value="side">Side</option>
            <option value="top">Top</option>
          </select>
        </div>

        <div className={styles.controlGroup}>
          <label>Animation:</label>
          <select
            value={animationType}
            onChange={(e) => setAnimationType(e.target.value)}
            className={styles.controlSelect}
          >
            <option value="none">None</option>
            <option value="walk">Walking</option>
            <option value="gesture">Gesturing</option>
            <option value="balance">Balancing</option>
          </select>
        </div>

        <button
          className={clsx('button button--secondary', styles.controlButton)}
          onClick={() => setIsRotating(!isRotating)}
        >
          {isRotating ? 'Stop Rotation' : 'Auto-Rotate'}
        </button>

        <button
          className={clsx('button button--secondary', styles.controlButton)}
          onClick={resetView}
        >
          Reset View
        </button>
      </div>

      <div className={styles.viewer3DContainer} ref={containerRef}>
        <div
          className={styles.robotModelWrapper}
          style={{
            transform: getTransformForViewMode(),
            transition: isRotating ? 'none' : 'transform 0.3s ease'
          }}
        >
          {renderRobotModel()}
        </div>
      </div>

      <div className={styles.viewerInfo}>
        <div className={styles.rotationInfo}>
          <div>Rotation: X: {rotation.x.toFixed(1)}° Y: {rotation.y.toFixed(1)}° Z: {rotation.z.toFixed(1)}°</div>
          <div>Position: X: {position.x.toFixed(1)} Y: {position.y.toFixed(1)} Z: {position.z.toFixed(1)}</div>
        </div>

        <div className={styles.manualControls}>
          <div className={styles.axisControls}>
            <div>
              <button onClick={() => handleRotationChange('x', -5)} className={styles.rotationButton}>X-</button>
              <button onClick={() => handleRotationChange('x', 5)} className={styles.rotationButton}>X+</button>
              <span>X Axis</span>
            </div>
            <div>
              <button onClick={() => handleRotationChange('y', -5)} className={styles.rotationButton}>Y-</button>
              <button onClick={() => handleRotationChange('y', 5)} className={styles.rotationButton}>Y+</button>
              <span>Y Axis</span>
            </div>
            <div>
              <button onClick={() => handleRotationChange('z', -5)} className={styles.rotationButton}>Z-</button>
              <button onClick={() => handleRotationChange('z', 5)} className={styles.rotationButton}>Z+</button>
              <span>Z Axis</span>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.viewerExplanation}>
        <h4>About 3D Robot Visualization</h4>
        <p>
          This interactive 3D viewer demonstrates how humanoid robots can be visualized in educational content.
          The model shows key components like joints, limbs, and sensors that are essential for understanding
          robot kinematics and locomotion. Use the controls to rotate, animate, and explore different views of the robot.
        </p>
      </div>
    </div>
  );
};

// Humanoid Robot Component
const HumanoidRobot = ({ rotation }) => {
  return (
    <div className={styles.humanoidRobot}>
      {/* Robot Head */}
      <div className={clsx(styles.robotPart, styles.head)}>
        <div className={styles.headMain}></div>
        <div className={styles.camera}></div>
        <div className={styles.camera}></div>
      </div>

      {/* Robot Torso */}
      <div className={clsx(styles.robotPart, styles.torso)}>
        <div className={styles.torsoMain}></div>
        <div className={styles.sensorArray}></div>
      </div>

      {/* Robot Arms */}
      <div className={clsx(styles.robotPart, styles.leftArm)}>
        <div className={styles.armUpper}></div>
        <div className={styles.armLower}></div>
        <div className={styles.hand}></div>
      </div>
      <div className={clsx(styles.robotPart, styles.rightArm)}>
        <div className={styles.armUpper}></div>
        <div className={styles.armLower}></div>
        <div className={styles.hand}></div>
      </div>

      {/* Robot Legs */}
      <div className={clsx(styles.robotPart, styles.leftLeg)}>
        <div className={styles.legUpper}></div>
        <div className={styles.legLower}></div>
        <div className={styles.foot}></div>
      </div>
      <div className={clsx(styles.robotPart, styles.rightLeg)}>
        <div className={styles.legUpper}></div>
        <div className={styles.legLower}></div>
        <div className={styles.foot}></div>
      </div>

      {/* Joints */}
      <div className={clsx(styles.joint, styles.neckJoint)}></div>
      <div className={clsx(styles.joint, styles.shoulderLeft)}></div>
      <div className={clsx(styles.joint, styles.shoulderRight)}></div>
      <div className={clsx(styles.joint, styles.elbowLeft)}></div>
      <div className={clsx(styles.joint, styles.elbowRight)}></div>
      <div className={clsx(styles.joint, styles.hipLeft)}></div>
      <div className={clsx(styles.joint, styles.hipRight)}></div>
      <div className={clsx(styles.joint, styles.kneeLeft)}></div>
      <div className={clsx(styles.joint, styles.kneeRight)}></div>
      <div className={clsx(styles.joint, styles.ankleLeft)}></div>
      <div className={clsx(styles.joint, styles.ankleRight)}></div>
    </div>
  );
};

// Wheeled Robot Component
const WheeledRobot = ({ rotation }) => {
  return (
    <div className={styles.wheeledRobot}>
      {/* Robot Body */}
      <div className={clsx(styles.robotPart, styles.body)}>
        <div className={styles.bodyMain}></div>
        <div className={styles.sensorArray}></div>
      </div>

      {/* Wheels */}
      <div className={clsx(styles.robotPart, styles.leftWheel)}>
        <div className={styles.wheel}></div>
      </div>
      <div className={clsx(styles.robotPart, styles.rightWheel)}>
        <div className={styles.wheel}></div>
      </div>

      {/* Additional wheels for 4WD */}
      <div className={clsx(styles.robotPart, styles.frontWheel)}>
        <div className={styles.wheel}></div>
      </div>
      <div className={clsx(styles.robotPart, styles.rearWheel)}>
        <div className={styles.wheel}></div>
      </div>

      {/* Camera/Sensor */}
      <div className={clsx(styles.robotPart, styles.cameraMount)}>
        <div className={styles.camera}></div>
      </div>
    </div>
  );
};

// Quadruped Robot Component
const QuadrupedRobot = ({ rotation }) => {
  return (
    <div className={styles.quadrupedRobot}>
      {/* Robot Body */}
      <div className={clsx(styles.robotPart, styles.body)}>
        <div className={styles.bodyMain}></div>
        <div className={styles.head}></div>
      </div>

      {/* Legs */}
      <div className={clsx(styles.robotPart, styles.frontLeftLeg)}>
        <div className={styles.legUpper}></div>
        <div className={styles.legLower}></div>
        <div className={styles.foot}></div>
      </div>
      <div className={clsx(styles.robotPart, styles.frontRightLeg)}>
        <div className={styles.legUpper}></div>
        <div className={styles.legLower}></div>
        <div className={styles.foot}></div>
      </div>
      <div className={clsx(styles.robotPart, styles.rearLeftLeg)}>
        <div className={styles.legUpper}></div>
        <div className={styles.legLower}></div>
        <div className={styles.foot}></div>
      </div>
      <div className={clsx(styles.robotPart, styles.rearRightLeg)}>
        <div className={styles.legUpper}></div>
        <div className={styles.legLower}></div>
        <div className={styles.foot}></div>
      </div>

      {/* Joints */}
      <div className={clsx(styles.joint, styles.shoulderFL)}></div>
      <div className={clsx(styles.joint, styles.shoulderFR)}></div>
      <div className={clsx(styles.joint, styles.hipRL)}></div>
      <div className={clsx(styles.joint, styles.hipRR)}></div>
    </div>
  );
};

export default RobotModelViewer;