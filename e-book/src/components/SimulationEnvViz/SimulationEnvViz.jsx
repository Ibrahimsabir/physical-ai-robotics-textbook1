import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import styles from './SimulationEnvViz.module.css';

// Simulation Environment Visualization Component
const SimulationEnvViz = ({
  title = 'Simulation Environment Visualization',
  description = 'Interactive visualization of simulation environments',
  environmentType = 'gazebo' // 'gazebo', 'unity', or 'combined'
}) => {
  const [activeView, setActiveView] = useState('overview');
  const [isSimulating, setIsSimulating] = useState(false);
  const [environmentData, setEnvironmentData] = useState({
    robots: [],
    sensors: [],
    obstacles: [],
    lighting: {}
  });

  // Simulate environment data
  useEffect(() => {
    if (isSimulating) {
      const interval = setInterval(() => {
        setEnvironmentData(prev => ({
          ...prev,
          robots: [
            { id: 1, name: 'Robot 1', x: Math.random() * 10, y: Math.random() * 10, theta: Math.random() * 2 * Math.PI },
            { id: 2, name: 'Robot 2', x: Math.random() * 10, y: Math.random() * 10, theta: Math.random() * 2 * Math.PI }
          ],
          sensors: [
            { id: 1, type: 'lidar', range: 5 + Math.random() * 2, resolution: 0.1 },
            { id: 2, type: 'camera', fov: 60 + Math.random() * 30, range: 10 }
          ],
          obstacles: Array.from({ length: 5 }, (_, i) => ({
            id: i,
            x: Math.random() * 10,
            y: Math.random() * 10,
            width: 0.5 + Math.random() * 0.5,
            height: 0.5 + Math.random() * 0.5
          })),
          lighting: {
            intensity: 0.5 + Math.random() * 0.5,
            color: `hsl(${Math.random() * 360}, 70%, 80%)`
          }
        }));
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [isSimulating]);

  const toggleSimulation = () => {
    setIsSimulating(!isSimulating);
  };

  const renderOverview = () => (
    <div className={styles.overviewGrid}>
      <div className={styles.overviewCard}>
        <h4>Environment Type</h4>
        <p className={styles.envType}>
          {environmentType === 'gazebo' && 'Gazebo Physics Simulation'}
          {environmentType === 'unity' && 'Unity Rendering Environment'}
          {environmentType === 'combined' && 'Combined Gazebo/Unity'}
        </p>
      </div>
      <div className={styles.overviewCard}>
        <h4>Active Robots</h4>
        <p className={styles.statValue}>{environmentData.robots.length}</p>
      </div>
      <div className={styles.overviewCard}>
        <h4>Sensor Types</h4>
        <p className={styles.statValue}>{environmentData.sensors.length}</p>
      </div>
      <div className={styles.overviewCard}>
        <h4>Obstacles</h4>
        <p className={styles.statValue}>{environmentData.obstacles.length}</p>
      </div>
    </div>
  );

  const render2DView = () => (
    <div className={styles.twoDView}>
      <svg width="100%" height="400" viewBox="0 0 10 10" className={styles.simulationSvg}>
        {/* Grid background */}
        <defs>
          <pattern id="grid" width="0.5" height="0.5" patternUnits="userSpaceOnUse">
            <path d="M 0.5 0 L 0 0 0 0.5" fill="none" stroke="#e0e0e0" strokeWidth="0.01"/>
          </pattern>
        </defs>
        <rect width="10" height="10" fill="url(#grid)" />

        {/* Obstacles */}
        {environmentData.obstacles.map(obs => (
          <rect
            key={`obs-${obs.id}`}
            x={obs.x - obs.width/2}
            y={obs.y - obs.height/2}
            width={obs.width}
            height={obs.height}
            fill="#8B4513"
            opacity="0.7"
          />
        ))}

        {/* Robots */}
        {environmentData.robots.map(robot => (
          <g key={`robot-${robot.id}`}>
            <circle
              cx={robot.x}
              cy={robot.y}
              r="0.3"
              fill="#1E90FF"
              opacity="0.8"
            />
            <line
              x1={robot.x}
              y1={robot.y}
              x2={robot.x + 0.4 * Math.cos(robot.theta)}
              y2={robot.y + 0.4 * Math.sin(robot.theta)}
              stroke="#FF4500"
              strokeWidth="0.05"
            />
            <text
              x={robot.x}
              y={robot.y + 0.5}
              textAnchor="middle"
              fontSize="0.2"
              fill="#333"
            >
              {robot.name}
            </text>
          </g>
        ))}

        {/* Sensor ranges */}
        {environmentData.sensors.map((sensor, idx) => {
          if (sensor.type === 'lidar' && environmentData.robots.length > 0) {
            const robot = environmentData.robots[0]; // Show for first robot
            return (
              <circle
                key={`sensor-${sensor.id}`}
                cx={robot.x}
                cy={robot.y}
                r={sensor.range}
                fill="none"
                stroke="#32CD32"
                strokeWidth="0.02"
                strokeDasharray="0.1,0.1"
                opacity="0.5"
              />
            );
          }
          return null;
        })}
      </svg>
    </div>
  );

  const render3DView = () => (
    <div className={styles.threeDView}>
      <div className={styles.sceneContainer}>
        <div className={clsx(styles.sceneElement, styles.floor)}>
          <div className={styles.floorPattern}></div>
        </div>

        {environmentData.obstacles.map(obs => (
          <div
            key={`3d-obs-${obs.id}`}
            className={clsx(styles.sceneElement, styles.obstacle)}
            style={{
              left: `${(obs.x / 10) * 100}%`,
              top: `${(obs.y / 10) * 100}%`,
              width: `${(obs.width / 10) * 100}%`,
              height: `${(obs.height / 10) * 100}%`
            }}
          ></div>
        ))}

        {environmentData.robots.map(robot => (
          <div
            key={`3d-robot-${robot.id}`}
            className={clsx(styles.sceneElement, styles.robot)}
            style={{
              left: `${(robot.x / 10) * 100}%`,
              top: `${(robot.y / 10) * 100}%`,
              transform: `rotate(${robot.theta}rad)`
            }}
          >
            <div className={styles.robotBody}></div>
            <div className={styles.robotDirection}></div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className={clsx('margin-vert--md', styles.simulationEnvContainer)}>
      <div className={styles.vizHeader}>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>

      <div className={styles.controls}>
        <div className={styles.viewSelector}>
          <button
            className={clsx('button button--sm', styles.viewButton, {
              [styles.activeView]: activeView === 'overview'
            })}
            onClick={() => setActiveView('overview')}
          >
            Overview
          </button>
          <button
            className={clsx('button button--sm', styles.viewButton, {
              [styles.activeView]: activeView === '2d'
            })}
            onClick={() => setActiveView('2d')}
          >
            2D View
          </button>
          <button
            className={clsx('button button--sm', styles.viewButton, {
              [styles.activeView]: activeView === '3d'
            })}
            onClick={() => setActiveView('3d')}
          >
            3D View
          </button>
        </div>

        <button
          className={clsx('button', styles.simButton, {
            [styles.simActive]: isSimulating
          })}
          onClick={toggleSimulation}
        >
          {isSimulating ? 'Stop Simulation' : 'Start Simulation'}
        </button>
      </div>

      <div className={styles.visualizationArea}>
        {activeView === 'overview' && renderOverview()}
        {activeView === '2d' && render2DView()}
        {activeView === '3d' && render3DView()}
      </div>

      {isSimulating && (
        <div className={styles.simulationStats}>
          <div className={styles.statRow}>
            <span className={styles.statLabel}>Simulation Status:</span>
            <span className={styles.statValue}>Running</span>
          </div>
          <div className={styles.statRow}>
            <span className={styles.statLabel}>Real-time Factor:</span>
            <span className={styles.statValue}>1.0x</span>
          </div>
          <div className={styles.statRow}>
            <span className={styles.statLabel}>Physics Updates:</span>
            <span className={styles.statValue}>1000 Hz</span>
          </div>
        </div>
      )}

      <div className={styles.explanation}>
        <h4>About {environmentType === 'gazebo' ? 'Gazebo' : environmentType === 'unity' ? 'Unity' : 'Combined'} Simulation:</h4>
        <p>
          {environmentType === 'gazebo' && 'Gazebo provides accurate physics simulation with realistic collision detection, sensor modeling, and dynamics. It\'s ideal for validating robot behaviors in a physics-accurate environment.'}
          {environmentType === 'unity' && 'Unity offers high-quality rendering and visualization capabilities, perfect for creating realistic environments and human-robot interaction interfaces.'}
          {environmentType === 'combined' && 'Combining Gazebo and Unity provides the best of both worlds: accurate physics simulation with high-quality visualization for comprehensive robotic system development.'}
        </p>
      </div>
    </div>
  );
};

export default SimulationEnvViz;