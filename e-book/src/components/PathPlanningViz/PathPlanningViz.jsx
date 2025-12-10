import React, { useState, useEffect, useRef } from 'react';
import clsx from 'clsx';
import styles from './PathPlanningViz.module.css';

const PathPlanningViz = ({ title = 'Path Planning Visualization', description = 'Interactive visualization of robotic path planning algorithms' }) => {
  const canvasRef = useRef(null);
  const [algorithm, setAlgorithm] = useState('astar'); // astar, dijkstra, rrt
  const [isSimulating, setIsSimulating] = useState(false);
  const [startPoint, setStartPoint] = useState({ x: 50, y: 50 });
  const [endPoint, setEndPoint] = useState({ x: 400, y: 300 });
  const [obstacles, setObstacles] = useState([
    { x: 150, y: 100, width: 80, height: 40 },
    { x: 250, y: 200, width: 60, height: 80 },
    { x: 100, y: 250, width: 100, height: 30 }
  ]);
  const [path, setPath] = useState([]);

  // Sample path for visualization
  useEffect(() => {
    if (isSimulating) {
      // Simulate path planning algorithm
      const simulatedPath = [];
      for (let i = 0; i <= 20; i++) {
        const t = i / 20;
        simulatedPath.push({
          x: startPoint.x + t * (endPoint.x - startPoint.x),
          y: startPoint.y + t * (endPoint.y - startPoint.y)
        });
      }
      setPath(simulatedPath);
    } else {
      setPath([]);
    }
  }, [isSimulating, startPoint, endPoint]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background grid
    drawGrid(ctx, canvas.width, canvas.height);

    // Draw obstacles
    obstacles.forEach(obstacle => {
      ctx.fillStyle = '#8B4513'; // Brown for obstacles
      ctx.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);

      // Add border
      ctx.strokeStyle = '#5D2906';
      ctx.lineWidth = 2;
      ctx.strokeRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
    });

    // Draw path if it exists
    if (path.length > 0) {
      ctx.beginPath();
      ctx.moveTo(path[0].x, path[0].y);

      for (let i = 1; i < path.length; i++) {
        ctx.lineTo(path[i].x, path[i].y);
      }

      ctx.strokeStyle = '#32CD32'; // Green for path
      ctx.lineWidth = 3;
      ctx.stroke();

      // Draw path points
      path.forEach(point => {
        ctx.fillStyle = '#32CD32';
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Draw start point
    ctx.fillStyle = '#1E90FF'; // Blue for start
    ctx.beginPath();
    ctx.arc(startPoint.x, startPoint.y, 10, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw "S" inside start circle
    ctx.fillStyle = '#FFF';
    ctx.font = 'bold 12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('S', startPoint.x, startPoint.y);

    // Draw end point
    ctx.fillStyle = '#FF4500'; // Orange for end
    ctx.beginPath();
    ctx.arc(endPoint.x, endPoint.y, 10, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw "G" inside end circle (Goal)
    ctx.fillStyle = '#FFF';
    ctx.fillText('G', endPoint.x, endPoint.y);
  };

  const drawGrid = (ctx, width, height) => {
    ctx.strokeStyle = '#EEEEEE';
    ctx.lineWidth = 0.5;

    // Vertical lines
    for (let x = 0; x <= width; x += 20) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = 0; y <= height; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  };

  useEffect(() => {
    drawCanvas();
  }, [startPoint, endPoint, obstacles, path, isSimulating]);

  const handleCanvasClick = (e) => {
    if (!isSimulating) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if click is on start point
    const startDist = Math.sqrt((x - startPoint.x) ** 2 + (y - startPoint.y) ** 2);
    if (startDist < 15) {
      setStartPoint({ x, y });
      return;
    }

    // Check if click is on end point
    const endDist = Math.sqrt((x - endPoint.x) ** 2 + (y - endPoint.y) ** 2);
    if (endDist < 15) {
      setEndPoint({ x, y });
      return;
    }

    // For now, just update the end point when clicking elsewhere
    setEndPoint({ x, y });
  };

  const resetSimulation = () => {
    setStartPoint({ x: 50, y: 50 });
    setEndPoint({ x: 400, y: 300 });
    setPath([]);
    setIsSimulating(false);
  };

  const addObstacle = () => {
    const newObstacle = {
      x: Math.random() * 300 + 50,
      y: Math.random() * 200 + 50,
      width: Math.random() * 60 + 30,
      height: Math.random() * 60 + 30
    };
    setObstacles([...obstacles, newObstacle]);
  };

  const clearObstacles = () => {
    setObstacles([]);
  };

  return (
    <div className={clsx('margin-vert--md', styles.container)}>
      <div className={styles.header}>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>

      <div className={styles.controls}>
        <div className={styles.algorithmSelector}>
          <label htmlFor="algorithm">Path Planning Algorithm: </label>
          <select
            id="algorithm"
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            className={styles.select}
          >
            <option value="astar">A* (A-star)</option>
            <option value="dijkstra">Dijkstra</option>
            <option value="rrt">RRT (Rapidly-exploring Random Tree)</option>
            <option value="potential">Potential Fields</option>
          </select>
        </div>

        <div className={styles.buttonGroup}>
          <button
            className={clsx('button button--primary', styles.btn)}
            onClick={() => setIsSimulating(!isSimulating)}
          >
            {isSimulating ? 'Stop Simulation' : 'Start Simulation'}
          </button>

          <button
            className={clsx('button button--secondary', styles.btn)}
            onClick={resetSimulation}
          >
            Reset
          </button>

          <button
            className={clsx('button button--outline', styles.btn)}
            onClick={addObstacle}
          >
            Add Obstacle
          </button>

          <button
            className={clsx('button button--outline', styles.btn)}
            onClick={clearObstacles}
          >
            Clear Obstacles
          </button>
        </div>
      </div>

      <div className={styles.canvasContainer}>
        <canvas
          ref={canvasRef}
          width={500}
          height={400}
          className={styles.canvas}
          onClick={handleCanvasClick}
        />
      </div>

      <div className={styles.infoPanel}>
        <div className={styles.infoItem}>
          <strong>Start Point (S):</strong> ({Math.round(startPoint.x)}, {Math.round(startPoint.y)})
        </div>
        <div className={styles.infoItem}>
          <strong>End Point (G):</strong> ({Math.round(endPoint.x)}, {Math.round(endPoint.y)})
        </div>
        <div className={styles.infoItem}>
          <strong>Algorithm:</strong> {algorithm.charAt(0).toUpperCase() + algorithm.slice(1)}
        </div>
        <div className={styles.infoItem}>
          <strong>Obstacles:</strong> {obstacles.length}
        </div>
        <div className={styles.infoItem}>
          <strong>Path Length:</strong> {path.length > 0 ? path.length : 'N/A'} steps
        </div>
      </div>

      <div className={styles.explanation}>
        <h4>Path Planning Algorithms:</h4>
        <ul>
          <li><strong>A* (A-star)</strong>: Best-first search that uses heuristics to find the shortest path efficiently</li>
          <li><strong>Dijkstra</strong>: Finds shortest path from start to all nodes, guaranteed optimal</li>
          <li><strong>RRT (Rapidly-exploring Random Tree)</strong>: Probabilistically complete, good for high-dimensional spaces</li>
          <li><strong>Potential Fields</strong>: Uses attractive and repulsive forces to navigate</li>
        </ul>

        <p>
          Path planning is essential for autonomous robots to navigate safely from a start position to a goal position
          while avoiding obstacles. The choice of algorithm depends on factors like environment complexity,
          real-time requirements, and optimality constraints.
        </p>
      </div>
    </div>
  );
};

export default PathPlanningViz;