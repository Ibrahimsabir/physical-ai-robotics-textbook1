import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import styles from './InteractiveDemo.module.css';

// Interactive ROS 2 Communication Diagram Component
const InteractiveDemo = ({ title, description, type = 'publisher-subscriber' }) => {
  const [activeDemo, setActiveDemo] = useState(false);
  const [messages, setMessages] = useState([]);
  const [nodeState, setNodeState] = useState({
    publisher: { status: 'idle', messageCount: 0 },
    subscriber: { status: 'waiting', messageCount: 0 }
  });

  // Simulate ROS 2 communication
  useEffect(() => {
    let interval;
    if (activeDemo && type === 'publisher-subscriber') {
      interval = setInterval(() => {
        const newMessage = {
          id: Date.now(),
          content: `Message ${nodeState.publisher.messageCount + 1}`,
          timestamp: new Date().toLocaleTimeString(),
          from: 'publisher',
          to: 'subscriber'
        };

        setMessages(prev => [...prev.slice(-4), newMessage]); // Keep only last 5 messages

        setNodeState(prev => ({
          ...prev,
          publisher: {
            ...prev.publisher,
            status: 'publishing',
            messageCount: prev.publisher.messageCount + 1
          },
          subscriber: {
            ...prev.subscriber,
            status: 'receiving',
            messageCount: prev.subscriber.messageCount + 1
          }
        }));

        // Reset status after a short delay
        setTimeout(() => {
          setNodeState(prev => ({
            ...prev,
            publisher: { ...prev.publisher, status: 'idle' },
            subscriber: { ...prev.subscriber, status: 'waiting' }
          }));
        }, 500);
      }, 2000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [activeDemo, type, nodeState.publisher.messageCount, nodeState.subscriber.messageCount]);

  const toggleDemo = () => {
    if (activeDemo) {
      // Reset when stopping
      setMessages([]);
      setNodeState({
        publisher: { status: 'idle', messageCount: 0 },
        subscriber: { status: 'waiting', messageCount: 0 }
      });
    }
    setActiveDemo(!activeDemo);
  };

  return (
    <div className={clsx('margin-vert--md', styles.interactiveDemoContainer)}>
      <div className={styles.demoHeader}>
        <h3>{title || 'ROS 2 Communication Demo'}</h3>
        <p>{description || 'Interactive demonstration of ROS 2 communication patterns'}</p>
      </div>

      <div className={styles.demoVisualization}>
        {/* Publisher Node */}
        <div className={clsx(styles.node, styles.publisherNode, {
          [styles.activeNode]: nodeState.publisher.status !== 'idle'
        })}>
          <div className={styles.nodeHeader}>
            <span className={styles.nodeStatus}>
              {nodeState.publisher.status === 'idle' && '🟢'}
              {nodeState.publisher.status === 'publishing' && '🔴'}
            </span>
            <strong>Publisher Node</strong>
          </div>
          <div className={styles.nodeInfo}>
            <div>Status: {nodeState.publisher.status}</div>
            <div>Messages: {nodeState.publisher.messageCount}</div>
          </div>
        </div>

        {/* Communication Arrow */}
        <div className={styles.communicationArrow}>
          <div className={styles.arrowHead}>/topics</div>
          <div className={clsx(styles.arrow, {
            [styles.activeArrow]: nodeState.publisher.status === 'publishing'
          })}>
            <div className={styles.arrowBody}></div>
            <div className={styles.arrowTip}></div>
          </div>
        </div>

        {/* Subscriber Node */}
        <div className={clsx(styles.node, styles.subscriberNode, {
          [styles.activeNode]: nodeState.subscriber.status !== 'waiting'
        })}>
          <div className={styles.nodeHeader}>
            <span className={styles.nodeStatus}>
              {nodeState.subscriber.status === 'waiting' && '🟡'}
              {nodeState.subscriber.status === 'receiving' && '🔴'}
            </span>
            <strong>Subscriber Node</strong>
          </div>
          <div className={styles.nodeInfo}>
            <div>Status: {nodeState.subscriber.status}</div>
            <div>Messages: {nodeState.subscriber.messageCount}</div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className={styles.demoControls}>
        <button
          className={clsx('button button--primary', styles.demoButton)}
          onClick={toggleDemo}
        >
          {activeDemo ? 'Stop Demo' : 'Start Demo'}
        </button>
      </div>

      {/* Message Log */}
      {messages.length > 0 && (
        <div className={styles.messageLog}>
          <h4>Message Log</h4>
          <div className={styles.logContainer}>
            {messages.map(msg => (
              <div key={msg.id} className={styles.logEntry}>
                <span className={styles.timestamp}>[{msg.timestamp}]</span>
                <span className={styles.messageContent}>{msg.content}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Explanation */}
      <div className={styles.explanation}>
        <h4>How it works:</h4>
        <p>
          In ROS 2's publisher-subscriber pattern, the publisher node sends messages to a topic,
          and any subscriber nodes listening to that topic receive the messages. This is an
          asynchronous, many-to-many communication pattern ideal for streaming data like sensor readings.
        </p>
      </div>
    </div>
  );
};

export default InteractiveDemo;