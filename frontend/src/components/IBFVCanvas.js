import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import './IBFVCanvas.css';

const IBFVCanvas = ({ 
  dataInfo, 
  onPointSelect, 
  selectedPoint, 
  isMetricAware = false,
  showDataPoints: initialShowDataPoints = true 
}) => {
  const canvasRef = useRef();
  const overlayRef = useRef();
  const animationRef = useRef();
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [ibfvSetup, setIbfvSetup] = useState(false);
  const [ibfvParams, setIbfvParams] = useState({
    injection_rate: 0.05,
    decay_rate: 0.98,
    advection_steps: 1
  });
  const [dataPoints, setDataPoints] = useState([]);
  const [frameCount, setFrameCount] = useState(0);
  const [showDataPoints, setShowDataPoints] = useState(initialShowDataPoints);
  
  const width = 800;
  const height = 600;

  // Setup IBFV renderer when data is available
  useEffect(() => {
    if (dataInfo && dataInfo.bounding_box) {
      setupIBFVRenderer();
      fetchDataPoints();
    }
  }, [dataInfo, isMetricAware]); // eslint-disable-line react-hooks/exhaustive-deps

  // Animation loop
  useEffect(() => {
    if (isPlaying && ibfvSetup) {
      startAnimation();
    } else {
      stopAnimation();
    }
    return () => stopAnimation();
  }, [isPlaying, ibfvSetup]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAnimation();
      setIsPlaying(false);
    };
  }, []);

  const setupIBFVRenderer = async () => {
    try {
      const response = await axios.post('/api/render/ibfv/setup', {
        metric_aware: isMetricAware,
        parameters: ibfvParams
      });
      
      if (response.data.success) {
        setIbfvSetup(true);
        console.log('IBFV renderer setup:', response.data.parameters);
      }
    } catch (error) {
      console.error('Failed to setup IBFV renderer:', error);
    }
  };

  const fetchDataPoints = async () => {
    try {
      const endpoint = isMetricAware ? '/api/get_projected_points' : '/api/get_data_points';
      const response = await axios.get(endpoint);
      setDataPoints(response.data.points || []);
    } catch (error) {
      console.error('Failed to fetch data points:', error);
    }
  };

  const renderIBFVFrame = async () => {
    if (!ibfvSetup) return;
    
    try {
      const response = await axios.get('/api/render/ibfv/realtime');
      
      if (response.data.success) {
        const canvas = canvasRef.current;
        if (!canvas) return; // Component might be unmounted
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        // Create image from base64
        const img = new Image();
        img.onload = () => {
          // Check if component is still mounted
          if (!canvasRef.current || !overlayRef.current) return;
          
          // Clear canvas
          ctx.clearRect(0, 0, width, height);
          
          // Draw IBFV texture
          ctx.drawImage(img, 0, 0, width, height);
          
          // Draw data points overlay if enabled
          if (showDataPoints) {
            drawDataPointsOverlay();
          }
          
          setFrameCount(response.data.frame_count);
        };
        img.src = response.data.frame;
      }
    } catch (error) {
      console.error('Failed to render IBFV frame:', error);
    }
  };

  const drawDataPointsOverlay = () => {
    const overlayCanvas = overlayRef.current;
    if (!overlayCanvas) return;
    
    const ctx = overlayCanvas.getContext('2d');
    if (!ctx) return;
    
    if (!dataInfo || !dataInfo.bounding_box || dataPoints.length === 0) return;
    
    const [xmin, xmax, ymin, ymax] = dataInfo.bounding_box;
    
    ctx.clearRect(0, 0, width, height);
    
    // Transform coordinates from data space to canvas space
    const scaleX = width / (xmax - xmin);
    const scaleY = height / (ymax - ymin);
    
    dataPoints.forEach((point, index) => {
      const [x, y] = point.position;
      const canvasX = (x - xmin) * scaleX;
      const canvasY = height - (y - ymin) * scaleY; // Flip Y axis
      
      // Point styling
      const isSelected = selectedPoint && selectedPoint.point_idx === index;
      const radius = isSelected ? 8 : 4;
      const color = isSelected ? '#ff6b6b' : 'rgba(255, 255, 255, 0.8)';
      const strokeColor = isSelected ? '#fff' : 'rgba(0, 0, 0, 0.5)';
      
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.stroke();
      
      // Label for selected point
      if (isSelected) {
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Point ${index}`, canvasX, canvasY - 12);
      }
    });
  };

  const handleCanvasClick = (event) => {
    if (!dataInfo || !dataInfo.bounding_box || dataPoints.length === 0) return;
    
    const rect = overlayRef.current.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    
    const [xmin, xmax, ymin, ymax] = dataInfo.bounding_box;
    const scaleX = width / (xmax - xmin);
    const scaleY = height / (ymax - ymin);
    
    // Find nearest data point
    let nearestPoint = null;
    let minDistance = Infinity;
    
    dataPoints.forEach((point, index) => {
      const [x, y] = point.position;
      const canvasX = (x - xmin) * scaleX;
      const canvasY = height - (y - ymin) * scaleY;
      
      const distance = Math.sqrt(
        Math.pow(clickX - canvasX, 2) + Math.pow(clickY - canvasY, 2)
      );
      
      if (distance < minDistance && distance < 20) { // 20px threshold
        minDistance = distance;
        nearestPoint = { ...point, point_idx: index };
      }
    });
    
    if (nearestPoint && onPointSelect) {
      onPointSelect(nearestPoint);
    }
  };

  const startAnimation = () => {
    const animate = () => {
      renderIBFVFrame();
      animationRef.current = requestAnimationFrame(animate);
    };
    animate();
  };

  const stopAnimation = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  };

  const updateIBFVParameters = async (newParams) => {
    try {
      const response = await axios.post('/api/render/ibfv/parameters', newParams);
      if (response.data.success) {
        setIbfvParams(prev => ({ ...prev, ...newParams }));
      }
    } catch (error) {
      console.error('Failed to update IBFV parameters:', error);
    }
  };

  const resetIBFVTexture = async () => {
    try {
      await axios.post('/api/render/ibfv/reset');
      if (isPlaying) {
        // Continue animation after reset
        renderIBFVFrame();
      }
    } catch (error) {
      console.error('Failed to reset IBFV texture:', error);
    }
  };

  return (
    <div className="ibfv-canvas-container">
      <div className="ibfv-controls">
        <div className="control-group">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`control-button ${isPlaying ? 'active' : ''}`}
            disabled={!ibfvSetup}
          >
            {isPlaying ? '⏸️ Pause' : '▶️ Play'} IBFV
          </button>
          
          <button
            onClick={resetIBFVTexture}
            className="control-button"
            disabled={!ibfvSetup}
          >
            🔄 Reset
          </button>
          
          <button
            onClick={() => setShowDataPoints(!showDataPoints)}
            className={`control-button ${showDataPoints ? 'active' : ''}`}
          >
            📍 Data Points
          </button>
        </div>
        
        <div className="parameter-controls">
          <div className="param-control">
            <label>Injection Rate:</label>
            <input
              type="range"
              min="0.01"
              max="0.2"
              step="0.01"
              value={ibfvParams.injection_rate}
              onChange={(e) => updateIBFVParameters({ injection_rate: parseFloat(e.target.value) })}
            />
            <span>{ibfvParams.injection_rate.toFixed(3)}</span>
          </div>
          
          <div className="param-control">
            <label>Decay Rate:</label>
            <input
              type="range"
              min="0.9"
              max="0.99"
              step="0.01"
              value={ibfvParams.decay_rate}
              onChange={(e) => updateIBFVParameters({ decay_rate: parseFloat(e.target.value) })}
            />
            <span>{ibfvParams.decay_rate.toFixed(3)}</span>
          </div>
        </div>
        
        <div className="status-info">
          <span className={`status ${ibfvSetup ? 'ready' : 'loading'}`}>
            IBFV: {ibfvSetup ? 'Ready' : 'Loading...'}
          </span>
          <span className="frame-count">Frame: {frameCount}</span>
          <span className={`metric-mode ${isMetricAware ? 'active' : ''}`}>
            {isMetricAware ? '📐 Metric-Aware' : '📏 Standard'}
          </span>
        </div>
      </div>
      
      <div className="canvas-wrapper">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="ibfv-canvas"
        />
        <canvas
          ref={overlayRef}
          width={width}
          height={height}
          className="ibfv-overlay"
          onClick={handleCanvasClick}
        />
      </div>
      
      {!ibfvSetup && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Setting up IBFV renderer...</p>
        </div>
      )}
    </div>
  );
};

export default IBFVCanvas;