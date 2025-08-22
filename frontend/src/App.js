import React, { useState, useEffect } from 'react';
import axios from 'axios';
import WindMapVisualization from './components/WindMapVisualization';
import IBFVCanvas from './components/IBFVCanvas';
import WindVane from './components/WindVane';
import ResearchDataControls from './components/ResearchDataControls';
import './App.css';

function App() {
  const [isDataLoaded, setIsDataLoaded] = useState(false);
  const [isVisualizationSetup, setIsVisualizationSetup] = useState(false);
  const [dataInfo, setDataInfo] = useState(null);
  const [selectedCell, setSelectedCell] = useState({ i: 20, j: 20 });
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [windVaneData, setWindVaneData] = useState(null);
  const [visualizationParams, setVisualizationParams] = useState({
    k: 5,
    grid_res: 30,
    kdtree_scale: 0.03
  });
  const [researchMode, setResearchMode] = useState(true);
  const [isMetricAware, setIsMetricAware] = useState(false);
  const [drMethod, setDrMethod] = useState('Precomputed');
  const [renderMode, setRenderMode] = useState('IBFV'); // 'IBFV' or 'Particles'
  const [showConvexHull, setShowConvexHull] = useState(true);

  const handleDataLoaded = (info) => {
    setDataInfo(info);
    setIsDataLoaded(true);
    setIsVisualizationSetup(false);
    setSelectedPoint(null);
    setWindVaneData(null);
  };

  const handleVisualizationSetup = (params) => {
    setVisualizationParams(params);
    setIsVisualizationSetup(true);
  };

  const handleModeChange = (mode) => {
    setIsMetricAware(mode.isMetricAware);
    setDrMethod(mode.drMethod);
  };

  const handleCellSelect = async (cellI, cellJ) => {
    setSelectedCell({ i: cellI, j: cellJ });
    setSelectedPoint(null); // Clear point selection when selecting cell
    
    try {
      const endpoint = isMetricAware ? '/api/get_metric_wind_vane_data' : '/api/get_wind_vane_data';
      const payload = isMetricAware ? 
        { point_idx: cellI * visualizationParams.grid_res + cellJ } : // Convert cell to point index
        { cell_i: cellI, cell_j: cellJ };
      
      const response = await axios.post(endpoint, payload);
      setWindVaneData(response.data);
    } catch (error) {
      console.error('Error fetching wind vane data:', error);
    }
  };

  const handlePointSelect = async (point) => {
    setSelectedPoint(point);
    setSelectedCell(null); // Clear cell selection when selecting point
    
    try {
      const endpoint = isMetricAware ? '/api/get_metric_wind_vane_data' : '/api/get_wind_vane_data';
      const payload = { point_idx: point.point_idx };
      
      const response = await axios.post(endpoint, payload);
      setWindVaneData(response.data);
    } catch (error) {
      console.error('Error fetching wind vane data:', error);
    }
  };

  // Load initial wind vane data when visualization is setup
  useEffect(() => {
    if (isVisualizationSetup) {
      if (isMetricAware && !selectedPoint) {
        // Select first point for metric-aware mode
        handlePointSelect({ point_idx: 0 });
      } else if (!isMetricAware && selectedCell) {
        handleCellSelect(selectedCell.i, selectedCell.j);
      }
    }
  }, [isVisualizationSetup, isMetricAware]);

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1>🌪️ FeatureWind Research Platform</h1>
          <div className="header-controls">
            <div className="render-mode-toggle">
              <button
                onClick={() => setRenderMode(renderMode === 'IBFV' ? 'Particles' : 'IBFV')}
                className={`mode-button ${renderMode === 'IBFV' ? 'active' : ''}`}
              >
                {renderMode === 'IBFV' ? '🌊 IBFV' : '⚡ Particles'}
              </button>
            </div>
            <div className="research-mode-indicator">
              <span className={`mode-indicator ${researchMode ? 'active' : ''}`}>
                {researchMode ? '🔬 Research Mode' : '📊 Standard Mode'}
              </span>
            </div>
          </div>
        </div>
      </header>
      
      <div className="App-content research-layout">
        {/* Left Panel - Research Controls */}
        <div className="controls-panel">
          <ResearchDataControls 
            onDataLoaded={handleDataLoaded}
            onVisualizationSetup={handleVisualizationSetup}
            onModeChange={handleModeChange}
            dataInfo={dataInfo}
            isDataLoaded={isDataLoaded}
            visualizationParams={visualizationParams}
            setVisualizationParams={setVisualizationParams}
            researchMode={researchMode}
          />
        </div>
        
        {/* Center Panel - Visualization */}
        {isVisualizationSetup && (
          <div className="visualization-panel">
            <div className="visualization-header">
              <h3>
                {renderMode === 'IBFV' ? '🌊 Wind Map (IBFV)' : '⚡ Wind Map (Particles)'}
              </h3>
              <div className="visualization-info">
                <span>{drMethod}</span>
                {isMetricAware && <span className="metric-badge">📐 Metric-Aware</span>}
              </div>
            </div>
            
            <div className="visualization-container">
              {renderMode === 'IBFV' ? (
                <IBFVCanvas
                  dataInfo={dataInfo}
                  onPointSelect={handlePointSelect}
                  selectedPoint={selectedPoint}
                  isMetricAware={isMetricAware}
                  showDataPoints={true}
                />
              ) : (
                <WindMapVisualization 
                  onCellSelect={handleCellSelect}
                  selectedCell={selectedCell}
                  dataInfo={dataInfo}
                />
              )}
            </div>
          </div>
        )}
        
        {/* Right Panel - Wind Vane Analysis */}
        {isVisualizationSetup && (
          <div className="analysis-panel">
            <WindVane 
              windVaneData={windVaneData}
              selectedCell={selectedCell}
              selectedPoint={selectedPoint}
              boundingBox={dataInfo?.bounding_box}
              isMetricAware={isMetricAware}
              onToggleConvexHull={() => setShowConvexHull(!showConvexHull)}
              showConvexHull={showConvexHull}
            />
          </div>
        )}
      </div>
      
      {/* Loading State */}
      {!isVisualizationSetup && isDataLoaded && (
        <div className="setup-prompt">
          <div className="setup-message">
            <h3>🎯 Ready to Visualize</h3>
            <p>Data loaded successfully. Configure parameters and click "Setup Visualization" to begin.</p>
            <div className="setup-stats">
              <span>📊 {dataInfo?.num_points} points</span>
              <span>🏷️ {dataInfo?.num_features} features</span>
              <span>⚙️ {dataInfo?.method || 'Precomputed'}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;