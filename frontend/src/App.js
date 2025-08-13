import React, { useState, useEffect } from 'react';
import axios from 'axios';
import WindMapVisualization from './components/WindMapVisualization';
import WindVane from './components/WindVane';
import DataControls from './components/DataControls';
import './App.css';

function App() {
  const [isDataLoaded, setIsDataLoaded] = useState(false);
  const [isVisualizationSetup, setIsVisualizationSetup] = useState(false);
  const [dataInfo, setDataInfo] = useState(null);
  const [selectedCell, setSelectedCell] = useState({ i: 20, j: 20 });
  const [windVaneData, setWindVaneData] = useState(null);
  const [visualizationParams, setVisualizationParams] = useState({
    k: 10,
    grid_res: 40,
    kdtree_scale: 0.03
  });

  const handleDataLoaded = (info) => {
    setDataInfo(info);
    setIsDataLoaded(true);
    setIsVisualizationSetup(false);
  };

  const handleVisualizationSetup = (params) => {
    setVisualizationParams(params);
    setIsVisualizationSetup(true);
  };

  const handleCellSelect = async (cellI, cellJ) => {
    setSelectedCell({ i: cellI, j: cellJ });
    
    try {
      const response = await axios.post('/api/get_wind_vane_data', {
        cell_i: cellI,
        cell_j: cellJ
      });
      setWindVaneData(response.data);
    } catch (error) {
      console.error('Error fetching wind vane data:', error);
    }
  };

  // Load initial wind vane data when visualization is setup
  useEffect(() => {
    if (isVisualizationSetup) {
      handleCellSelect(selectedCell.i, selectedCell.j);
    }
  }, [isVisualizationSetup]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>FeatureWind Visualization</h1>
      </header>
      
      <div className="App-content">
        <div className="controls-panel">
          <DataControls 
            onDataLoaded={handleDataLoaded}
            onVisualizationSetup={handleVisualizationSetup}
            dataInfo={dataInfo}
            isDataLoaded={isDataLoaded}
            visualizationParams={visualizationParams}
            setVisualizationParams={setVisualizationParams}
          />
        </div>
        
        {isVisualizationSetup && (
          <div className="visualization-container">
            <div className="wind-map-container">
              <WindMapVisualization 
                onCellSelect={handleCellSelect}
                selectedCell={selectedCell}
                dataInfo={dataInfo}
              />
            </div>
            
            <div className="wind-vane-container">
              <WindVane 
                windVaneData={windVaneData}
                selectedCell={selectedCell}
                boundingBox={dataInfo?.bounding_box}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;