import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './DataControls.css';

const DataControls = ({ 
  onDataLoaded, 
  onVisualizationSetup, 
  dataInfo, 
  isDataLoaded,
  visualizationParams,
  setVisualizationParams
}) => {
  const [availableFiles, setAvailableFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('breast_cancer.tmap');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAvailableFiles();
  }, []);

  const fetchAvailableFiles = async () => {
    try {
      const response = await axios.get('/api/available_files');
      setAvailableFiles(response.data.files);
    } catch (error) {
      console.error('Error fetching available files:', error);
    }
  };

  const handleLoadData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/load_data', {
        filename: selectedFile
      });
      
      onDataLoaded(response.data);
    } catch (error) {
      setError(error.response?.data?.error || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const handleSetupVisualization = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/setup_visualization', visualizationParams);
      onVisualizationSetup(response.data);
    } catch (error) {
      setError(error.response?.data?.error || 'Failed to setup visualization');
    } finally {
      setLoading(false);
    }
  };

  const handleParamChange = (param, value) => {
    setVisualizationParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  return (
    <div className="data-controls">
      <div className="control-section">
        <h3>Data Loading</h3>
        <div className="control-group">
          <label>
            Select Data File:
            <select 
              value={selectedFile} 
              onChange={(e) => setSelectedFile(e.target.value)}
              disabled={loading}
            >
              {availableFiles.map(file => (
                <option key={file} value={file}>{file}</option>
              ))}
            </select>
          </label>
          <button 
            onClick={handleLoadData} 
            disabled={loading}
            className="load-btn"
          >
            {loading ? 'Loading...' : 'Load Data'}
          </button>
        </div>
        
        {dataInfo && (
          <div className="data-info">
            <p><strong>Points:</strong> {dataInfo.num_points}</p>
            <p><strong>Features:</strong> {dataInfo.num_features}</p>
          </div>
        )}
      </div>

      {isDataLoaded && (
        <div className="control-section">
          <h3>Visualization Parameters</h3>
          <div className="param-controls">
            <label>
              Number of Features (k):
              <input
                type="number"
                min="1"
                max={dataInfo?.num_features || 20}
                value={visualizationParams.k}
                onChange={(e) => handleParamChange('k', parseInt(e.target.value))}
              />
            </label>
            
            <label>
              Grid Resolution:
              <input
                type="number"
                min="10"
                max="100"
                value={visualizationParams.grid_res}
                onChange={(e) => handleParamChange('grid_res', parseInt(e.target.value))}
              />
            </label>
            
            <label>
              KDTree Scale:
              <input
                type="number"
                min="0.01"
                max="0.1"
                step="0.01"
                value={visualizationParams.kdtree_scale}
                onChange={(e) => handleParamChange('kdtree_scale', parseFloat(e.target.value))}
              />
            </label>
          </div>
          
          <button 
            onClick={handleSetupVisualization} 
            disabled={loading}
            className="setup-btn"
          >
            {loading ? 'Setting up...' : 'Setup Visualization'}
          </button>
        </div>
      )}

      {error && (
        <div className="error-message">
          Error: {error}
        </div>
      )}
    </div>
  );
};

export default DataControls;