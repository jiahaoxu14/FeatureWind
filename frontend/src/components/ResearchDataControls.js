import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ResearchDataControls.css';

const ResearchDataControls = ({ 
  onDataLoaded, 
  onVisualizationSetup, 
  onModeChange,
  dataInfo, 
  isDataLoaded,
  visualizationParams,
  setVisualizationParams,
  researchMode = false
}) => {
  const [availableFiles, setAvailableFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('tworings.tmap');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isMetricAware, setIsMetricAware] = useState(false);
  const [drMethod, setDrMethod] = useState('Precomputed');

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
      // Only use expensive t-SNE computation if explicitly requested
      const shouldRecomputeTSNE = researchMode && drMethod === 't-SNE';
      const endpoint = shouldRecomputeTSNE ? '/api/load_data_tsne' : '/api/load_data';
      
      if (shouldRecomputeTSNE) {
        console.log('⚠️ Computing fresh t-SNE with Jacobian - this will take time...');
      } else {
        console.log('⚡ Loading pre-computed .tmap data');
      }
      
      const response = await axios.post(endpoint, {
        filename: selectedFile
      });
      
      onDataLoaded(response.data);
      if (onModeChange) {
        onModeChange({ isMetricAware, drMethod });
      }
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
      const endpoint = researchMode && drMethod === 't-SNE' ? 
        '/api/setup_metric_visualization' : '/api/setup_visualization';
      
      const setupParams = {
        ...visualizationParams,
        metric_aware: isMetricAware
      };
      
      const response = await axios.post(endpoint, setupParams);
      
      onVisualizationSetup(response.data);
    } catch (error) {
      setError(error.response?.data?.error || 'Failed to setup visualization');
    } finally {
      setLoading(false);
    }
  };


  const handleParameterChange = (param, value) => {
    setVisualizationParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  return (
    <div className="research-data-controls">
      <div className="control-header">
        <h3>🔬 Research Controls</h3>
        <div className="mode-toggle">
          <button
            onClick={() => setIsMetricAware(!isMetricAware)}
            className={`mode-button ${isMetricAware ? 'active' : ''}`}
            title="Toggle metric-aware processing"
          >
            {isMetricAware ? '📐 Metric-Aware' : '📏 Standard'}
          </button>
        </div>
      </div>

      {/* Dataset Selection */}
      <div className="control-section">
        <h4>📊 Dataset</h4>
        <div className="control-group">
          <label>File:</label>
          <select
            value={selectedFile}
            onChange={(e) => setSelectedFile(e.target.value)}
            disabled={loading}
          >
            {availableFiles.map(file => (
              <option key={file} value={file}>
                {file.replace('.tmap', '')}
              </option>
            ))}
          </select>
        </div>

        {researchMode && (
          <div className="control-group">
            <label>Processing:</label>
            <select
              value={drMethod}
              onChange={(e) => setDrMethod(e.target.value)}
              disabled={loading}
            >
              <option value="Precomputed">Use .tmap gradients (Fast)</option>
              <option value="t-SNE">Recompute t-SNE + Jacobian (Slow)</option>
            </select>
          </div>
        )}

        <button
          onClick={handleLoadData}
          disabled={loading}
          className="action-button load-button"
        >
          {loading ? '⏳ Loading...' : '📂 Load Data'}
        </button>
      </div>

      {/* Visualization Parameters */}
      {isDataLoaded && (
        <div className="control-section">
          <h4>⚙️ Visualization Setup</h4>
          
          <div className="control-group">
            <label>Top-k Features:</label>
            <input
              type="range"
              min="3"
              max={Math.min(dataInfo?.num_features || 10, 15)}
              value={visualizationParams.k || 5}
              onChange={(e) => handleParameterChange('k', parseInt(e.target.value))}
            />
            <span className="param-value">{visualizationParams.k || 5}</span>
          </div>

          <div className="control-group">
            <label>Grid Resolution:</label>
            <input
              type="range"
              min="10"
              max="60"
              value={visualizationParams.grid_res || 30}
              onChange={(e) => handleParameterChange('grid_res', parseInt(e.target.value))}
            />
            <span className="param-value">{visualizationParams.grid_res || 30}</span>
          </div>

          <div className="control-group">
            <label>KDTree Scale:</label>
            <input
              type="range"
              min="0.01"
              max="0.1"
              step="0.01"
              value={visualizationParams.kdtree_scale || 0.03}
              onChange={(e) => handleParameterChange('kdtree_scale', parseFloat(e.target.value))}
            />
            <span className="param-value">{visualizationParams.kdtree_scale?.toFixed(3) || '0.030'}</span>
          </div>

          <button
            onClick={handleSetupVisualization}
            disabled={loading}
            className="action-button setup-button"
          >
            {loading ? '⏳ Setting up...' : '🎯 Setup Visualization'}
          </button>
        </div>
      )}

      {/* Data Information */}
      {dataInfo && (
        <div className="control-section">
          <h4>📋 Dataset Info</h4>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">Points:</span>
              <span className="info-value">{dataInfo.num_points}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Features:</span>
              <span className="info-value">{dataInfo.num_features}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Method:</span>
              <span className="info-value">{dataInfo.method || 'Precomputed'}</span>
            </div>
            {isMetricAware && (
              <div className="info-item">
                <span className="info-label">Processing:</span>
                <span className="info-value metric-badge">Metric-Aware</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Feature List */}
      {dataInfo && dataInfo.feature_labels && (
        <div className="control-section">
          <h4>🏷️ Features</h4>
          <div className="feature-list">
            {dataInfo.feature_labels.slice(0, 10).map((label, index) => (
              <div key={index} className="feature-item">
                <span className="feature-index">{index + 1}</span>
                <span className="feature-name">{label}</span>
              </div>
            ))}
            {dataInfo.feature_labels.length > 10 && (
              <div className="feature-item more">
                <span>... and {dataInfo.feature_labels.length - 10} more</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="error-display">
          <h4>❌ Error</h4>
          <p>{error}</p>
        </div>
      )}

      {/* Research Mode Toggle */}
      <div className="control-section">
        <div className="research-mode-info">
          <h4>🧪 Research Mode</h4>
          <p>
            {researchMode ? 
              'Advanced mathematical features enabled' : 
              'Standard visualization mode active'
            }
          </p>
          
          {researchMode && (
            <div className="processing-info">
              <h5>📋 Processing Options:</h5>
              <div className="option-info">
                <strong>Use .tmap gradients:</strong> Fast loading using pre-computed tangent vectors from .tmap files
              </div>
              <div className="option-info">
                <strong>Recompute t-SNE + Jacobian:</strong> Slow but generates fresh metric-aware gradients with full Jacobian matrices
              </div>
            </div>
          )}
          
          {isMetricAware && (
            <div className="metric-info">
              <p>📐 Using Jacobian-based metric-aware transformations</p>
              <p>⚡ GPU acceleration with PyTorch MPS</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResearchDataControls;