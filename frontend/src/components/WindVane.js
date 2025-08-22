import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import './WindVane.css';

const WindVane = ({ 
  windVaneData, 
  selectedCell, 
  boundingBox, 
  isMetricAware = false,
  onToggleConvexHull,
  showConvexHull = true 
}) => {
  const svgRef = useRef();
  const [windStrength, setWindStrength] = useState(0);
  const [anisotropy, setAnisotropy] = useState({ ratio: 1, angle: 0 });
  const width = 400;
  const height = 400;

  // Helper function to compute convex hull of wind direction endpoints
  const computeConvexHull = (vectors, scale) => {
    if (vectors.length < 3) return vectors;
    
    const points = vectors.map(v => [v.u / scale, v.v / scale]);
    
    // Simple gift wrapping convex hull algorithm
    const orientation = (p, q, r) => {
      const val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);
      if (val === 0) return 0;
      return val > 0 ? 1 : 2;
    };

    const n = points.length;
    if (n < 3) return vectors;

    const hull = [];
    let l = 0;
    
    for (let i = 1; i < n; i++) {
      if (points[i][0] < points[l][0]) l = i;
    }

    let p = l;
    do {
      hull.push(p);
      let q = (p + 1) % n;
      
      for (let i = 0; i < n; i++) {
        if (orientation(points[p], points[i], points[q]) === 2) {
          q = i;
        }
      }
      p = q;
    } while (p !== l);

    return hull.map(i => vectors[i]);
  };

  // Helper function to compute wind rose (ellipse from covariance)
  const computeWindRose = (vectors, scale) => {
    if (vectors.length < 2) return { a: 20, b: 20, angle: 0, area: 0 };
    
    const endpoints = vectors.map(v => [v.u / scale, v.v / scale]);
    const n = endpoints.length;
    
    // Compute mean
    const meanX = endpoints.reduce((sum, p) => sum + p[0], 0) / n;
    const meanY = endpoints.reduce((sum, p) => sum + p[1], 0) / n;
    
    // Compute covariance matrix
    let cxx = 0, cxy = 0, cyy = 0;
    endpoints.forEach(([x, y]) => {
      const dx = x - meanX;
      const dy = y - meanY;
      cxx += dx * dx;
      cxy += dx * dy;
      cyy += dy * dy;
    });
    cxx /= n;
    cxy /= n;
    cyy /= n;
    
    // Eigenvalues for ellipse axes
    const trace = cxx + cyy;
    const det = cxx * cyy - cxy * cxy;
    const discriminant = Math.sqrt(trace * trace - 4 * det);
    
    const lambda1 = (trace + discriminant) / 2;
    const lambda2 = (trace - discriminant) / 2;
    
    const a = Math.sqrt(Math.abs(lambda1)) * 40; // Scale for visualization
    const b = Math.sqrt(Math.abs(lambda2)) * 40;
    const angle = Math.atan2(2 * cxy, cxx - cyy) / 2;
    const area = Math.PI * a * b;
    
    return { a, b, angle: angle * 180 / Math.PI, area };
  };

  useEffect(() => {
    if (!windVaneData || !boundingBox) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const centerX = width / 2;
    const centerY = height / 2;
    const targetLength = Math.min(width, height) * 0.25;

    // Calculate dynamic scaling
    const vectorMagnitudes = windVaneData.vectors.map(v => v.magnitude);
    const maxMagnitude = Math.max(...vectorMagnitudes);
    const dynamicScale = maxMagnitude > 0 ? maxMagnitude / targetLength : 1.0;

    // Apply convex hull filtering if enabled
    const displayVectors = showConvexHull ? 
      computeConvexHull(windVaneData.vectors, dynamicScale) : 
      windVaneData.vectors;

    // Compute wind rose parameters
    const windRose = computeWindRose(windVaneData.vectors, dynamicScale);
    setWindStrength(windRose.area / 1000); // Normalize for display
    setAnisotropy({ 
      ratio: windRose.a > 0 ? windRose.b / windRose.a : 1, 
      angle: windRose.angle 
    });

    // Wind Rose Base (elliptical baseplate)
    svg.append('ellipse')
      .attr('cx', centerX)
      .attr('cy', centerY)
      .attr('rx', windRose.a)
      .attr('ry', windRose.b)
      .attr('transform', `rotate(${windRose.angle} ${centerX} ${centerY})`)
      .attr('fill', 'rgba(135, 206, 250, 0.2)')
      .attr('stroke', 'rgba(135, 206, 250, 0.8)')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,3');

    // Background compass circle
    svg.append('circle')
      .attr('cx', centerX)
      .attr('cy', centerY)
      .attr('r', targetLength * 1.3)
      .attr('fill', 'none')
      .attr('stroke', '#e0e0e0')
      .attr('stroke-width', 1)
      .attr('opacity', 0.5);

    // Compass directions (N, E, S, W)
    const compassLabels = [
      { text: 'N', x: centerX, y: centerY - targetLength * 1.5 },
      { text: 'E', x: centerX + targetLength * 1.5, y: centerY + 5 },
      { text: 'S', x: centerX, y: centerY + targetLength * 1.5 + 10 },
      { text: 'W', x: centerX - targetLength * 1.5, y: centerY + 5 }
    ];
    
    compassLabels.forEach(label => {
      svg.append('text')
        .attr('x', label.x)
        .attr('y', label.y)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('fill', '#666')
        .text(label.text);
    });

    // Center point marker
    svg.append('circle')
      .attr('cx', centerX)
      .attr('cy', centerY)
      .attr('r', 4)
      .attr('fill', isMetricAware ? '#4CAF50' : '#666')
      .attr('stroke', 'white')
      .attr('stroke-width', 2);

    // Point/cell label
    svg.append('text')
      .attr('x', centerX)
      .attr('y', centerY - targetLength * 1.7)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .text(selectedCell ? 
        `Grid Cell (${selectedCell.i}, ${selectedCell.j})` : 
        `Point ${windVaneData.point_idx || 0}`
      );

    // Metric-aware indicator
    if (isMetricAware && windVaneData.metric_condition) {
      svg.append('text')
        .attr('x', centerX)
        .attr('y', centerY - targetLength * 1.5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('fill', windVaneData.metric_condition > 100 ? '#ff9800' : '#4CAF50')
        .text(`Metric Condition: ${windVaneData.metric_condition.toFixed(1)}`);
    }

    // Draw wind direction indicators (needles)
    displayVectors.forEach((vector, index) => {
      const { u, v, magnitude, feature_name, feature_idx } = vector;
      
      if (magnitude <= 0) return;

      // Calculate arrow endpoints
      const endX = centerX + u / dynamicScale;
      const endY = centerY + v / dynamicScale;
      
      // Color coding for features
      const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
      const color = colorScale(feature_idx || index);
      const isSignificant = magnitude > maxMagnitude * 0.3;
      
      // Arrow properties
      const alpha = isSignificant ? 0.9 : 0.5;
      const strokeWidth = isSignificant ? 3 : 1.5;

      // Draw needle shaft
      svg.append('line')
        .attr('x1', centerX)
        .attr('y1', centerY)
        .attr('x2', endX)
        .attr('y2', endY)
        .attr('stroke', color)
        .attr('stroke-width', strokeWidth)
        .attr('opacity', alpha)
        .attr('marker-end', 'url(#arrowhead)');

      // Draw arrowhead
      const arrowLength = isSignificant ? 10 : 6;
      const arrowAngle = Math.PI / 6;
      const angle = Math.atan2(v, u);
      
      const arrowhead1X = endX - arrowLength * Math.cos(angle - arrowAngle);
      const arrowhead1Y = endY - arrowLength * Math.sin(angle - arrowAngle);
      const arrowhead2X = endX - arrowLength * Math.cos(angle + arrowAngle);
      const arrowhead2Y = endY - arrowLength * Math.sin(angle + arrowAngle);

      svg.append('polygon')
        .attr('points', `${endX},${endY} ${arrowhead1X},${arrowhead1Y} ${arrowhead2X},${arrowhead2Y}`)
        .attr('fill', color)
        .attr('opacity', alpha);

      // Feature labels for significant needles
      if (isSignificant && feature_name) {
        const labelRadius = Math.sqrt(u*u + v*v) / dynamicScale + 15;
        const labelX = centerX + (u / Math.abs(u || 1)) * labelRadius;
        const labelY = centerY + (v / Math.abs(v || 1)) * labelRadius;
        
        svg.append('text')
          .attr('x', labelX)
          .attr('y', labelY)
          .attr('text-anchor', 'middle')
          .attr('font-size', '9px')
          .attr('font-weight', 'bold')
          .attr('fill', color)
          .attr('stroke', 'white')
          .attr('stroke-width', 2)
          .attr('paint-order', 'stroke')
          .text(feature_name.substring(0, 8));
      }
    });

    // Draw net wind direction (resultant)
    if (windVaneData.resultant && windVaneData.resultant.magnitude > 0) {
      const { u, v, magnitude, angle } = windVaneData.resultant;
      
      const resultantEndX = centerX + u / dynamicScale;
      const resultantEndY = centerY + v / dynamicScale;
      
      // Net wind shaft (distinctive styling)
      svg.append('line')
        .attr('x1', centerX)
        .attr('y1', centerY)
        .attr('x2', resultantEndX)
        .attr('y2', resultantEndY)
        .attr('stroke', '#2E7D32')
        .attr('stroke-width', 4)
        .attr('opacity', 0.9)
        .attr('stroke-dasharray', '8,2');

      // Net wind arrowhead
      const arrowLength = 14;
      const arrowAngle = Math.PI / 6;
      const vectorAngle = Math.atan2(v, u);
      
      const arrowhead1X = resultantEndX - arrowLength * Math.cos(vectorAngle - arrowAngle);
      const arrowhead1Y = resultantEndY - arrowLength * Math.sin(vectorAngle - arrowAngle);
      const arrowhead2X = resultantEndX - arrowLength * Math.cos(vectorAngle + arrowAngle);
      const arrowhead2Y = resultantEndY - arrowLength * Math.sin(vectorAngle + arrowAngle);

      svg.append('polygon')
        .attr('points', `${resultantEndX},${resultantEndY} ${arrowhead1X},${arrowhead1Y} ${arrowhead2X},${arrowhead2Y}`)
        .attr('fill', '#2E7D32')
        .attr('opacity', 0.9);

      // Net wind label with magnitude
      const labelOffset = 20;
      const labelX = resultantEndX + Math.cos(vectorAngle) * labelOffset;
      const labelY = resultantEndY + Math.sin(vectorAngle) * labelOffset;
      
      svg.append('text')
        .attr('x', labelX)
        .attr('y', labelY)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .attr('fill', '#2E7D32')
        .attr('stroke', 'white')
        .attr('stroke-width', 2)
        .attr('paint-order', 'stroke')
        .text(`Net (${magnitude.toFixed(2)})`);
    }

    // Research-grade legend
    const legendGroup = svg.append('g').attr('class', 'legend');
    
    legendGroup.append('text')
      .attr('x', 15)
      .attr('y', height - 90)
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .text('Wind Analysis:');

    // Wind rose indicator
    legendGroup.append('ellipse')
      .attr('cx', 25)
      .attr('cy', height - 75)
      .attr('rx', 8)
      .attr('ry', 4)
      .attr('fill', 'rgba(135, 206, 250, 0.4)')
      .attr('stroke', 'rgba(135, 206, 250, 0.8)');
    
    legendGroup.append('text')
      .attr('x', 40)
      .attr('y', height - 72)
      .attr('font-size', '10px')
      .text('Wind Rose Base');

    // Convex hull indicator
    if (showConvexHull) {
      legendGroup.append('text')
        .attr('x', 15)
        .attr('y', height - 55)
        .attr('font-size', '10px')
        .attr('fill', '#4CAF50')
        .attr('font-weight', 'bold')
        .text('✓ Convex Hull Filtered');
    }

    // Metric-aware indicator
    if (isMetricAware) {
      legendGroup.append('circle')
        .attr('cx', 20)
        .attr('cy', height - 40)
        .attr('r', 3)
        .attr('fill', '#4CAF50');
      
      legendGroup.append('text')
        .attr('x', 30)
        .attr('y', height - 37)
        .attr('font-size', '10px')
        .attr('fill', '#4CAF50')
        .attr('font-weight', 'bold')
        .text('Metric-Aware');
    }

    // Net wind legend
    legendGroup.append('line')
      .attr('x1', 15)
      .attr('y1', height - 25)
      .attr('x2', 35)
      .attr('y2', height - 25)
      .attr('stroke', '#2E7D32')
      .attr('stroke-width', 3)
      .attr('stroke-dasharray', '6,2');
    
    legendGroup.append('text')
      .attr('x', 40)
      .attr('y', height - 22)
      .attr('font-size', '10px')
      .text('Net Wind Direction');

  }, [windVaneData, selectedCell, boundingBox, isMetricAware, showConvexHull]);

  return (
    <div className="wind-vane">
      <div className="wind-vane-header">
        <h3>🌪️ Wind Vane Analysis</h3>
        <div className="wind-vane-controls">
          <button
            onClick={() => onToggleConvexHull && onToggleConvexHull()}
            className={`control-button ${showConvexHull ? 'active' : ''}`}
            title="Toggle convex hull filtering"
          >
            Convex Hull
          </button>
          <div className="metric-indicator">
            <span className={`metric-status ${isMetricAware ? 'active' : ''}`}>
              {isMetricAware ? '📐 Metric-Aware' : '📏 Standard'}
            </span>
          </div>
        </div>
        {windVaneData && (
          <div className="wind-vane-info">
            <div className="info-row">
              <span><strong>Location:</strong> {selectedCell ? 
                `Grid (${selectedCell.i}, ${selectedCell.j})` : 
                `Point ${windVaneData.point_idx || 0}`
              }</span>
            </div>
            <div className="info-row">
              <span><strong>Features:</strong> {windVaneData.vectors.length}</span>
              <span><strong>Filtered:</strong> {showConvexHull ? 
                computeConvexHull(windVaneData.vectors, 1).length : 
                windVaneData.vectors.length
              }</span>
            </div>
          </div>
        )}
      </div>
      
      <div className="wind-vane-svg-container">
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="wind-vane-svg"
        />
      </div>

      {windVaneData && (
        <div className="wind-analysis">
          {/* Wind Strength Gauge */}
          <div className="wind-strength-section">
            <h4>💨 Wind Strength</h4>
            <div className="strength-gauge">
              <div className="gauge-bar">
                <div 
                  className="gauge-fill"
                  style={{ width: `${Math.min(windStrength * 100, 100)}%` }}
                />
              </div>
              <span className="strength-value">{windStrength.toFixed(3)}</span>
            </div>
          </div>

          {/* Wind Rose Metrics */}
          <div className="wind-rose-section">
            <h4>🌹 Wind Rose</h4>
            <div className="rose-metrics">
              <div className="metric-item">
                <span>Anisotropy:</span>
                <span>{anisotropy.ratio.toFixed(2)}</span>
              </div>
              <div className="metric-item">
                <span>Orientation:</span>
                <span>{anisotropy.angle.toFixed(1)}°</span>
              </div>
              {windVaneData.metric_condition && (
                <div className="metric-item">
                  <span>Condition #:</span>
                  <span className={windVaneData.metric_condition > 100 ? 'warning' : 'good'}>
                    {windVaneData.metric_condition.toFixed(1)}
                  </span>
                </div>
              )}
            </div>
          </div>
          
          {/* Wind Components Table */}
          <div className="wind-components-section">
            <h4>🧭 Wind Components</h4>
            <div className="components-table">
              <div className="table-header">
                <span>Feature</span>
                <span>Magnitude</span>
                <span>Angle</span>
              </div>
              {windVaneData.vectors
                .sort((a, b) => b.magnitude - a.magnitude)
                .slice(0, 8) // Show top 8
                .map((vector, index) => (
                <div key={index} className="table-row">
                  <span className="feature-name" title={vector.feature_name || vector.label}>
                    {(vector.feature_name || vector.label || `F${index}`).substring(0, 12)}
                  </span>
                  <span className="magnitude">
                    {vector.magnitude.toFixed(3)}
                  </span>
                  <span className="angle">
                    {vector.angle ? vector.angle.toFixed(1) : 
                     (Math.atan2(vector.v, vector.u) * 180 / Math.PI).toFixed(1)}°
                  </span>
                </div>
              ))}
            </div>
          </div>
          
          {/* Net Wind Summary */}
          {windVaneData.resultant && (
            <div className="net-wind-section">
              <h4>🎯 Net Wind Direction</h4>
              <div className="net-wind-summary">
                <div className="net-magnitude">
                  <span>Magnitude:</span>
                  <span className="large-value">{windVaneData.resultant.magnitude.toFixed(3)}</span>
                </div>
                <div className="net-direction">
                  <span>Direction:</span>
                  <span>{windVaneData.resultant.angle ? 
                    windVaneData.resultant.angle.toFixed(1) :
                    (Math.atan2(windVaneData.resultant.v, windVaneData.resultant.u) * 180 / Math.PI).toFixed(1)
                  }°</span>
                </div>
                <div className="net-components">
                  <span>Components:</span>
                  <span>({windVaneData.resultant.u.toFixed(3)}, {windVaneData.resultant.v.toFixed(3)})</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default WindVane;