import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './WindVane.css';

const WindVane = ({ windVaneData, selectedCell, boundingBox }) => {
  const svgRef = useRef();
  const width = 400;
  const height = 400;

  useEffect(() => {
    if (!windVaneData || !boundingBox) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const [xmin, xmax, ymin, ymax] = boundingBox;
    const centerX = width / 2;
    const centerY = height / 2;
    const canvasSize = Math.min(xmax - xmin, ymax - ymin);
    const targetLength = Math.min(width, height) * 0.3;

    // Calculate dynamic scaling
    const vectorMagnitudes = windVaneData.vectors.map(v => v.magnitude);
    const maxMagnitude = Math.max(...vectorMagnitudes);
    const dynamicScale = maxMagnitude > 0 ? maxMagnitude / targetLength : 1.0;

    // Background circle
    svg.append('circle')
      .attr('cx', centerX)
      .attr('cy', centerY)
      .attr('r', targetLength * 1.2)
      .attr('fill', 'none')
      .attr('stroke', '#e0e0e0')
      .attr('stroke-width', 1);

    // Grid cell marker
    svg.append('rect')
      .attr('x', centerX - 6)
      .attr('y', centerY - 6)
      .attr('width', 12)
      .attr('height', 12)
      .attr('fill', 'gray')
      .attr('opacity', 0.8);

    // Grid cell label
    svg.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .text(`Grid Cell (${selectedCell.i}, ${selectedCell.j})`);

    // Draw feature vectors
    windVaneData.vectors.forEach((vector, index) => {
      const { u, v, magnitude, is_dominant, label } = vector;
      
      if (magnitude <= 0) return;

      // Calculate arrow endpoints
      const endX = centerX + u / dynamicScale;
      const endY = centerY + v / dynamicScale;
      
      // Arrow properties
      const alpha = is_dominant ? 0.9 : 0.6 * (magnitude / maxMagnitude);
      const color = is_dominant ? '#ff6b6b' : 'gray';
      const strokeWidth = is_dominant ? 2 : 1;

      // Draw arrow shaft
      svg.append('line')
        .attr('x1', centerX)
        .attr('y1', centerY)
        .attr('x2', endX)
        .attr('y2', endY)
        .attr('stroke', color)
        .attr('stroke-width', strokeWidth)
        .attr('opacity', alpha);

      // Draw arrowhead
      const arrowLength = 8;
      const arrowAngle = Math.PI / 6; // 30 degrees
      const angle = Math.atan2(v, u);
      
      const arrowhead1X = endX - arrowLength * Math.cos(angle - arrowAngle);
      const arrowhead1Y = endY - arrowLength * Math.sin(angle - arrowAngle);
      const arrowhead2X = endX - arrowLength * Math.cos(angle + arrowAngle);
      const arrowhead2Y = endY - arrowLength * Math.sin(angle + arrowAngle);

      svg.append('polygon')
        .attr('points', `${endX},${endY} ${arrowhead1X},${arrowhead1Y} ${arrowhead2X},${arrowhead2Y}`)
        .attr('fill', color)
        .attr('opacity', alpha);

      // Label for dominant feature
      if (is_dominant) {
        const labelX = endX + (u / dynamicScale) * 0.2;
        const labelY = endY + (v / dynamicScale) * 0.2;
        
        svg.append('text')
          .attr('x', labelX)
          .attr('y', labelY)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('font-weight', 'bold')
          .attr('fill', '#333')
          .attr('background', 'white')
          .text(label);
      }
    });

    // Draw resultant vector
    if (windVaneData.resultant && windVaneData.resultant.magnitude > 0) {
      const { u, v, magnitude } = windVaneData.resultant;
      
      const resultantEndX = centerX + u / dynamicScale;
      const resultantEndY = centerY + v / dynamicScale;
      
      // Resultant arrow shaft (thicker and black)
      svg.append('line')
        .attr('x1', centerX)
        .attr('y1', centerY)
        .attr('x2', resultantEndX)
        .attr('y2', resultantEndY)
        .attr('stroke', 'black')
        .attr('stroke-width', 3)
        .attr('opacity', 0.8);

      // Resultant arrowhead
      const arrowLength = 12;
      const arrowAngle = Math.PI / 6;
      const angle = Math.atan2(v, u);
      
      const arrowhead1X = resultantEndX - arrowLength * Math.cos(angle - arrowAngle);
      const arrowhead1Y = resultantEndY - arrowLength * Math.sin(angle - arrowAngle);
      const arrowhead2X = resultantEndX - arrowLength * Math.cos(angle + arrowAngle);
      const arrowhead2Y = resultantEndY - arrowLength * Math.sin(angle + arrowAngle);

      svg.append('polygon')
        .attr('points', `${resultantEndX},${resultantEndY} ${arrowhead1X},${arrowhead1Y} ${arrowhead2X},${arrowhead2Y}`)
        .attr('fill', 'black')
        .attr('opacity', 0.8);

      // Resultant label
      const labelX = resultantEndX + (u / dynamicScale) * 0.3;
      const labelY = resultantEndY + (v / dynamicScale) * 0.3;
      
      svg.append('text')
        .attr('x', labelX)
        .attr('y', labelY)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('fill', '#000')
        .text('Resultant');
    }

    // Legend
    const legendGroup = svg.append('g').attr('class', 'legend');
    
    legendGroup.append('text')
      .attr('x', 20)
      .attr('y', height - 60)
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .text('Legend:');

    legendGroup.append('line')
      .attr('x1', 20)
      .attr('y1', height - 45)
      .attr('x2', 40)
      .attr('y2', height - 45)
      .attr('stroke', '#ff6b6b')
      .attr('stroke-width', 2);
    
    legendGroup.append('text')
      .attr('x', 45)
      .attr('y', height - 42)
      .attr('font-size', '10px')
      .text('Dominant Feature');

    legendGroup.append('line')
      .attr('x1', 20)
      .attr('y1', height - 30)
      .attr('x2', 40)
      .attr('y2', height - 30)
      .attr('stroke', 'gray')
      .attr('stroke-width', 1);
    
    legendGroup.append('text')
      .attr('x', 45)
      .attr('y', height - 27)
      .attr('font-size', '10px')
      .text('Other Features');

    legendGroup.append('line')
      .attr('x1', 20)
      .attr('y1', height - 15)
      .attr('x2', 40)
      .attr('y2', height - 15)
      .attr('stroke', 'black')
      .attr('stroke-width', 3);
    
    legendGroup.append('text')
      .attr('x', 45)
      .attr('y', height - 12)
      .attr('font-size', '10px')
      .text('Resultant Vector');

  }, [windVaneData, selectedCell, boundingBox]);

  return (
    <div className="wind-vane">
      <div className="wind-vane-header">
        <h3>Wind Vane Analysis</h3>
        {windVaneData && (
          <div className="wind-vane-info">
            <p><strong>Cell:</strong> ({selectedCell.i}, {selectedCell.j})</p>
            <p><strong>Features:</strong> {windVaneData.vectors.length}</p>
            <p><strong>Dominant:</strong> {windVaneData.vectors.find(v => v.is_dominant)?.label || 'None'}</p>
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
        <div className="feature-details">
          <h4>Feature Vectors</h4>
          <div className="feature-list">
            {windVaneData.vectors.map((vector, index) => (
              <div 
                key={index} 
                className={`feature-item ${vector.is_dominant ? 'dominant' : ''}`}
              >
                <span className="feature-name">{vector.label}</span>
                <span className="feature-magnitude">
                  Mag: {vector.magnitude.toFixed(3)}
                </span>
                {vector.is_dominant && <span className="dominant-badge">Dominant</span>}
              </div>
            ))}
          </div>
          
          {windVaneData.resultant && (
            <div className="resultant-info">
              <h4>Resultant Vector</h4>
              <p>Magnitude: {windVaneData.resultant.magnitude.toFixed(3)}</p>
              <p>Direction: ({windVaneData.resultant.u.toFixed(3)}, {windVaneData.resultant.v.toFixed(3)})</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default WindVane;