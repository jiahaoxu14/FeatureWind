import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';
import './WindMapVisualization.css';

const WindMapVisualization = ({ onCellSelect, selectedCell, dataInfo }) => {
  const svgRef = useRef();
  const [dataPoints, setDataPoints] = useState([]);
  const [gridData, setGridData] = useState(null);
  const [particles, setParticles] = useState([]);
  const [showGrid, setShowGrid] = useState(false);
  const animationRef = useRef();
  const [isAnimating, setIsAnimating] = useState(false);

  const width = 800;
  const height = 600;
  const numParticles = 2000;

  useEffect(() => {
    if (dataInfo) {
      fetchDataPoints();
      fetchGridData();
      initializeParticles();
    }
  }, [dataInfo]);

  useEffect(() => {
    if (isAnimating) {
      startAnimation();
    } else {
      stopAnimation();
    }
    return () => stopAnimation();
  }, [isAnimating, particles]);

  const fetchDataPoints = async () => {
    try {
      const response = await axios.get('/api/get_data_points');
      setDataPoints(response.data.points);
    } catch (error) {
      console.error('Error fetching data points:', error);
    }
  };

  const fetchGridData = async () => {
    try {
      const response = await axios.get('/api/get_grid_data');
      setGridData(response.data);
    } catch (error) {
      console.error('Error fetching grid data:', error);
    }
  };

  const initializeParticles = () => {
    if (!dataInfo?.bounding_box) return;
    
    const [xmin, xmax, ymin, ymax] = dataInfo.bounding_box;
    const newParticles = [];
    
    for (let i = 0; i < numParticles; i++) {
      newParticles.push({
        id: i,
        x: Math.random() * (xmax - xmin) + xmin,
        y: Math.random() * (ymax - ymin) + ymin,
        vx: 0,
        vy: 0,
        age: 0,
        maxAge: 200 + Math.random() * 200,
        trail: []
      });
    }
    setParticles(newParticles);
  };

  const updateParticles = async () => {
    if (!particles.length || !dataInfo?.bounding_box) return;

    const [xmin, xmax, ymin, ymax] = dataInfo.bounding_box;
    const positions = particles.map(p => [p.x, p.y]);
    
    try {
      const response = await axios.post('/api/get_velocity_field', {
        positions: positions
      });
      
      const velocities = response.data.velocities;
      const velocityScale = 0.5;
      
      setParticles(prevParticles => 
        prevParticles.map((particle, index) => {
          const [vx, vy] = velocities[index] || [0, 0];
          
          // Update position
          let newX = particle.x + vx * velocityScale;
          let newY = particle.y + vy * velocityScale;
          let newAge = particle.age + 1;
          
          // Reset if out of bounds or too old
          if (newX < xmin || newX > xmax || newY < ymin || newY > ymax || newAge > particle.maxAge) {
            newX = Math.random() * (xmax - xmin) + xmin;
            newY = Math.random() * (ymax - ymin) + ymin;
            newAge = 0;
          }
          
          // Update trail
          const newTrail = [...particle.trail, { x: particle.x, y: particle.y }];
          if (newTrail.length > 10) {
            newTrail.shift();
          }
          
          return {
            ...particle,
            x: newX,
            y: newY,
            vx: vx,
            vy: vy,
            age: newAge,
            trail: newTrail
          };
        })
      );
    } catch (error) {
      console.error('Error updating particles:', error);
    }
  };

  const startAnimation = () => {
    const animate = () => {
      updateParticles();
      animationRef.current = requestAnimationFrame(animate);
    };
    animationRef.current = requestAnimationFrame(animate);
  };

  const stopAnimation = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    if (!dataInfo?.bounding_box) return;

    const [xmin, xmax, ymin, ymax] = dataInfo.bounding_box;
    const xScale = d3.scaleLinear().domain([xmin, xmax]).range([50, width - 50]);
    const yScale = d3.scaleLinear().domain([ymin, ymax]).range([height - 50, 50]);

    // Grid visualization
    if (showGrid && gridData) {
      const gridGroup = svg.append('g').attr('class', 'grid-group');
      
      // Draw grid lines
      gridData.grid_lines.forEach(line => {
        const pathData = d3.line()
          .x(d => xScale(d[0]))
          .y(d => yScale(d[1]))(line);
        
        gridGroup.append('path')
          .attr('d', pathData)
          .attr('stroke', 'gray')
          .attr('stroke-width', 0.3)
          .attr('stroke-dasharray', '2,2')
          .attr('fill', 'none')
          .attr('opacity', 0.5);
      });
      
      // Draw grid cells
      gridData.cells.forEach(cell => {
        const [x1, y1, x2, y2] = cell.bounds;
        gridGroup.append('rect')
          .attr('x', xScale(x1))
          .attr('y', yScale(y2))
          .attr('width', xScale(x2) - xScale(x1))
          .attr('height', yScale(y1) - yScale(y2))
          .attr('fill', cell.is_empty ? 'white' : 'gray')
          .attr('opacity', 0.1)
          .attr('stroke', 'none')
          .on('click', () => onCellSelect(cell.i, cell.j))
          .on('mouseover', function() {
            d3.select(this).attr('opacity', 0.3);
          })
          .on('mouseout', function() {
            d3.select(this).attr('opacity', 0.1);
          })
          .style('cursor', 'pointer');
      });
    }

    // Data points
    const pointsGroup = svg.append('g').attr('class', 'points-group');
    const uniqueLabels = [...new Set(dataPoints.map(p => p.label))];
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10).domain(uniqueLabels);
    const symbolScale = d3.scaleOrdinal(d3.symbols).domain(uniqueLabels);

    uniqueLabels.forEach(label => {
      const points = dataPoints.filter(p => p.label === label);
      pointsGroup.selectAll(`.points-${label}`)
        .data(points)
        .enter()
        .append('path')
        .attr('class', `points-${label}`)
        .attr('d', d3.symbol().type(symbolScale(label)).size(20))
        .attr('transform', d => `translate(${xScale(d.position[0])}, ${yScale(d.position[1])})`)
        .attr('fill', colorScale(label))
        .attr('opacity', 0.7);
    });

    // Particles
    const particlesGroup = svg.append('g').attr('class', 'particles-group');
    
    const updateParticleVisualization = () => {
      // Draw particle trails
      const trails = particlesGroup.selectAll('.trail')
        .data(particles.filter(p => p.trail.length > 1));
      
      trails.enter()
        .append('path')
        .attr('class', 'trail')
        .merge(trails)
        .attr('d', d => {
          if (d.trail.length < 2) return '';
          const line = d3.line()
            .x(point => xScale(point.x))
            .y(point => yScale(point.y));
          return line(d.trail);
        })
        .attr('stroke', 'gray')
        .attr('stroke-width', 1)
        .attr('fill', 'none')
        .attr('opacity', d => Math.max(0.1, 1 - d.age / d.maxAge));
      
      trails.exit().remove();
      
      // Draw particles
      const particleDots = particlesGroup.selectAll('.particle')
        .data(particles);
      
      particleDots.enter()
        .append('circle')
        .attr('class', 'particle')
        .attr('r', 1.5)
        .merge(particleDots)
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .attr('fill', 'gray')
        .attr('opacity', d => Math.max(0.3, 1 - d.age / d.maxAge));
      
      particleDots.exit().remove();
    };

    // Initial particle visualization
    updateParticleVisualization();

    // Update particles when they change
    const interval = setInterval(updateParticleVisualization, 50);
    return () => clearInterval(interval);

  }, [dataPoints, gridData, showGrid, particles, dataInfo, onCellSelect]);

  // Highlight selected cell
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    
    // Remove previous selection highlight
    svg.select('.selection-highlight').remove();
    
    if (gridData && selectedCell && dataInfo?.bounding_box) {
      const [xmin, xmax, ymin, ymax] = dataInfo.bounding_box;
      const xScale = d3.scaleLinear().domain([xmin, xmax]).range([50, width - 50]);
      const yScale = d3.scaleLinear().domain([ymin, ymax]).range([height - 50, 50]);
      
      const cell = gridData.cells.find(c => c.i === selectedCell.i && c.j === selectedCell.j);
      if (cell) {
        const [x1, y1, x2, y2] = cell.bounds;
        svg.append('rect')
          .attr('class', 'selection-highlight')
          .attr('x', xScale(x1))
          .attr('y', yScale(y2))
          .attr('width', xScale(x2) - xScale(x1))
          .attr('height', yScale(y1) - yScale(y2))
          .attr('fill', 'none')
          .attr('stroke', 'red')
          .attr('stroke-width', 2);
      }
    }
  }, [selectedCell, gridData, dataInfo]);

  return (
    <div className="wind-map-visualization">
      <div className="controls">
        <button onClick={() => setShowGrid(!showGrid)}>
          {showGrid ? 'Hide Grid' : 'Show Grid'}
        </button>
        <button onClick={() => setIsAnimating(!isAnimating)}>
          {isAnimating ? 'Stop Animation' : 'Start Animation'}
        </button>
      </div>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="wind-map-svg"
      />
    </div>
  );
};

export default WindMapVisualization;