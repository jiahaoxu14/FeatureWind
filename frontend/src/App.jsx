import React, { useState, useEffect, useMemo, useRef } from 'react'
import './styles.css'
import { uploadFile, compute, recolor } from './services/api'
import CanvasWind from './components/CanvasWind.jsx'
import WindVane from './components/WindVane.jsx'
import ColorLegend from './components/ColorLegend.jsx'
import PointSeries from './components/PointSeries.jsx'

export default function App() {
  const fileInputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [dataset, setDataset] = useState(null)
  const [topK, setTopK] = useState(0)
  const [gridRes, setGridRes] = useState(25)
  const [payload, setPayload] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [hoverPos, setHoverPos] = useState(null)
  const [selectedCells, setSelectedCells] = useState([]) // array of {i,j}
  // Interactive config (frontend + backend overrides)
  const [showGrid, setShowGrid] = useState(true)
  const [particleCount, setParticleCount] = useState(1000)
  const [speedScale, setSpeedScale] = useState(1.0)
  const [tailLength, setTailLength] = useState(10)
  const [trailTailMin, setTrailTailMin] = useState(0.10)
  const [trailTailExp, setTrailTailExp] = useState(2.0)
  const [trailLineWidth, setTrailLineWidth] = useState(2.0)
  const [maxLifetime, setMaxLifetime] = useState(200)
  const [maskBufferFactor, setMaskBufferFactor] = useState(0.2)
  const [showHull, setShowHull] = useState(false)
  const [showVectorLabels, setShowVectorLabels] = useState(false)
  const [showAllVectors, setShowAllVectors] = useState(false)
  const [hideParticles, setHideParticles] = useState(false)
  const [pointColorFeature, setPointColorFeature] = useState('') // '' or feature index string
  const [showPointGradients, setShowPointGradients] = useState(false)
  const [showCellGradients, setShowCellGradients] = useState(false)
  const [showCellAggGradients, setShowCellAggGradients] = useState(false)
  const [showParticleInits, setShowParticleInits] = useState(false)
  const [uniformPointShape, setUniformPointShape] = useState(false)
  const [showParticleArrowheads, setShowParticleArrowheads] = useState(false)
  const [restrictSpawnToSelection, setRestrictSpawnToSelection] = useState(false)
  const [autoRespawnEnabled, setAutoRespawnEnabled] = useState(false)
  const [assessAllCells, setAssessAllCells] = useState(false)
  const [brushRadius, setBrushRadius] = useState(1)
  const [selectPointsMode, setSelectPointsMode] = useState(false)
  const [selectedPointIndices, setSelectedPointIndices] = useState([])
  const [seriesFeatureIndices, setSeriesFeatureIndices] = useState([])
  // Manual feature selection (overrides Top-K when enabled)
  const [selectedFeatureIndices, setSelectedFeatureIndices] = useState([])
  const [useManualFeatures, setUseManualFeatures] = useState(false)

  // Refs to canvases for exporting PNGs
  const windMapCanvasRef = useRef(null)
  const windVaneCanvasRef = useRef(null)

  function saveCanvasAsPng(canvas, filename) {
    try {
      if (!canvas) return
      const url = canvas.toDataURL('image/png')
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    } catch (e) { /* ignore */ }
  }

  // Controls update live; no separate "Apply/Update" states

  // Clamp manual feature selection when payload changes (e.g., new dataset)
  useEffect(() => {
    const n = (payload?.col_labels?.length) || 0
    if (!n) { setSelectedFeatureIndices([]); return }
    setSelectedFeatureIndices((prev) => prev.filter((i) => i >= 0 && i < n))
    setUseManualFeatures(false)
  }, [payload?.col_labels])

  async function handleUpload(selected) {
    const f = selected || file
    if (!f) return
    setError('')
    try {
      // Only accept .tmap or .json uploads
      const lower = (f.name || '').toLowerCase()
      if (!(lower.endsWith('.tmap') || lower.endsWith('.json'))) {
        setError('Please upload a .tmap (or JSON with tmap structure).')
        return
      }
      const res = await uploadFile(f)
      // Default Top-K to all features in the dataset
      const m = Array.isArray(res.col_labels) ? res.col_labels.length : 0
      if (m > 0) { setTopK(m) }
      setDataset(res)
    } catch (e) {
      setError(e.message)
    }
  }

  async function handleCompute(forDatasetId) {
    const dsId = forDatasetId || (dataset && dataset.datasetId)
    if (!dsId) return
    setBusy(true)
    setError('')
    try {
      const res = await compute({
        dataset_id: dsId,
        topK: Number(topK),
        gridRes: Number(gridRes),
        config: { maskBufferFactor: Number(maskBufferFactor) }
      })
      setPayload(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  // Auto compute whenever dataset or server-affecting options change
  useEffect(() => {
    const dsId = dataset && dataset.datasetId
    if (!dsId) return
    const t = setTimeout(() => {
      handleCompute(dsId)
    }, 200)
    return () => clearTimeout(t)
  // Only recompute when applied values change or dataset changes
  }, [dataset?.datasetId, topK, gridRes, maskBufferFactor])

  // Selection handlers
  function toggleCell(i, j) {
    setSelectedCells((prev) => {
      const exists = prev.some((c) => c.i === i && c.j === j)
      if (exists) return prev.filter((c) => !(c.i === i && c.j === j))
      return [...prev, { i, j }]
    })
  }
  function setSingleCell(i, j) {
    setSelectedCells([{ i, j }])
  }
  function clearSelection() { setSelectedCells([]) }

  // Keyboard: 'c' clears selection
  useEffect(() => {
    function onKey(e) { if (e.key === 'c' || e.key === 'C') clearSelection() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Derive focus for Wind Vane: center of last selected cell, else hover, else center
  const vaneFocus = (() => {
    const bbox = payload?.bbox || [0,1,0,1]
    const [xmin,xmax,ymin,ymax] = bbox
    const H = payload?.grid_res || 25
    const W = H
    if (selectedCells && selectedCells.length > 0) {
      const { i, j } = selectedCells[selectedCells.length - 1]
      const x = xmin + (j + 0.5) * (xmax - xmin) / W
      const y = ymin + (i + 0.5) * (ymax - ymin) / H
      return { x, y }
    }
    return hoverPos || null
  })()

  // Compute average number of active (non-zero) selected features per valid grid cell
  const avgActiveFeatures = useMemo(() => {
    try {
      if (!payload) return null
      const { uAll = [], vAll = [], grid_res = 25, selection = {}, unmasked = null } = payload
      const H = grid_res, W = grid_res
      // Determine feature indices in scope
      let indices = []
      if (useManualFeatures) indices = selectedFeatureIndices
      else if (selection && Array.isArray(selection.topKIndices)) indices = selection.topKIndices
      else if (selection && typeof selection.featureIndex === 'number') indices = [selection.featureIndex]
      if (!Array.isArray(indices) || indices.length === 0) return 0
      let total = 0
      let cells = 0
      const eps = 1e-9
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          if (Array.isArray(unmasked) && unmasked.length === H && unmasked[0].length === W) {
            if (!unmasked[i][j]) continue
          }
          cells += 1
          let cnt = 0
          for (const fi of indices) {
            const u = (uAll[fi]?.[i]?.[j] ?? 0)
            const v = (vAll[fi]?.[i]?.[j] ?? 0)
            if (u*u + v*v > eps) cnt += 1
          }
          total += cnt
        }
      }
      if (cells === 0) return 0
      return total / cells
    } catch { return null }
  }, [payload, selectedFeatureIndices, useManualFeatures])

  // Compute which features are visible in the current Wind Vane view
  const visibleFeatures = useMemo(() => {
    const result = new Set()
    if (!payload) return result
    const { bbox = [0,1,0,1], grid_res = 25, uAll = [], vAll = [], selection = {}, unmasked = null, dominant = null } = payload
    const [xmin, xmax, ymin, ymax] = bbox
    const H = grid_res, W = grid_res
    // Determine selected features (indices rendered in vane)
    let indices = []
    if (useManualFeatures) indices = selectedFeatureIndices
    else if (selection && Array.isArray(selection.topKIndices)) indices = selection.topKIndices
    else if (selection && typeof selection.featureIndex === 'number') indices = [selection.featureIndex]
    if (!indices.length) return result
    // Selection mode
    const useSel = selectedCells && selectedCells.length > 0
    const hasUnmasked = Array.isArray(unmasked) && unmasked.length === H && unmasked[0].length === W
    const hasDominant = Array.isArray(dominant) && dominant.length === H && dominant[0].length === W
    const eps = 1e-9
    // Helper: convex hull (monotone chain) returns indices
    function hullIdx(pts) {
      if (!pts || pts.length < 3) return pts.map((_, i) => i)
      const arr = pts.map((p, i) => [p[0], p[1], i]).sort((a,b)=> (a[0]===b[0]? a[1]-b[1] : a[0]-b[0]))
      const cross = (o,a,b)=> (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
      const lower=[]; for (const p of arr){ while(lower.length>=2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0) lower.pop(); lower.push(p) }
      const upper=[]; for (let k=arr.length-1;k>=0;k--){ const p=arr[k]; while(upper.length>=2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0) upper.pop(); upper.push(p) }
      const hull = lower.slice(0,-1).concat(upper.slice(0,-1))
      const seen=new Set(), idxs=[]; for (const h of hull){ if(!seen.has(h[2])){ seen.add(h[2]); idxs.push(h[2]) } }
      return idxs
    }
    if (useSel) {
      // Filter to valid cells
      const valid = []
      if (hasUnmasked) {
        for (const c of selectedCells) {
          const ii = Math.max(0, Math.min(H - 1, c.i|0))
          const jj = Math.max(0, Math.min(W - 1, c.j|0))
          if (unmasked[ii][jj]) valid.push({ i: ii, j: jj })
        }
      } else {
        for (const c of selectedCells) {
          const ii = Math.max(0, Math.min(H - 1, c.i|0))
          const jj = Math.max(0, Math.min(W - 1, c.j|0))
          valid.push({ i: ii, j: jj })
        }
      }
      if (!valid.length) return result
      // Build vectors for selected features
      const vecs = []
      const feats = []
      for (const idx of indices) {
        let u = 0, v = 0
        for (const c of valid) { u += (uAll[idx]?.[c.i]?.[c.j] ?? 0); v += (vAll[idx]?.[c.i]?.[c.j] ?? 0) }
        const mag2 = u*u + v*v
        if (mag2 > eps) { vecs.push([u, v]); feats.push(idx) }
      }
      if (vecs.length === 0) return result
      if (vecs.length < 3) { feats.forEach(i=>result.add(i)); return result }
      // Normalize endpoints by max magnitude to mirror vane scaling
      let maxMag = 0; for (const [u,v] of vecs){ const m=Math.hypot(u,v); if(m>maxMag) maxMag=m }
      const pts = vecs.map(([u,v]) => [u/(maxMag||1), v/(maxMag||1)])
      const hidx = hullIdx(pts)
      for (const hi of hidx) result.add(feats[hi])
      return result
    }
    // Hover mode: snap to clicked/hovered cell center indices using floor mapping
    const xCoord = vaneFocus?.x
    const yCoord = vaneFocus?.y
    if (typeof xCoord !== 'number' || typeof yCoord !== 'number') return result
    const cj = Math.max(0, Math.min(W - 1, Math.floor(((xCoord - xmin) / (xmax - xmin)) * W)))
    const ci = Math.max(0, Math.min(H - 1, Math.floor(((yCoord - ymin) / (ymax - ymin)) * H)))
    // Masked?
    if (hasUnmasked && !unmasked[ci][cj]) return result
    if (hasDominant && dominant[ci][cj] === -1) return result
    // Build vectors
    const vecs = []
    const feats = []
    for (const idx of indices) {
      const u = (uAll[idx]?.[ci]?.[cj] ?? 0)
      const v = (vAll[idx]?.[ci]?.[cj] ?? 0)
      if ((u*u + v*v) > eps) { vecs.push([u, v]); feats.push(idx) }
    }
    if (vecs.length === 0) return result
    if (vecs.length < 3) { feats.forEach(i=>result.add(i)); return result }
    let maxMag = 0; for (const [u,v] of vecs){ const m=Math.hypot(u,v); if(m>maxMag) maxMag=m }
    const pts = vecs.map(([u,v]) => [u/(maxMag||1), v/(maxMag||1)])
    const hidx = hullIdx(pts)
    for (const hi of hidx) result.add(feats[hi])
    return result
  }, [payload, selectedCells, vaneFocus, selectedFeatureIndices, useManualFeatures])

  return (
    <div className="app">
      <div className="header">
        <h2 className="title">Feature Wind Map</h2>
        <div className="file-upload">
          <button
            className="btn"
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
            title="Choose a .tmap or JSON file"
          >Choose File</button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".tmap,.json"
            style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0] || null
              setFile(f)
              if (f) handleUpload(f)
            }}
          />
        </div>
      </div>
      <div className="content">
        <div className="three-up">
          {/* Wind Map */}
          <div className="panel canvas-frame">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <p className="panel-title" style={{ margin: 0 }}>Wind Map</p>
              {payload && (
                <button
                  className="btn"
                  title="Save Wind Map as PNG"
                  onClick={() => saveCanvasAsPng(windMapCanvasRef.current, 'wind_map.png')}
                >Save PNG</button>
              )}
            </div>
            {payload ? (
              <CanvasWind
                payload={payload}
                onHover={setHoverPos}
                onSelectCell={({ i, j, shift }) => shift ? toggleCell(i, j) : setSingleCell(i, j)}
                onBrushCell={(payload) => {
                  if (payload && Array.isArray(payload.cells)) {
                    const cells = payload.cells
                    setSelectedCells((prev) => {
                      const key = (c) => `${c.i}:${c.j}`
                      const setPrev = new Set(prev.map(key))
                      const out = prev.slice()
                      for (const c of cells) { const k = key(c); if (!setPrev.has(k)) { setPrev.add(k); out.push({ i: c.i, j: c.j }) } }
                      return out
                    })
                    return
                  }
                  const { i, j } = payload || {}
                  if (typeof i === 'number' && typeof j === 'number') {
                    setSelectedCells((prev) => {
                      const exists = prev.some((c) => c.i === i && c.j === j)
                      if (exists) return prev
                      return [...prev, { i, j }]
                    })
                  }
                }}
                showGrid={showGrid}
                particleCount={particleCount}
                speedScale={speedScale}
                tailLength={tailLength}
                trailTailMin={trailTailMin}
                trailTailExp={trailTailExp}
                maxLifetime={maxLifetime}
                size={600}
                showParticles={!hideParticles}
                pointColorFeatureIndex={pointColorFeature !== '' ? Number(pointColorFeature) : null}
                showPointGradients={showPointGradients}
                showCellGradients={showCellGradients}
                showCellAggregatedGradients={showCellAggGradients}
                showParticleInits={showParticleInits}
                gradientFeatureIndices={useManualFeatures ? selectedFeatureIndices : []}
                selectedCells={selectedCells}
                featureIndices={useManualFeatures ? selectedFeatureIndices : null}
                uniformPointShape={uniformPointShape}
                showParticleArrowheads={showParticleArrowheads}
                allowGridSelection={restrictSpawnToSelection}
                restrictSpawnToSelection={restrictSpawnToSelection}
                autoRespawnRate={autoRespawnEnabled ? 0.3 : 0.0}
                selectPointsMode={selectPointsMode}
                selectedPointIndices={selectedPointIndices}
                onSelectPoint={({ idx, append }) => {
                  setSelectedPointIndices((prev) => {
                    const exists = prev.includes(idx)
                    if (append) {
                      // Toggle in multi-select mode
                      return exists ? prev.filter((i) => i !== idx) : [...prev, idx]
                    }
                    // Single-click: select this point, unless already selected → then unselect it
                    return exists ? prev.filter((i) => i !== idx) : [idx]
                  })
                }}
                onBrushPoints={({ indices }) => {
                  if (!Array.isArray(indices) || !indices.length) return
                  setSelectedPointIndices((prev) => {
                    const setPrev = new Set(prev)
                    const out = prev.slice()
                    for (const i of indices) { if (!setPrev.has(i)) { setPrev.add(i); out.push(i) } }
                    return out
                  })
                }}
                trailLineWidth={trailLineWidth}
                onCanvasElement={(el) => { windMapCanvasRef.current = el }}
              />
            ) : (
              <div className="panel placeholder" style={{ width: 600, height: 600 }}>Wind Map</div>
            )}
          </div>

          {/* Wind Vane */}
          <div className="panel canvas-frame">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <p className="panel-title" style={{ margin: 0 }}>Wind Vane{selectedCells.length > 0 ? ` (selection: ${selectedCells.length} cells)` : ''}</p>
              {payload && (
                <button
                  className="btn"
                  title="Save Wind Vane as PNG"
                  onClick={() => saveCanvasAsPng(windVaneCanvasRef.current, 'wind_vane.png')}
                >Save PNG</button>
              )}
            </div>
            {payload ? (
                <WindVane
                  payload={payload}
                  focus={vaneFocus}
                  selectedCells={selectedCells}
                  useConvexHull={!showAllVectors}
                  showHull={showHull}
                  showLabels={showVectorLabels}
                  featureIndices={useManualFeatures ? selectedFeatureIndices : null}
                  onCanvasElement={(el) => { windVaneCanvasRef.current = el }}
                />
            ) : (
              <div className="panel placeholder" style={{ width: 600, height: 600 }}>Wind Vane</div>
            )}
          </div>

          {/* Color Families */}
          <div className="panel padded color-panel">
            <p className="panel-title">Color Families</p>
            {payload ? (
              <ColorLegend
                payload={payload}
                dataset={dataset}
                visible={visibleFeatures}
                selectedFeatures={useManualFeatures ? selectedFeatureIndices : null}
                onChangeSelectedFeatures={(arr) => { setSelectedFeatureIndices(arr); setUseManualFeatures(true) }}
                onSwapFamilyColors={(famA, famB) => {
                  // Swap colors between two family IDs across all features (front-end only)
                  setPayload((prev) => {
                    try {
                      if (!prev) return prev
                      const fams = prev.family_assignments || []
                      const cols = (prev.colors || []).slice()
                      if (!Array.isArray(fams) || !Array.isArray(cols)) return prev
                      // Representative colors (first feature in each family)
                      const idxA = fams.findIndex((f) => String(f) === String(famA))
                      const idxB = fams.findIndex((f) => String(f) === String(famB))
                      const colA = idxA >= 0 ? cols[idxA] : '#888'
                      const colB = idxB >= 0 ? cols[idxB] : '#888'
                      const out = cols.slice()
                      for (let i = 0; i < fams.length; i++) {
                        if (String(fams[i]) === String(famA)) out[i] = colB
                        else if (String(fams[i]) === String(famB)) out[i] = colA
                      }
                      return { ...prev, colors: out }
                    } catch { return prev }
                  })
                }}
                onApplyFamilies={async (families) => {
                  try {
                    if (!dataset) return
                    const res = await recolor(dataset.datasetId, families)
                    setPayload((prev) => ({ ...prev, colors: res.colors, family_assignments: res.family_assignments }))
                  } catch (e) {
                    setError(e.message)
                  }
                }}
              />
            ) : (
              <div className="hint">Upload a dataset to see colors</div>
            )}
          </div>

          {/* Point Series */}
          <div className="panel padded">
            <p className="panel-title">Point Series</p>
            <div className="row" style={{ alignItems: 'center', marginBottom: 8, gap: 10 }}>
              <label style={{ color: 'var(--muted)', fontSize: 13 }}>Select Points Mode</label>
              <input type="checkbox" checked={selectPointsMode} onChange={(e) => setSelectPointsMode(e.target.checked)} />
              <button className="btn" onClick={() => setSelectedPointIndices([])}>Clear Points</button>
            </div>
            {payload ? (
              <PointSeries
                indices={selectedPointIndices}
                featureValues={payload.feature_values || []}
                featureIndices={seriesFeatureIndices}
                onChangeFeatureIndices={setSeriesFeatureIndices}
                colLabels={payload.col_labels || []}
                colors={payload.colors || []}
                width={600}
                height={220}
              />
            ) : (
              <div className="hint">Upload a dataset to view series</div>
            )}
          </div>

          {/* Controls */}
          <div className="panel padded controls-grid full-span">
            {null}

            {null}

            <label>Grid Res</label>
            <div className="slider-row">
              <input type="range" min={8} max={200} step={1} value={gridRes}
                onChange={(e) => setGridRes(Number(e.target.value))} />
              <span className="control-val">{gridRes}</span>
            </div>

            <label>Mask Buffer</label>
            <div className="slider-row">
              <input type="range" min={0} max={2} step={0.05} value={maskBufferFactor}
                onChange={(e) => setMaskBufferFactor(Number(e.target.value))} />
              <span className="control-val">{maskBufferFactor.toFixed(2)}</span>
            </div>

            <label>Show Grid</label>
            <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />

            <label>Show Convex Hull</label>
            <input type="checkbox" checked={showHull} onChange={(e) => setShowHull(e.target.checked)} />

            <label>Show Vector Labels</label>
            <input type="checkbox" checked={showVectorLabels} onChange={(e) => setShowVectorLabels(e.target.checked)} />

            <label>Show All Vectors</label>
            <input type="checkbox" checked={showAllVectors} onChange={(e) => setShowAllVectors(e.target.checked)} />

            <label>Hide Particles</label>
            <input type="checkbox" checked={hideParticles} onChange={(e) => setHideParticles(e.target.checked)} />

            <label>Point Color By</label>
            <select
              value={pointColorFeature}
              onChange={(e) => setPointColorFeature(e.target.value)}
              disabled={!payload || !Array.isArray(payload.feature_values)}
            >
              <option value="">None</option>
              {Array.isArray(payload?.col_labels) && payload?.col_labels.map((name, idx) => (
                <option key={idx} value={String(idx)}>{name}</option>
              ))}
            </select>

            <label>Use Circle Markers</label>
            <input type="checkbox" checked={uniformPointShape} onChange={(e) => setUniformPointShape(e.target.checked)} />

            <label>Particle Arrowheads</label>
            <input type="checkbox" checked={showParticleArrowheads} onChange={(e) => setShowParticleArrowheads(e.target.checked)} />

            <label>Show Point Gradients</label>
            <input type="checkbox" checked={showPointGradients} onChange={(e) => setShowPointGradients(e.target.checked)} />

            <label>Show Cell Gradients</label>
            <input type="checkbox" checked={showCellGradients} onChange={(e) => setShowCellGradients(e.target.checked)} />

            <label>Show Aggregated Cell Gradients</label>
            <input type="checkbox" checked={showCellAggGradients} onChange={(e) => setShowCellAggGradients(e.target.checked)} />

            <label>Show Particle Inits</label>
            <input type="checkbox" checked={showParticleInits} onChange={(e) => setShowParticleInits(e.target.checked)} />

            <label>Restrict Spawns to Selection</label>
            <input type="checkbox" checked={restrictSpawnToSelection} onChange={(e) => setRestrictSpawnToSelection(e.target.checked)} />

            <label>Auto Respawn (30% bursts)</label>
            <input type="checkbox" checked={autoRespawnEnabled} onChange={(e) => setAutoRespawnEnabled(e.target.checked)} />

            <label>Assess Feature Coverage</label>
            <div className="slider-row">
              <input type="checkbox" checked={assessAllCells} onChange={(e) => setAssessAllCells(e.target.checked)} />
              {assessAllCells && (
                <span className="control-val">
                  Avg active: {typeof avgActiveFeatures === 'number' ? avgActiveFeatures.toFixed(2) : '-'}
                </span>
              )}
            </div>

            <label>Particles</label>
            <div className="slider-row">
              <input type="range" min={50} max={5000} step={50} value={particleCount}
                onChange={(e) => setParticleCount(Number(e.target.value))} />
              <span className="control-val">{particleCount}</span>
            </div>

            {null}

            <label>Speed Scale</label>
            <div className="slider-row">
              <input type="range" min={0.1} max={10} step={0.1} value={speedScale}
                onChange={(e) => setSpeedScale(Number(e.target.value))} />
              <span className="control-val">{speedScale.toFixed(1)}</span>
            </div>

            <label>Tail Length</label>
            <div className="slider-row">
              <input type="range" min={2} max={60} step={1} value={tailLength}
                onChange={(e) => setTailLength(Number(e.target.value))} />
              <span className="control-val">{tailLength}</span>
            </div>

            <label>Trail Width</label>
            <div className="slider-row">
              <input type="range" min={0.5} max={4} step={0.1} value={trailLineWidth}
                onChange={(e) => setTrailLineWidth(Number(e.target.value))} />
              <span className="control-val">{trailLineWidth.toFixed(1)} px</span>
            </div>

            <label>Tail Min Alpha</label>
            <div className="slider-row">
              <input type="range" min={0} max={1} step={0.01} value={trailTailMin}
                onChange={(e) => setTrailTailMin(Number(e.target.value))} />
              <span className="control-val">{trailTailMin.toFixed(2)}</span>
            </div>

            <label>Tail Exp</label>
            <div className="slider-row">
              <input type="range" min={0.5} max={6} step={0.1} value={trailTailExp}
                onChange={(e) => setTrailTailExp(Number(e.target.value))} />
              <span className="control-val">{trailTailExp.toFixed(1)}</span>
            </div>

            <label>Max Lifetime</label>
            <div className="slider-row">
              <input type="range" min={1} max={300} step={1} value={maxLifetime}
                onChange={(e) => setMaxLifetime(Number(e.target.value))} />
              <span className="control-val">{maxLifetime}</span>
            </div>

            <div className="spacer" />
            <label>Selection</label>
            <div>
              <button onClick={clearSelection} style={{ height: 32, padding: '0 10px', borderRadius: 8, border: '1px solid #e5e7eb', background: '#fff' }}>Clear (C)</button>
            </div>
            <div className="spacer" />
            {error && <div className="hint" style={{ color: '#b91c1c' }}>{error}</div>}
          </div>
        </div>
      </div>
    </div>
  )
}
