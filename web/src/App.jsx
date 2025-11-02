import React, { useState, useEffect, useMemo } from 'react'
import './styles.css'
import { uploadFile, compute, recolor } from './services/api'
import CanvasWind from './components/CanvasWind.jsx'
import WindVane from './components/WindVane.jsx'
import ColorLegend from './components/ColorLegend.jsx'

export default function App() {
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
  const [consistentSpeed, setConsistentSpeed] = useState(true)
  const [speedConstRel, setSpeedConstRel] = useState(0.06)
  const [speedScale, setSpeedScale] = useState(1.0)
  const [tailLength, setTailLength] = useState(10)
  const [trailTailMin, setTrailTailMin] = useState(0.10)
  const [trailTailExp, setTrailTailExp] = useState(2.0)
  const [maxLifetime, setMaxLifetime] = useState(200)
  const [maskBufferFactor, setMaskBufferFactor] = useState(0.2)
  const [showHull, setShowHull] = useState(false)
  // Manual feature selection (overrides Top-K for visualization when non-empty)
  const [selectedFeatureIndices, setSelectedFeatureIndices] = useState([])
  const [featureFilter, setFeatureFilter] = useState('')

  // Clamp manual feature selection when payload changes (e.g., new dataset)
  useEffect(() => {
    const n = (payload?.col_labels?.length) || 0
    if (!n) { setSelectedFeatureIndices([]); return }
    setSelectedFeatureIndices((prev) => prev.filter((i) => i >= 0 && i < n))
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
      if (m > 0) setTopK(m)
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
    }, 200) // small debounce to avoid rapid recomputes while typing
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  // Compute which features are visible in the current Wind Vane view
  const visibleFeatures = useMemo(() => {
    const result = new Set()
    if (!payload) return result
    const { bbox = [0,1,0,1], grid_res = 25, uAll = [], vAll = [], selection = {}, unmasked = null, dominant = null } = payload
    const [xmin, xmax, ymin, ymax] = bbox
    const H = grid_res, W = grid_res
    // Determine selected features (indices rendered in vane)
    let indices = []
    if (Array.isArray(selectedFeatureIndices) && selectedFeatureIndices.length > 0) indices = selectedFeatureIndices
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
  }, [payload, selectedCells, vaneFocus, selectedFeatureIndices])

  return (
    <div className="app">
      <div className="header">
        <h2 className="title">Feature Wind Map</h2>
        <p className="subtitle">Interactive feature wind visualization</p>
      </div>
      <div className="content">
        <div className="main">
          {payload ? (
            <div className="row">
              <div className="panel canvas-frame">
                <p className="panel-title">Wind Map</p>
                <CanvasWind
                  payload={payload}
                  onHover={setHoverPos}
                  onSelectCell={({ i, j, shift }) => shift ? toggleCell(i, j) : setSingleCell(i, j)}
                  showGrid={showGrid}
                  particleCount={particleCount}
                  consistentSpeed={consistentSpeed}
                  speedConstRel={speedConstRel}
                  speedScale={speedScale}
                  tailLength={tailLength}
                  trailTailMin={trailTailMin}
                  trailTailExp={trailTailExp}
                  maxLifetime={maxLifetime}
                  size={720}
                  selectedCells={selectedCells}
                  featureIndices={selectedFeatureIndices && selectedFeatureIndices.length ? selectedFeatureIndices : null}
                />
              </div>
              <div className="panel canvas-frame">
                <p className="panel-title">Wind Vane{selectedCells.length > 0 ? ` (selection: ${selectedCells.length} cells)` : ''}</p>
                <WindVane payload={payload} focus={vaneFocus} selectedCells={selectedCells} size={420} showHull={showHull} featureIndices={selectedFeatureIndices && selectedFeatureIndices.length ? selectedFeatureIndices : null} />
              </div>
            </div>
          ) : (
            <div className="row">
              <div className="panel placeholder" style={{ width: 720, height: 720 }}>Wind Map</div>
              <div className="panel placeholder" style={{ width: 420, height: 420 }}>Wind Vane</div>
            </div>
          )}
        </div>
        <div className="row rows-below">
          <div className="panel padded controls-grid" style={{ flex: 2 }}>
            <label>Choose File</label>
            <input
              className="file-input"
              type="file"
              accept=".tmap,.json"
              onChange={(e) => {
                const f = e.target.files?.[0] || null
                setFile(f)
                if (f) handleUpload(f)
              }}
            />
            {(() => {
              const maxFeatures = (dataset?.col_labels?.length) || (payload?.col_labels?.length) || 100
              return (
                <>
                  <label>Top-K</label>
                  <div className="slider-row">
                    <input type="range" min={1} max={maxFeatures} step={1} value={topK}
                      disabled={selectedFeatureIndices.length > 0}
                      onChange={(e) => setTopK(Math.max(1, Number(e.target.value)))} />
                    <span className="control-val">{topK}</span>
                  </div>
                </>
              )
            })()}

            {payload && (
              <>
                <label>Feature Selection {selectedFeatureIndices.length > 0 ? `(${selectedFeatureIndices.length} selected)` : '(all)'}</label>
                <div style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
                  <input
                    type="text"
                    placeholder="Filter features..."
                    value={featureFilter}
                    onChange={(e) => setFeatureFilter(e.target.value)}
                    style={{ flex: 1, height: 28, padding: '0 8px', borderRadius: 6, border: '1px solid #e5e7eb' }}
                  />
                  <button
                    onClick={() => {
                      const n = (payload?.col_labels?.length) || 0
                      setSelectedFeatureIndices(Array.from({ length: n }, (_, i) => i))
                    }}
                    style={{ height: 28, padding: '0 10px', borderRadius: 6, border: '1px solid #e5e7eb', background: '#fff' }}
                  >All</button>
                  <button
                    onClick={() => setSelectedFeatureIndices([])}
                    style={{ height: 28, padding: '0 10px', borderRadius: 6, border: '1px solid #e5e7eb', background: '#fff' }}
                  >None</button>
                </div>
                <div style={{ maxHeight: 180, overflow: 'auto', border: '1px solid #e5e7eb', borderRadius: 6, padding: 8 }}>
                  {(payload.col_labels || []).map((name, idx) => {
                    const f = (featureFilter || '').toLowerCase()
                    if (f && !String(name).toLowerCase().includes(f)) return null
                    const checked = selectedFeatureIndices.includes(idx)
                    return (
                      <label key={idx} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '2px 0' }}>
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={(e) => {
                            setSelectedFeatureIndices((prev) => {
                              const set = new Set(prev)
                              if (e.target.checked) set.add(idx); else set.delete(idx)
                              return Array.from(set).sort((a,b)=>a-b)
                            })
                          }}
                        />
                        <span title={name} style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{name}</span>
                      </label>
                    )
                  })}
                </div>
                <div className="hint" style={{ marginTop: 6, color: '#6b7280' }}>Manual selection overrides Top‑K for visualization.</div>
              </>
            )}

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

            <label>Particles</label>
            <div className="slider-row">
              <input type="range" min={50} max={5000} step={50} value={particleCount}
                onChange={(e) => setParticleCount(Number(e.target.value))} />
              <span className="control-val">{particleCount}</span>
            </div>

            <label>Consistent Speed</label>
            <input type="checkbox" checked={consistentSpeed} onChange={(e) => setConsistentSpeed(e.target.checked)} />

            <label>Speed (const rel)</label>
            <div className="slider-row">
              <input type="range" min={0} max={0.2} step={0.005} value={speedConstRel}
                onChange={(e) => setSpeedConstRel(Number(e.target.value))} />
              <span className="control-val">{speedConstRel.toFixed(3)}</span>
            </div>

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
          </div>
          {error && <div className="hint" style={{ color: '#b91c1c', alignSelf: 'center' }}>{error}</div>}
        </div>
        <div className="panel padded color-panel">
          <p className="panel-title">Color Families</p>
          {payload ? (
            <ColorLegend
              payload={payload}
              dataset={dataset}
              visible={visibleFeatures}
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
      </div>
    </div>
  )
}
