import React, { useState, useEffect } from 'react'
import './styles.css'
import { uploadFile, compute } from './services/api'
import CanvasWind from './components/CanvasWind.jsx'
import WindVane from './components/WindVane.jsx'

export default function App() {
  const [file, setFile] = useState(null)
  const [dataset, setDataset] = useState(null)
  const [topK, setTopK] = useState(5)
  const [gridRes, setGridRes] = useState(25)
  const [payload, setPayload] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [hoverPos, setHoverPos] = useState(null)
  // Interactive config (frontend + backend overrides)
  const [showGrid, setShowGrid] = useState(true)
  const [particleCount, setParticleCount] = useState(600)
  const [consistentSpeed, setConsistentSpeed] = useState(true)
  const [speedConstRel, setSpeedConstRel] = useState(0.06)
  const [speedScale, setSpeedScale] = useState(1.0)
  const [tailLength, setTailLength] = useState(10)
  const [trailTailMin, setTrailTailMin] = useState(0.10)
  const [trailTailExp, setTrailTailExp] = useState(2.0)
  const [maxLifetime, setMaxLifetime] = useState(200)
  const [maskBufferFactor, setMaskBufferFactor] = useState(0.2)

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

  return (
    <div className="app">
      <div className="header">
        <h2 className="title">FeatureWind Web</h2>
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
                  showGrid={showGrid}
                  particleCount={particleCount}
                  consistentSpeed={consistentSpeed}
                  speedConstRel={speedConstRel}
                  speedScale={speedScale}
                  tailLength={tailLength}
                  trailTailMin={trailTailMin}
                  trailTailExp={trailTailExp}
                  maxLifetime={maxLifetime}
                  size={580}
                />
              </div>
              <div className="panel canvas-frame">
                <p className="panel-title">Wind Vane {hoverPos ? `(x=${hoverPos.x.toFixed(3)}, y=${hoverPos.y.toFixed(3)})` : '(center)'}</p>
                <WindVane payload={payload} focus={hoverPos} size={580} />
              </div>
            </div>
          ) : (
            <div className="row">
              <div className="panel placeholder" style={{ width: 580, height: 580 }}>Wind Map</div>
              <div className="panel placeholder" style={{ width: 580, height: 580 }}>Wind Vane</div>
            </div>
          )}
        </div>
        <div className="row">
          <div className="panel padded file-row" style={{ flex: 1 }}>
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
            {dataset && (
              <div className="meta">
                <div>Dataset</div> <div><code>{dataset.datasetId}</code></div>
                <div>Type</div> <div>{dataset.type}</div>
                <div>Features</div> <div>{dataset.col_labels?.length ?? 0}</div>
              </div>
            )}
          </div>
          <div className="panel padded controls-grid" style={{ flex: 2 }}>
            <label>Top-K</label>
            <input type="number" min={1} value={topK} onChange={(e) => setTopK(e.target.value)} />
            <label>Grid Res</label>
            <input type="number" min={8} max={200} value={gridRes} onChange={(e) => setGridRes(e.target.value)} />
            <label>Mask Buffer</label>
            <input type="number" step="0.05" min={0} max={2} value={maskBufferFactor}
                   onChange={(e) => setMaskBufferFactor(e.target.value)} />
            <label>Show Grid</label>
            <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
            <label>Particles</label>
            <input type="number" min={50} max={5000} value={particleCount} onChange={(e) => setParticleCount(Number(e.target.value))} />
            <label>Consistent Speed</label>
            <input type="checkbox" checked={consistentSpeed} onChange={(e) => setConsistentSpeed(e.target.checked)} />
            <label>Speed (const rel)</label>
            <input type="number" step="0.01" min={0.0} max={1.0} value={speedConstRel} onChange={(e) => setSpeedConstRel(Number(e.target.value))} />
            <label>Speed Scale</label>
            <input type="number" step="0.1" min={0.1} max={10} value={speedScale} onChange={(e) => setSpeedScale(Number(e.target.value))} />
            <label>Tail Length</label>
            <input type="number" min={2} max={60} value={tailLength} onChange={(e) => setTailLength(Number(e.target.value))} />
            <label>Tail Min Alpha</label>
            <input type="number" step="0.05" min={0} max={1} value={trailTailMin} onChange={(e) => setTrailTailMin(Number(e.target.value))} />
            <label>Tail Exp</label>
            <input type="number" step="0.1" min={0.5} max={6} value={trailTailExp} onChange={(e) => setTrailTailExp(Number(e.target.value))} />
            <label>Max Lifetime</label>
            <input type="number" min={1} max={300} value={maxLifetime} onChange={(e) => setMaxLifetime(Number(e.target.value))} />
          </div>
          {error && <div className="hint" style={{ color: '#b91c1c', alignSelf: 'center' }}>{error}</div>}
        </div>
      </div>
    </div>
  )
}
