import React, { useState } from 'react'
import { uploadFile, compute } from './services/api'
import CanvasWind from './components/CanvasWind.jsx'
import WindVane from './components/WindVane.jsx'

export default function App() {
  const [file, setFile] = useState(null)
  const [dataset, setDataset] = useState(null)
  const [algo, setAlgo] = useState('tsne')
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

  async function handleUpload() {
    if (!file) return
    setError('')
    try {
      const res = await uploadFile(file)
      setDataset(res)
    } catch (e) {
      setError(e.message)
    }
  }

  async function handleCompute() {
    if (!dataset) return
    setBusy(true)
    setError('')
    try {
      const res = await compute({
        dataset_id: dataset.datasetId,
        algorithm: algo,
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

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: 16 }}>
      <h2>FeatureWind Web (MVP)</h2>
      <div style={{ display: 'flex', gap: 24 }}>
        <div style={{ minWidth: 320 }}>
          <div style={{ marginBottom: 8 }}>
            <input type="file" accept=".tmap,.json,.csv" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            <button onClick={handleUpload} disabled={!file}>Upload</button>
          </div>
          {dataset && (
            <div style={{ marginBottom: 8 }}>
              <div>Dataset: <code>{dataset.datasetId}</code></div>
              <div>Type: {dataset.type}</div>
              <div>Features: {dataset.col_labels?.length ?? 0}</div>
            </div>
          )}
          <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: 8, alignItems: 'center' }}>
            <label>Algorithm</label>
            <select value={algo} onChange={(e) => setAlgo(e.target.value)}>
              <option value="tsne">t-SNE</option>
              <option value="mds">MDS</option>
            </select>
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
          <div style={{ marginTop: 8 }}>
            <button onClick={handleCompute} disabled={!dataset || busy}>{busy ? 'Computing…' : 'Compute'}</button>
          </div>
          {error && <div style={{ color: 'crimson', marginTop: 8 }}>{error}</div>}
        </div>
        <div style={{ flex: 1 }}>
          {payload ? (
            <div style={{ display: 'flex', gap: 16 }}>
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
              />
              <div>
                <div style={{ marginBottom: 8, fontSize: 13, color: '#666' }}>
                  Wind Vane {hoverPos ? `(x=${hoverPos.x.toFixed(3)}, y=${hoverPos.y.toFixed(3)})` : '(center)'}
                </div>
                <WindVane payload={payload} focus={hoverPos} />
              </div>
            </div>
          ) : (
            <div style={{ border: '1px dashed #aaa', width: 800, height: 600, display: 'grid', placeItems: 'center', color: '#888' }}>
              Upload a dataset and click Compute
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
