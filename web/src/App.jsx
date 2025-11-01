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
          <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: 8, alignItems: 'center' }}>
            <label>Algorithm</label>
            <select value={algo} onChange={(e) => setAlgo(e.target.value)}>
              <option value="tsne">t-SNE</option>
              <option value="mds">MDS</option>
            </select>
            <label>Top-K</label>
            <input type="number" min={1} value={topK} onChange={(e) => setTopK(e.target.value)} />
            <label>Grid Res</label>
            <input type="number" min={8} max={200} value={gridRes} onChange={(e) => setGridRes(e.target.value)} />
          </div>
          <div style={{ marginTop: 8 }}>
            <button onClick={handleCompute} disabled={!dataset || busy}>{busy ? 'Computing…' : 'Compute'}</button>
          </div>
          {error && <div style={{ color: 'crimson', marginTop: 8 }}>{error}</div>}
        </div>
        <div style={{ flex: 1 }}>
          {payload ? (
            <div style={{ display: 'flex', gap: 16 }}>
              <CanvasWind payload={payload} onHover={setHoverPos} />
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
