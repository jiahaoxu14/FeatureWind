import React, { useEffect, useRef, useState } from 'react'
import './styles.css'
import { uploadFile, compute } from './services/api'
import CanvasWind from './components/CanvasWind.jsx'

export default function App() {
  const fileInputRef = useRef(null)
  const windMapCanvasRef = useRef(null)

  const [file, setFile] = useState(null)
  const [dataset, setDataset] = useState(null)
  const [payload, setPayload] = useState(null)
  const [topK, setTopK] = useState(0)
  const [showVectors, setShowVectors] = useState(true)
  // Higher grid resolution to sharpen LIC / vector field visualization
  const gridRes = 300
  const maskBufferFactor = 0.2
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  // Grid selection disabled for this view

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

  async function handleUpload(selected) {
    const f = selected || file
    if (!f) return
    setError('')
    try {
      const lower = (f.name || '').toLowerCase()
      if (!(lower.endsWith('.tmap') || lower.endsWith('.json'))) {
        setError('Please upload a .tmap (or JSON with tmap structure).')
        return
      }
      const res = await uploadFile(f)
      const m = Array.isArray(res.col_labels) ? res.col_labels.length : 0
      if (m > 0) setTopK(m)
      setDataset(res)
      setPayload(null)
    } catch (e) {
      setError(e.message)
    }
  }

  async function handleCompute(forDatasetId) {
    const dsId = forDatasetId || dataset?.datasetId
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

  useEffect(() => {
    const dsId = dataset?.datasetId
    if (!dsId) return
    handleCompute(dsId)
  }, [dataset?.datasetId, topK])

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
        {busy && <span className="hint">Computing…</span>}
      </div>

      {error && <div className="hint" style={{ color: '#b91c1c' }}>{error}</div>}

      <div className="content">
        <div className="panel canvas-frame">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <p className="panel-title" style={{ margin: 0 }}>Wind Map</p>
            {payload && (
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className="btn"
                  title="Toggle vector arrows"
                  onClick={() => setShowVectors(v => !v)}
                >{showVectors ? 'Hide Vectors' : 'Show Vectors'}</button>
                <button
                  className="btn"
                  title="Save Wind Map as PNG"
                  onClick={() => saveCanvasAsPng(windMapCanvasRef.current, 'wind_map.png')}
                >Save PNG</button>
              </div>
            )}
          </div>
          {payload ? (
            <CanvasWind
              payload={payload}
              showParticles={false}
              particleCount={0}
              allowGridSelection={false}
              showGrid={false}
              showPoints={false}
              showCellAggregatedGradients={showVectors}
              showCellGradients={false}
              showPointGradients={false}
              colorCells={false}
              showLIC={true}
              useMask={false}
              onCanvasElement={(el) => { windMapCanvasRef.current = el }}
            />
          ) : (
            <div className="panel placeholder" style={{ width: 600, height: 600 }}>Wind Map</div>
          )}
        </div>
      </div>
    </div>
  )
}
