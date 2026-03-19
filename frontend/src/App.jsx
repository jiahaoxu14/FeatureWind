import React, { useEffect, useMemo, useRef, useState } from 'react'
import './styles.css'
import {
  uploadFile,
  uploadTmapJson,
  compute,
  recolor,
  validateAnalysis,
  downloadJsonFile,
} from './services/api'
import CanvasWind from './components/CanvasWind.jsx'
import WindVane from './components/WindVane.jsx'
import ColorLegend from './components/ColorLegend.jsx'

function sanitizeFileStem(name) {
  return String(name || 'featurewind')
    .replace(/\.[^/.]+$/, '')
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'featurewind'
}

function formatNumber(value, digits = 3) {
  const num = Number(value)
  return Number.isFinite(num) ? num.toFixed(digits) : '-'
}

function validationModeLabel(mode) {
  switch (mode) {
    case 'exact-tsne-rerun':
      return 'Exact t-SNE rerun'
    case 'projection-not-yet-supported':
      return 'Unsupported projection'
    case 'domain-values-missing':
      return 'Domain values missing'
    case 'projection-metadata-missing':
    default:
      return 'Not available in this file'
  }
}

function colorWithAlpha(hex, alpha) {
  if (!hex || typeof hex !== 'string') return `rgba(37,99,235,${alpha})`
  const raw = hex.replace('#', '')
  const normalized = raw.length === 3 ? raw.split('').map((c) => c + c).join('') : raw
  if (normalized.length !== 6) return `rgba(37,99,235,${alpha})`
  const r = parseInt(normalized.slice(0, 2), 16)
  const g = parseInt(normalized.slice(2, 4), 16)
  const b = parseInt(normalized.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

async function readJsonFile(file) {
  const text = await file.text()
  return JSON.parse(text)
}

export default function App() {
  const fileInputRef = useRef(null)
  const sessionInputRef = useRef(null)
  const windMapCanvasRef = useRef(null)
  const windVaneCanvasRef = useRef(null)
  const canvasViewportRef = useRef(null)
  const skipValidationResetRef = useRef(false)

  const [dataset, setDataset] = useState(null)
  const [payload, setPayload] = useState(null)
  const [uploadedTmapData, setUploadedTmapData] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const [mode, setMode] = useState('analyze')
  const [selectionMode, setSelectionMode] = useState('point')
  const [selectedPointIndices, setSelectedPointIndices] = useState([])
  const [selectedFeatureIndex, setSelectedFeatureIndex] = useState(null)
  const [featureDelta, setFeatureDelta] = useState(0.1)

  const [gridRes, setGridRes] = useState(25)
  const [maskBufferFactor, setMaskBufferFactor] = useState(0.2)
  const [showGrid, setShowGrid] = useState(false)
  const [pointColorFeature, setPointColorFeature] = useState('')
  const [showPredictedTrail, setShowPredictedTrail] = useState(false)
  const [showSelectionVectors, setShowSelectionVectors] = useState(false)
  const [showParticlesPreview, setShowParticlesPreview] = useState(false)
  const [showWindVane, setShowWindVane] = useState(false)
  const [showFeatureFamilies, setShowFeatureFamilies] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [featureSearch, setFeatureSearch] = useState('')
  const [hoverPos, setHoverPos] = useState(null)

  const [validationBusy, setValidationBusy] = useState(false)
  const [validationResult, setValidationResult] = useState(null)

  const selectedPointKey = useMemo(() => selectedPointIndices.join(','), [selectedPointIndices])

  function resetForNewDataset() {
    setDataset(null)
    setPayload(null)
    setUploadedTmapData(null)
    setSelectedPointIndices([])
    setSelectionMode('point')
    setSelectedFeatureIndex(null)
    setFeatureDelta(0.1)
    setGridRes(25)
    setMaskBufferFactor(0.2)
    setShowGrid(false)
    setPointColorFeature('')
    setShowPredictedTrail(false)
    setShowSelectionVectors(false)
    setShowParticlesPreview(false)
    setShowWindVane(false)
    setShowFeatureFamilies(false)
    setShowAdvanced(false)
    setFeatureSearch('')
    setHoverPos(null)
    setValidationResult(null)
    setMode('analyze')
  }

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
    } catch {
      // ignore download errors in the browser
    }
  }

  function applySessionState(state) {
    const next = state || {}
    setMode(next.mode === 'validate' ? 'validate' : 'analyze')
    setSelectionMode(next.selectionMode === 'region' ? 'region' : 'point')
    setSelectedPointIndices(Array.isArray(next.selectedPointIndices) ? next.selectedPointIndices.map((idx) => Number(idx)).filter(Number.isFinite) : [])
    setSelectedFeatureIndex(Number.isInteger(next.selectedFeatureIndex) ? next.selectedFeatureIndex : null)
    setFeatureDelta(Number.isFinite(Number(next.featureDelta)) ? Number(next.featureDelta) : 0.1)
    setGridRes(Number.isFinite(Number(next.gridRes)) ? Number(next.gridRes) : 25)
    setMaskBufferFactor(Number.isFinite(Number(next.maskBufferFactor)) ? Number(next.maskBufferFactor) : 0.2)
    setShowGrid(Boolean(next.showGrid))
    setShowPredictedTrail(Boolean(next.showPredictedTrail))
    setShowSelectionVectors(Boolean(next.showSelectionVectors))
    setShowParticlesPreview(Boolean(next.showParticlesPreview))
    setShowWindVane(Boolean(next.showWindVane))
    setShowFeatureFamilies(Boolean(next.showFeatureFamilies))
    setShowAdvanced(Boolean(next.showAdvanced))
    setPointColorFeature(
      Number.isInteger(next.pointColorFeatureIndex) ? String(next.pointColorFeatureIndex) : ''
    )
  }

  async function restoreSession(sessionPayload) {
    try {
      setBusy(true)
      setError('')
      skipValidationResetRef.current = true
      resetForNewDataset()
      applySessionState(sessionPayload?.state || {})

      if (sessionPayload?.embeddedTmap) {
        setUploadedTmapData(sessionPayload.embeddedTmap)
        const res = await uploadTmapJson(
          sessionPayload.embeddedTmap,
          sessionPayload?.dataset?.sourceName || 'restored-session.tmap',
        )
        setDataset(res)
      } else {
        throw new Error('Session file is missing embedded tangent-map data.')
      }

      if (sessionPayload?.state?.validationResult) {
        setValidationResult(sessionPayload.state.validationResult)
      }
    } catch (e) {
      setError(e.message)
      skipValidationResetRef.current = false
    } finally {
      setBusy(false)
    }
  }

  async function handleUpload(selected) {
    const file = selected
    if (!file) return
    setError('')
    setBusy(true)
    try {
      const parsed = await readJsonFile(file).catch(() => null)
      if (parsed?.sessionType === 'featurewind-analysis-session') {
        await restoreSession(parsed)
        return
      }

      resetForNewDataset()
      setUploadedTmapData(parsed && Array.isArray(parsed.tmap) ? parsed : null)
      const res = await uploadFile(file)
      setDataset(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  async function handleLoadSession(selected) {
    const file = selected
    if (!file) return
    setError('')
    try {
      const parsed = await readJsonFile(file)
      if (parsed?.sessionType !== 'featurewind-analysis-session') {
        throw new Error('Please choose a FeatureWind session JSON file.')
      }
      await restoreSession(parsed)
    } catch (e) {
      setError(e.message)
    } finally {
      if (sessionInputRef.current) sessionInputRef.current.value = ''
    }
  }

  useEffect(() => {
    if (!dataset?.datasetId) return undefined

    let cancelled = false
    const timer = window.setTimeout(async () => {
      setBusy(true)
      setError('')
      try {
        const result = await compute({
          dataset_id: dataset.datasetId,
          gridRes: Number(gridRes),
          featureIndex: Number.isInteger(selectedFeatureIndex) ? selectedFeatureIndex : undefined,
          includeRawGradients: true,
          config: {
            maskBufferFactor: Number(maskBufferFactor),
          },
        })
        if (!cancelled) setPayload(result)
      } catch (e) {
        if (!cancelled) setError(e.message)
      } finally {
        if (!cancelled) setBusy(false)
      }
    }, 180)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [dataset?.datasetId, gridRes, maskBufferFactor, selectedFeatureIndex])

  useEffect(() => {
    if (!payload?.positions?.length) {
      setSelectedPointIndices([])
      return
    }
    setSelectedPointIndices((prev) => prev.filter((idx) => idx >= 0 && idx < payload.positions.length))
  }, [payload?.positions?.length])

  useEffect(() => {
    if (!payload?.col_labels?.length) {
      setSelectedFeatureIndex(null)
      return
    }
    if (selectedFeatureIndex != null && selectedFeatureIndex >= 0 && selectedFeatureIndex < payload.col_labels.length) {
      return
    }
    const fallback = payload?.feature_stats?.globalRanking?.[0]
    if (Number.isInteger(fallback)) setSelectedFeatureIndex(fallback)
  }, [payload?.col_labels?.length, payload?.feature_stats, selectedFeatureIndex])

  useEffect(() => {
    if (!payload?.col_labels?.length) {
      setPointColorFeature('')
      return
    }
    if (pointColorFeature === '') return
    const idx = Number(pointColorFeature)
    if (!Number.isInteger(idx) || idx < 0 || idx >= payload.col_labels.length) {
      setPointColorFeature('')
    }
  }, [payload?.col_labels?.length, pointColorFeature])

  useEffect(() => {
    if (skipValidationResetRef.current) {
      skipValidationResetRef.current = false
      return
    }
    setValidationResult(null)
  }, [dataset?.datasetId, selectedFeatureIndex, featureDelta, selectedPointKey])

  function handleSelectPoint({ idx, append }) {
    if (!Number.isInteger(idx)) return
    setSelectedPointIndices((prev) => {
      if (!append) return [idx]
      if (prev.includes(idx)) return prev
      return [...prev, idx].sort((a, b) => a - b)
    })
    setSelectionMode(append ? 'region' : 'point')
  }

  function handleBrushPoints({ indices }) {
    if (!Array.isArray(indices) || indices.length === 0) return
    setSelectedPointIndices((prev) => {
      const next = new Set(prev)
      for (const idx of indices) {
        const num = Number(idx)
        if (Number.isInteger(num)) next.add(num)
      }
      return Array.from(next).sort((a, b) => a - b)
    })
    setSelectionMode('region')
  }

  function clearSelection() {
    setSelectedPointIndices([])
    setSelectionMode('point')
  }

  const selectedPoints = useMemo(() => {
    if (!payload?.positions || !selectedPointIndices.length) return []
    return selectedPointIndices
      .map((idx) => {
        const pos = payload.positions[idx]
        return Array.isArray(pos) && pos.length === 2 ? { idx, x: Number(pos[0]), y: Number(pos[1]) } : null
      })
      .filter(Boolean)
  }, [payload?.positions, selectedPointKey])

  const selectionCentroid = useMemo(() => {
    if (!selectedPoints.length) return null
    const sum = selectedPoints.reduce((acc, point) => {
      acc.x += point.x
      acc.y += point.y
      return acc
    }, { x: 0, y: 0 })
    return {
      x: sum.x / selectedPoints.length,
      y: sum.y / selectedPoints.length,
    }
  }, [selectedPoints])

  const selectionLabelSummary = useMemo(() => {
    if (!payload?.point_labels || !selectedPointIndices.length) return 'No labels'
    const counts = new Map()
    for (const idx of selectedPointIndices) {
      const label = String(payload.point_labels[idx])
      counts.set(label, (counts.get(label) || 0) + 1)
    }
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([label, count]) => `${label}: ${count}`)
      .join(', ')
  }, [payload?.point_labels, selectedPointKey])

  const localFeatureRanking = useMemo(() => {
    const labels = Array.isArray(payload?.col_labels) ? payload.col_labels : []
    if (!labels.length) return []

    if (!Array.isArray(payload?.gradVectors) || payload.gradVectors.length === 0 || selectedPointIndices.length === 0) {
      const globalRanking = Array.isArray(payload?.feature_stats?.globalRanking)
        ? payload.feature_stats.globalRanking
        : labels.map((_, idx) => idx)
      const avgMagnitude = Array.isArray(payload?.feature_stats?.avgMagnitude)
        ? payload.feature_stats.avgMagnitude
        : []
      return globalRanking.map((index) => ({
        index,
        label: labels[index],
        magnitude: Number(avgMagnitude[index] || 0),
        dx: 0,
        dy: 0,
      }))
    }

    return labels
      .map((label, index) => {
        let dx = 0
        let dy = 0
        for (const pointIndex of selectedPointIndices) {
          const vector = payload.gradVectors?.[pointIndex]?.[index]
          dx += Number(vector?.[0] || 0)
          dy += Number(vector?.[1] || 0)
        }
        dx /= selectedPointIndices.length
        dy /= selectedPointIndices.length
        return {
          index,
          label,
          dx,
          dy,
          magnitude: Math.hypot(dx, dy),
        }
      })
      .sort((a, b) => b.magnitude - a.magnitude)
  }, [payload?.col_labels, payload?.feature_stats, payload?.gradVectors, selectedPointKey])

  useEffect(() => {
    if (selectedFeatureIndex != null) return
    const fallback = localFeatureRanking[0]?.index
    if (Number.isInteger(fallback)) setSelectedFeatureIndex(fallback)
  }, [localFeatureRanking, selectedFeatureIndex])

  const selectedFeatureSummary = useMemo(() => {
    if (!Number.isInteger(selectedFeatureIndex)) return null
    return localFeatureRanking.find((item) => item.index === selectedFeatureIndex) || null
  }, [localFeatureRanking, selectedFeatureIndex])

  const featureOptions = useMemo(() => {
    const labels = Array.isArray(payload?.col_labels) ? payload.col_labels : []
    const query = featureSearch.trim().toLowerCase()
    return labels
      .map((label, index) => ({ index, label }))
      .filter((item) => !query || item.label.toLowerCase().includes(query))
  }, [payload?.col_labels, featureSearch])

  const competingFeatureIndices = useMemo(() => {
    const top = localFeatureRanking.slice(0, 5).map((item) => item.index)
    if (Number.isInteger(selectedFeatureIndex) && !top.includes(selectedFeatureIndex)) {
      top.unshift(selectedFeatureIndex)
    }
    return top.slice(0, 5)
  }, [localFeatureRanking, selectedFeatureIndex])

  const analysisOverlay = useMemo(() => {
    if (!payload?.gradVectors || !selectedPoints.length || !Number.isInteger(selectedFeatureIndex)) return null
    const featureColor = payload?.colors?.[selectedFeatureIndex] || '#2563eb'
    const pointVectors = selectedPoints.map((point) => {
      const vector = payload.gradVectors?.[point.idx]?.[selectedFeatureIndex]
      const dx = Number(vector?.[0] || 0) * Number(featureDelta)
      const dy = Number(vector?.[1] || 0) * Number(featureDelta)
      return {
        index: point.idx,
        anchor: [point.x, point.y],
        dx,
        dy,
      }
    })
    const centroid = selectionCentroid || {
      x: pointVectors[0].anchor[0],
      y: pointVectors[0].anchor[1],
    }
    const centroidVector = pointVectors.reduce((acc, item) => {
      acc.dx += item.dx
      acc.dy += item.dy
      return acc
    }, { dx: 0, dy: 0 })
    centroidVector.dx /= pointVectors.length
    centroidVector.dy /= pointVectors.length

    const trail = Array.from({ length: 4 }, (_, idx) => {
      const t = (idx + 1) / 4
      return {
        x: centroid.x + centroidVector.dx * t,
        y: centroid.y + centroidVector.dy * t,
      }
    })

    return {
      color: featureColor,
      centroid,
      centroidVector,
      pointVectors,
      trail,
    }
  }, [payload?.gradVectors, payload?.colors, selectedPoints, selectedFeatureIndex, featureDelta, selectionCentroid])

  const validationOverlay = useMemo(() => {
    if (!validationResult?.centroid) return null
    return {
      actualColor: '#f97316',
      centroid: validationResult.centroid,
      points: Array.isArray(validationResult.points) ? validationResult.points : [],
    }
  }, [validationResult])

  const canAnalyze = Boolean(payload && analysisOverlay && Number.isInteger(selectedFeatureIndex) && selectedPointIndices.length > 0)
  const canValidate = Boolean(dataset?.supportsValidation && canAnalyze)
  const focusPoint = selectionCentroid || hoverPos || null

  const currentSession = useMemo(() => ({
    sessionType: 'featurewind-analysis-session',
    version: 1,
    createdAt: new Date().toISOString(),
    dataset: dataset
      ? {
          datasetId: dataset.datasetId,
          sourceName: dataset.sourceName,
          datasetType: dataset.datasetType || dataset.type,
          fileHash: dataset.fileHash,
          projectionMeta: dataset.projectionMeta || {},
          supportsValidation: dataset.supportsValidation,
          validationMode: dataset.validationMode,
        }
      : null,
    state: {
      mode,
      selectionMode,
      selectedPointIndices,
      selectedFeatureIndex,
      featureDelta,
      gridRes,
      maskBufferFactor,
      pointColorFeatureIndex: pointColorFeature === '' ? null : Number(pointColorFeature),
      showGrid,
      showPredictedTrail,
      showSelectionVectors,
      showParticlesPreview,
      showWindVane,
      showFeatureFamilies,
      showAdvanced,
      validationResult,
    },
    embeddedTmap: uploadedTmapData && Array.isArray(uploadedTmapData.tmap) ? uploadedTmapData : null,
  }), [
    dataset,
    mode,
    selectionMode,
    selectedPointIndices,
    selectedFeatureIndex,
    featureDelta,
    gridRes,
    maskBufferFactor,
    pointColorFeature,
    showGrid,
    showPredictedTrail,
    showSelectionVectors,
    showParticlesPreview,
    showWindVane,
    showFeatureFamilies,
    showAdvanced,
    validationResult,
    uploadedTmapData,
  ])

  async function handleRunValidation() {
    if (!canValidate) return
    setValidationBusy(true)
    setError('')
    try {
      const result = await validateAnalysis({
        dataset_id: dataset.datasetId,
        pointIndices: selectedPointIndices,
        featureIndex: selectedFeatureIndex,
        delta: Number(featureDelta),
      })
      setValidationResult(result)
    } catch (e) {
      setError(e.message)
    } finally {
      setValidationBusy(false)
    }
  }

  function exportSessionOnly() {
    if (!dataset) return
    const stem = `${sanitizeFileStem(dataset.sourceName)}_${mode}`
    downloadJsonFile(currentSession, `${stem}.session.json`)
  }

  function exportFigure() {
    if (!windMapCanvasRef.current || !dataset) return
    const stamp = new Date().toISOString().replace(/[:.]/g, '-')
    const stem = `${sanitizeFileStem(dataset.sourceName)}_${mode}_${stamp}`
    saveCanvasAsPng(windMapCanvasRef.current, `${stem}.png`)
    downloadJsonFile(currentSession, `${stem}.session.json`)
  }

  const predictedVector = analysisOverlay?.centroidVector
  const selectedFeatureLabel = Number.isInteger(selectedFeatureIndex)
    ? payload?.col_labels?.[selectedFeatureIndex] || `Feature ${selectedFeatureIndex}`
    : 'Select a feature'
  const selectedFeatureColor = Number.isInteger(selectedFeatureIndex)
    ? payload?.colors?.[selectedFeatureIndex] || '#2563eb'
    : '#2563eb'

  return (
    <div className="app paper-shell">
      <header className="app-header">
        <div>
          <h1 className="title">FeatureWind Analysis</h1>
          <p className="subtitle">
            Inspect how one feature moves a selected point or region in the 2D embedding.
          </p>
        </div>

        <div className="header-actions">
          <button className="btn btn-primary" onClick={() => fileInputRef.current?.click()}>
            Upload .tmap
          </button>
          <button className="btn" onClick={() => sessionInputRef.current?.click()}>
            Load Session
          </button>
          <button className="btn" onClick={exportFigure} disabled={!dataset || !payload}>
            Export Figure
          </button>
          <button className="btn" onClick={exportSessionOnly} disabled={!dataset}>
            Export Session
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".tmap,.json"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0] || null
            if (file) handleUpload(file)
          }}
        />
        <input
          ref={sessionInputRef}
          type="file"
          accept=".json"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0] || null
            if (file) handleLoadSession(file)
          }}
        />
      </header>

      {error && <div className="banner banner-error">{error}</div>}

      <div className="mode-switch">
        <button
          className={`mode-pill ${mode === 'analyze' ? 'active' : ''}`}
          onClick={() => setMode('analyze')}
        >
          Analyze
        </button>
        <button
          className={`mode-pill ${mode === 'validate' ? 'active' : ''}`}
          onClick={() => setMode('validate')}
        >
          Validate
        </button>
      </div>

      <div className="workspace">
        <aside className="sidebar">
          <section className="panel section">
            <div className="section-header">
              <h2>Dataset</h2>
              {busy && <span className="status-pill">Computing</span>}
            </div>
            {dataset ? (
              <div className="section-body stack">
                <div className="meta-list">
                  <div><span>File</span><strong>{dataset.sourceName || 'Uploaded dataset'}</strong></div>
                  <div><span>Type</span><strong>{dataset.datasetType || dataset.type || 'tmap'}</strong></div>
                  <div><span>Points</span><strong>{payload?.positions?.length ?? dataset.pointCount ?? '-'}</strong></div>
                  <div><span>Features</span><strong>{payload?.col_labels?.length ?? dataset.col_labels?.length ?? '-'}</strong></div>
                  <div><span>Validation</span><strong>{validationModeLabel(payload?.validationMode || dataset.validationMode)}</strong></div>
                </div>
                <p className="hint">
                  Validation is exact only for t-SNE tangent maps generated with projection metadata.
                </p>
              </div>
            ) : (
              <div className="section-body">
                <p className="hint">Upload a `.tmap` file to begin analysis.</p>
              </div>
            )}
          </section>

          <section className="panel section">
            <div className="section-header">
              <h2>Target</h2>
              {selectedPointIndices.length > 0 && (
                <span className="status-pill">{selectionMode === 'region' ? 'Region' : 'Point'}</span>
              )}
            </div>
            <div className="section-body stack">
              {selectedPointIndices.length === 0 ? (
                <p className="hint">
                  Click a point to inspect it. Shift-click or drag on nearby points to build a region.
                </p>
              ) : (
                <>
                  <div className="meta-list">
                    <div><span>Selection</span><strong>{selectedPointIndices.length} point{selectedPointIndices.length === 1 ? '' : 's'}</strong></div>
                    <div><span>Centroid X</span><strong>{formatNumber(selectionCentroid?.x)}</strong></div>
                    <div><span>Centroid Y</span><strong>{formatNumber(selectionCentroid?.y)}</strong></div>
                    <div><span>Labels</span><strong>{selectionLabelSummary}</strong></div>
                  </div>
                  <button className="btn btn-subtle" onClick={clearSelection}>Clear Selection</button>
                </>
              )}
            </div>
          </section>

          <section className="panel section">
            <div className="section-header">
              <h2>Feature</h2>
              {Number.isInteger(selectedFeatureIndex) && (
                <span className="feature-pill" style={{ background: colorWithAlpha(selectedFeatureColor, 0.18), color: selectedFeatureColor }}>
                  {selectedFeatureLabel}
                </span>
              )}
            </div>
            <div className="section-body stack">
              <input
                className="text-input"
                type="text"
                placeholder="Search features"
                value={featureSearch}
                onChange={(e) => setFeatureSearch(e.target.value)}
                disabled={!payload?.col_labels?.length}
              />

              <div className="feature-chip-row">
                {localFeatureRanking.slice(0, 6).map((item) => (
                  <button
                    key={item.index}
                    className={`feature-chip ${item.index === selectedFeatureIndex ? 'active' : ''}`}
                    onClick={() => setSelectedFeatureIndex(item.index)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>

              <select
                className="feature-select"
                size={Math.min(10, Math.max(5, featureOptions.length || 5))}
                value={Number.isInteger(selectedFeatureIndex) ? String(selectedFeatureIndex) : ''}
                onChange={(e) => setSelectedFeatureIndex(Number(e.target.value))}
                disabled={!featureOptions.length}
              >
                {featureOptions.map((item) => (
                  <option key={item.index} value={item.index}>
                    {item.label}
                  </option>
                ))}
              </select>

              <label className="field-label">Feature delta</label>
              <div className="slider-row">
                <input
                  type="range"
                  min={-0.5}
                  max={0.5}
                  step={0.01}
                  value={featureDelta}
                  onChange={(e) => setFeatureDelta(Number(e.target.value))}
                />
                <span className="control-val">{featureDelta.toFixed(2)}</span>
              </div>

              {selectedFeatureSummary && (
                <div className="meta-list compact">
                  <div><span>Local rank</span><strong>{localFeatureRanking.findIndex((item) => item.index === selectedFeatureIndex) + 1}</strong></div>
                  <div><span>Mean dx</span><strong>{formatNumber(selectedFeatureSummary.dx)}</strong></div>
                  <div><span>Mean dy</span><strong>{formatNumber(selectedFeatureSummary.dy)}</strong></div>
                  <div><span>Magnitude</span><strong>{formatNumber(selectedFeatureSummary.magnitude)}</strong></div>
                </div>
              )}
            </div>
          </section>

          <section className="panel section">
            <div className="section-header">
              <h2>View</h2>
            </div>
            <div className="section-body stack">
              <label className="toggle-row">
                <span>Show predicted trail</span>
                <input type="checkbox" checked={showPredictedTrail} onChange={(e) => setShowPredictedTrail(e.target.checked)} />
              </label>
              <label className="toggle-row">
                <span>Show per-point vectors</span>
                <input type="checkbox" checked={showSelectionVectors} onChange={(e) => setShowSelectionVectors(e.target.checked)} />
              </label>
              <label className="toggle-row">
                <span>Show particles preview</span>
                <input type="checkbox" checked={showParticlesPreview} onChange={(e) => setShowParticlesPreview(e.target.checked)} />
              </label>
              <label className="toggle-row">
                <span>Show local inspector</span>
                <input type="checkbox" checked={showWindVane} onChange={(e) => setShowWindVane(e.target.checked)} />
              </label>
            </div>
          </section>

          {mode === 'validate' && (
            <section className="panel section">
              <div className="section-header">
                <h2>Validate</h2>
                {validationBusy && <span className="status-pill">Running</span>}
              </div>
              <div className="section-body stack">
                <p className="hint">
                  Run a perturb-and-compare rerun for the selected feature and selection.
                </p>
                <button className="btn btn-primary" onClick={handleRunValidation} disabled={!canValidate || validationBusy}>
                  Run Perturb-and-Compare
                </button>
                {!dataset?.supportsValidation && dataset && (
                  <p className="hint">
                    This file does not expose enough projection metadata for exact validation.
                  </p>
                )}
              </div>
            </section>
          )}

          <section className="panel section">
            <div className="section-header">
              <h2>Advanced</h2>
              <button className="btn btn-subtle" onClick={() => setShowAdvanced((prev) => !prev)}>
                {showAdvanced ? 'Hide' : 'Show'}
              </button>
            </div>
            {showAdvanced && (
              <div className="section-body stack">
                <label className="toggle-row">
                  <span>Show grid</span>
                  <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
                </label>
                <label className="toggle-row">
                  <span>Show feature families</span>
                  <input type="checkbox" checked={showFeatureFamilies} onChange={(e) => setShowFeatureFamilies(e.target.checked)} />
                </label>

                <label className="field-label">Grid resolution</label>
                <div className="slider-row">
                  <input type="range" min={10} max={120} step={1} value={gridRes} onChange={(e) => setGridRes(Number(e.target.value))} />
                  <span className="control-val">{gridRes}</span>
                </div>

                <label className="field-label">Mask buffer</label>
                <div className="slider-row">
                  <input type="range" min={0} max={2} step={0.05} value={maskBufferFactor} onChange={(e) => setMaskBufferFactor(Number(e.target.value))} />
                  <span className="control-val">{maskBufferFactor.toFixed(2)}</span>
                </div>

                <label className="field-label">Point color by feature</label>
                <select
                  className="text-input"
                  value={pointColorFeature}
                  onChange={(e) => setPointColorFeature(e.target.value)}
                  disabled={!payload?.feature_values}
                >
                  <option value="">None</option>
                  {Array.isArray(payload?.col_labels) && payload.col_labels.map((name, idx) => (
                    <option key={idx} value={String(idx)}>{name}</option>
                  ))}
                </select>
              </div>
            )}
          </section>
        </aside>

        <main className="main-column">
          <section className="panel canvas-panel">
            <div className="canvas-toolbar">
              <div className="canvas-toolbar-copy">
                <h2>Embedding View</h2>
                <p className="hint">
                  {selectedPointIndices.length
                    ? `Focused on ${selectionMode === 'region' ? 'a region' : 'one point'} and feature "${selectedFeatureLabel}".`
                    : 'Select one point or brush a region to start.'}
                </p>
                <p className="hint canvas-hint">
                  Scroll to zoom. Use the controls to zoom or reset the view.
                </p>
              </div>

              <div className="canvas-actions">
                <button className="btn btn-subtle" onClick={() => canvasViewportRef.current?.zoomOut()} disabled={!payload}>
                  Zoom Out
                </button>
                <button className="btn btn-subtle" onClick={() => canvasViewportRef.current?.zoomIn()} disabled={!payload}>
                  Zoom In
                </button>
                <button className="btn btn-subtle" onClick={() => canvasViewportRef.current?.reset()} disabled={!payload}>
                  Reset View
                </button>
              </div>
            </div>

            {payload ? (
              <div className="canvas-shell">
                <CanvasWind
                  payload={payload}
                  onHover={setHoverPos}
                  selectPointsMode
                  onSelectPoint={handleSelectPoint}
                  onBrushPoints={handleBrushPoints}
                  selectedPointIndices={selectedPointIndices}
                  showGrid={showGrid}
                  showParticles={showParticlesPreview}
                  particleCount={800}
                  speedScale={1.0}
                  tailLength={8}
                  trailTailMin={0.12}
                  trailTailExp={2.2}
                  maxLifetime={160}
                  width={1480}
                  height={880}
                  responsive
                  pointColorFeatureIndex={pointColorFeature !== '' ? Number(pointColorFeature) : null}
                  analysisOverlay={analysisOverlay}
                  showSelectionVectors={showSelectionVectors}
                  showPredictedTrail={showPredictedTrail}
                  validationOverlay={validationOverlay}
                  viewportControlsRef={canvasViewportRef}
                  onCanvasElement={(el) => { windMapCanvasRef.current = el }}
                />
              </div>
            ) : (
              <div className="placeholder canvas-placeholder">
                Upload a `.tmap` file to enter analysis mode.
              </div>
            )}
          </section>

          <section className="metric-grid">
            <div className="panel metric-card">
              <span className="metric-label">Feature</span>
              <strong>{selectedFeatureLabel}</strong>
            </div>
            <div className="panel metric-card">
              <span className="metric-label">Predicted dx</span>
              <strong>{formatNumber(predictedVector?.dx)}</strong>
            </div>
            <div className="panel metric-card">
              <span className="metric-label">Predicted dy</span>
              <strong>{formatNumber(predictedVector?.dy)}</strong>
            </div>
            <div className="panel metric-card">
              <span className="metric-label">Predicted magnitude</span>
              <strong>{formatNumber(predictedVector ? Math.hypot(predictedVector.dx, predictedVector.dy) : null)}</strong>
            </div>
          </section>

          {mode === 'validate' && (
            <section className="panel section">
              <div className="section-header">
                <h2>Validation Summary</h2>
                {validationResult?.status === 'ok' && <span className="status-pill ok">Ready</span>}
              </div>
              <div className="section-body stack">
                {validationResult ? (
                  <>
                    <div className="meta-list">
                      <div><span>Mean error</span><strong>{formatNumber(validationResult.metrics?.meanErrorNorm)}</strong></div>
                      <div><span>Mean cosine</span><strong>{formatNumber(validationResult.metrics?.meanCosineSimilarity)}</strong></div>
                      <div><span>Magnitude ratio</span><strong>{formatNumber(validationResult.metrics?.meanMagnitudeRatio)}</strong></div>
                      <div><span>Alignment RMSE</span><strong>{formatNumber(validationResult.metrics?.baselineAlignmentRmse)}</strong></div>
                    </div>
                    <div className="compare-strip">
                      <div className="compare-pill predicted">Predicted: ({formatNumber(validationResult.centroid?.predicted?.[0])}, {formatNumber(validationResult.centroid?.predicted?.[1])})</div>
                      <div className="compare-pill actual">Actual: ({formatNumber(validationResult.centroid?.actual?.[0])}, {formatNumber(validationResult.centroid?.actual?.[1])})</div>
                    </div>
                  </>
                ) : (
                  <p className="hint">Run validation to compare the Jacobian prediction with a rerun projection.</p>
                )}
              </div>
            </section>
          )}

          {showWindVane && payload && competingFeatureIndices.length > 0 && focusPoint && (
            <section className="panel section">
              <div className="section-header">
                <h2>Local Feature Inspector</h2>
              </div>
              <div className="section-body inline-panel">
                <WindVane
                  payload={payload}
                  focus={focusPoint}
                  selectedCells={[]}
                  useConvexHull={false}
                  showHull={false}
                  showLabels
                  featureIndices={competingFeatureIndices}
                  size={260}
                  onCanvasElement={(el) => { windVaneCanvasRef.current = el }}
                />
                <div className="stack grow">
                  <p className="hint">
                    This view is demoted from the default UI. Use it to compare the selected feature against the strongest local competitors.
                  </p>
                  <div className="feature-chip-row">
                    {competingFeatureIndices.map((idx) => (
                      <span key={idx} className={`feature-chip ${idx === selectedFeatureIndex ? 'active' : ''}`}>
                        {payload.col_labels?.[idx]}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          )}

          {showFeatureFamilies && payload && (
            <section className="panel section">
              <div className="section-header">
                <h2>Feature Families (Advanced)</h2>
              </div>
              <div className="section-body">
                <ColorLegend
                  payload={payload}
                  dataset={dataset}
                  visible={new Set(competingFeatureIndices)}
                  selectedFeatures={Number.isInteger(selectedFeatureIndex) ? [selectedFeatureIndex] : []}
                  onApplyFamilies={async (families) => {
                    try {
                      const res = await recolor(dataset.datasetId, families)
                      setPayload((prev) => prev ? { ...prev, colors: res.colors, family_assignments: res.family_assignments } : prev)
                    } catch (e) {
                      setError(e.message)
                    }
                  }}
                />
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  )
}
