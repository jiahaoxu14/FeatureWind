import React, { useState, useEffect, useMemo, useRef } from 'react'
import './styles.css'
import { uploadFile, compute, exportStaticTrailFigures } from './services/api'
import CanvasWind from './components/CanvasWind.jsx'
import WindVane from './components/WindVane.jsx'
import ColorLegend from './components/ColorLegend.jsx'
import { DEFAULT_FEATURE_HUE, FEATURE_COLOR_AUTO, FEATURE_COLOR_OPTIONS, FEATURE_PALETTE, buildFeatureColorMap, buildLabelColorMap } from './utils/colors.js'

const MODE_DEFAULT = 'default'
const MODE_AGGREGATE = 'aggregate'
const MODE_COMPARE = 'compare'
const MODE_OVERVIEW = 'overview'
const COMPARE_MAX = 9
const COMPARE_PALETTE = FEATURE_PALETTE
const OVERVIEW_NEUTRAL = '#4b5563'
const WIND_MAP_TOOL_PAN = 'pan'
const WIND_MAP_TOOL_BRUSH = 'brush'
const BRUSH_SELECTION_MODE_DRAG = 'drag'
const BRUSH_SELECTION_MODE_SELECT = 'select'
const MAX_MASK_DILATE_RADIUS_CELLS = 10
const INTERPOLATION_OPTIONS = [
  { value: 'linear-nearest', label: 'Linear + Nearest Fallback' },
  { value: 'linear', label: 'Linear' },
  { value: 'nearest', label: 'Nearest' },
  { value: 'cubic-nearest', label: 'Cubic + Nearest Fallback' },
]

function sanitizeFeatureIndices(indices, count, cap = Infinity) {
  if (!Array.isArray(indices)) return []
  const seen = new Set()
  const out = []
  for (const raw of indices) {
    const idx = Number(raw)
    if (!Number.isInteger(idx) || idx < 0 || idx >= count || seen.has(idx)) continue
    seen.add(idx)
    out.push(idx)
    if (out.length >= cap) break
  }
  return out
}

function resolveFeatureRanking(payload, count) {
  if (!payload || !Array.isArray(payload.featureRanking)) return [...Array(count).keys()]
  const seen = new Set()
  const ordered = []
  for (const raw of payload.featureRanking) {
    const idx = Number(raw)
    if (!Number.isInteger(idx) || idx < 0 || idx >= count || seen.has(idx)) continue
    seen.add(idx)
    ordered.push(idx)
  }
  for (let idx = 0; idx < count; idx++) {
    if (!seen.has(idx)) ordered.push(idx)
  }
  return ordered
}

function coerceFeatureIndex(value, count) {
  const idx = Number(value)
  if (!Number.isInteger(idx) || idx < 0 || idx >= count) return null
  return idx
}

function sanitizePointIndices(indices, count) {
  if (!Array.isArray(indices)) return []
  const seen = new Set()
  const out = []
  for (const raw of indices) {
    const idx = Number(raw)
    if (!Number.isInteger(idx) || idx < 0 || idx >= count || seen.has(idx)) continue
    seen.add(idx)
    out.push(idx)
  }
  return out
}

export default function App() {
  const fileInputRef = useRef(null)
  const lastDatasetIdRef = useRef(null)
  const allowedFeatureColorSet = useMemo(() => new Set(FEATURE_PALETTE.map((color) => String(color).toLowerCase())), [])
  const [file, setFile] = useState(null)
  const [dataset, setDataset] = useState(null)
  const [gridRes, setGridRes] = useState(25)
  const [interpolationMethod, setInterpolationMethod] = useState('linear-nearest')
  const [payload, setPayload] = useState(null)
  const [busy, setBusy] = useState(false)
  const [exportBusy, setExportBusy] = useState(false)
  const [error, setError] = useState('')
  const [exportMessage, setExportMessage] = useState('')
  const [hoverPos, setHoverPos] = useState(null)
  const [selectedCells, setSelectedCells] = useState([])
  const [selectedPointIndices, setSelectedPointIndices] = useState([])
  const [staticTrailData, setStaticTrailData] = useState([])
  const [windMapView, setWindMapView] = useState(null)
  const [windMapTool, setWindMapTool] = useState(WIND_MAP_TOOL_PAN)
  const [brushSelectionMode, setBrushSelectionMode] = useState(BRUSH_SELECTION_MODE_DRAG)
  const [mode, setMode] = useState(MODE_DEFAULT)
  const [defaultFeatureIndex, setDefaultFeatureIndex] = useState(null)
  const [compareFeatureIndices, setCompareFeatureIndices] = useState([])
  const [featureColorOverrides, setFeatureColorOverrides] = useState({})
  const [featureMessage, setFeatureMessage] = useState('')
  const [advancedControlsOpen, setAdvancedControlsOpen] = useState(false)

  // Interactive config (frontend + backend overrides)
  const [showGrid, setShowGrid] = useState(true)
  const [showMaskOverlay, setShowMaskOverlay] = useState(false)
  const [particleCount, setParticleCount] = useState(100)
  const [speedScale, setSpeedScale] = useState(1.0)
  const [animatedTrailLengthPx, setAnimatedTrailLengthPx] = useState(140)
  const [trailLineWidth, setTrailLineWidth] = useState(3.2)
  const [maskDilateRadiusCells, setMaskDilateRadiusCells] = useState(1)
  const [showHull, setShowHull] = useState(false)
  const [showVectorLabels, setShowVectorLabels] = useState(false)
  const [showAllVectors, setShowAllVectors] = useState(false)
  const [hideParticles, setHideParticles] = useState(false)
  const [pointColorFeature, setPointColorFeature] = useState('')
  const [showPointGradients, setShowPointGradients] = useState(false)
  const [showPointAggGradients, setShowPointAggGradients] = useState(false)
  const [showCellGradients, setShowCellGradients] = useState(false)
  const [showCellAggGradients, setShowCellAggGradients] = useState(false)
  const [showParticleInits, setShowParticleInits] = useState(false)
  const [uniformPointShape, setUniformPointShape] = useState(false)
  const [showParticleArrowheads, setShowParticleArrowheads] = useState(false)
  const [pointSize, setPointSize] = useState(4.4)
  const [restrictSpawnToSelection, setRestrictSpawnToSelection] = useState(false)
  const [autoRespawnEnabled, setAutoRespawnEnabled] = useState(false)
  const [assessAllCells, setAssessAllCells] = useState(false)

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

  const featureCount = Array.isArray(payload?.col_labels) ? payload.col_labels.length : 0
  const pointCount = Array.isArray(payload?.positions) ? payload.positions.length : 0
  const allFeatureIndices = useMemo(() => [...Array(featureCount).keys()], [featureCount])
  const featureRanking = useMemo(() => resolveFeatureRanking(payload, featureCount), [payload, featureCount])
  const alphabeticalFeatureIndices = useMemo(() => {
    const ordered = [...Array(featureCount).keys()]
    ordered.sort((a, b) => String(payload?.col_labels?.[a] || '').localeCompare(String(payload?.col_labels?.[b] || '')))
    return ordered
  }, [featureCount, payload?.col_labels])

  const effectiveDefaultFeatureIndex = useMemo(() => {
    if (Number.isInteger(defaultFeatureIndex) && defaultFeatureIndex >= 0 && defaultFeatureIndex < featureCount) {
      return defaultFeatureIndex
    }
    const fromPayload = coerceFeatureIndex(payload?.defaultFeatureIndex, featureCount)
    if (fromPayload !== null) return fromPayload
    if (featureRanking.length > 0) return featureRanking[0]
    return null
  }, [defaultFeatureIndex, payload?.defaultFeatureIndex, featureCount, featureRanking])

  const selectedMultiFeatureIndices = useMemo(() => {
    return sanitizeFeatureIndices(compareFeatureIndices, featureCount)
  }, [compareFeatureIndices, featureCount])

  const effectiveCompareFeatureIndices = useMemo(() => {
    return sanitizeFeatureIndices(compareFeatureIndices, featureCount, COMPARE_MAX)
  }, [compareFeatureIndices, featureCount])

  const activeFeatureIndices = useMemo(() => {
    if (!payload) return []
    if (mode === MODE_OVERVIEW) return allFeatureIndices
    if (mode === MODE_AGGREGATE) return selectedMultiFeatureIndices
    if (mode === MODE_COMPARE) return effectiveCompareFeatureIndices
    return effectiveDefaultFeatureIndex !== null ? [effectiveDefaultFeatureIndex] : []
  }, [payload, mode, allFeatureIndices, selectedMultiFeatureIndices, effectiveCompareFeatureIndices, effectiveDefaultFeatureIndex])

  const visualizedFeatureIndices = useMemo(() => {
    if (!payload || mode === MODE_OVERVIEW) return []
    return sanitizeFeatureIndices(activeFeatureIndices, featureCount)
  }, [payload, mode, activeFeatureIndices, featureCount])

  const activeFeatureColorMap = useMemo(() => {
    const out = buildFeatureColorMap(featureCount)
    visualizedFeatureIndices.forEach((featureIdx, order) => {
      out[featureIdx] = COMPARE_PALETTE[order % COMPARE_PALETTE.length] || DEFAULT_FEATURE_HUE
    })
    for (const [rawIdx, rawColor] of Object.entries(featureColorOverrides || {})) {
      const idx = Number(rawIdx)
      const color = String(rawColor || '').trim().toLowerCase()
      if (!Number.isInteger(idx) || idx < 0 || idx >= featureCount || !allowedFeatureColorSet.has(color)) continue
      out[idx] = color
    }
    return out
  }, [featureCount, visualizedFeatureIndices, featureColorOverrides, allowedFeatureColorSet])

  // Label color map — stable per dataset, never derived from payload.colors
  const labelColorMap = useMemo(() => buildLabelColorMap(payload?.point_labels), [payload?.point_labels])
  const exportDatasetName = useMemo(() => {
    const raw = file?.name || payload?.datasetId || dataset?.datasetId || 'dataset'
    return String(raw).replace(/\.(tmap|json|csv)$/i, '')
  }, [file?.name, payload?.datasetId, dataset?.datasetId])

  const vaneFeatureIndices = useMemo(() => {
    if (mode === MODE_OVERVIEW) return allFeatureIndices
    return activeFeatureIndices
  }, [mode, activeFeatureIndices, allFeatureIndices])

  const gradientFeatureIndices = useMemo(() => {
    if (mode === MODE_OVERVIEW) return []
    return activeFeatureIndices
  }, [mode, activeFeatureIndices])

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
      setDataset(res)
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
        gridRes: Number(gridRes),
        interpolationMethod,
        config: { maskDilateRadiusCells: Number(maskDilateRadiusCells) }
      })

      const nextCount = Array.isArray(res.col_labels) ? res.col_labels.length : 0
      const nextRanking = resolveFeatureRanking(res, nextCount)
      const nextDefault = coerceFeatureIndex(res.defaultFeatureIndex, nextCount) ?? (nextRanking[0] ?? null)
      const isNewDataset = lastDatasetIdRef.current !== dsId

      setPayload(res)
      setFeatureMessage('')
      setSelectedPointIndices([])
      setStaticTrailData([])
      setExportMessage('')

      setDefaultFeatureIndex((prev) => {
        if (isNewDataset) return nextDefault
        if (Number.isInteger(prev) && prev >= 0 && prev < nextCount) return prev
        return nextDefault
      })

      setCompareFeatureIndices((prev) => {
        if (isNewDataset) return nextDefault !== null ? [nextDefault] : []
        if (Array.isArray(prev) && prev.length === 0) return []
        const cleaned = sanitizeFeatureIndices(prev, nextCount)
        if (cleaned.length > 0) return cleaned
        return nextDefault !== null ? [nextDefault] : []
      })

      if (isNewDataset) setMode(MODE_DEFAULT)
      lastDatasetIdRef.current = dsId
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    const dsId = dataset?.datasetId
    if (!dsId) return
    const t = setTimeout(() => {
      handleCompute(dsId)
    }, 200)
    return () => clearTimeout(t)
  }, [dataset?.datasetId, gridRes, maskDilateRadiusCells, interpolationMethod])

  useEffect(() => {
    const n = Array.isArray(payload?.col_labels) ? payload.col_labels.length : 0
    if (!n) {
      setDefaultFeatureIndex(null)
      setCompareFeatureIndices([])
      setSelectedPointIndices([])
      setStaticTrailData([])
      return
    }
    setDefaultFeatureIndex((prev) => {
      if (Number.isInteger(prev) && prev >= 0 && prev < n) return prev
      const fallback = coerceFeatureIndex(payload?.defaultFeatureIndex, n)
      if (fallback !== null) return fallback
      return featureRanking[0] ?? 0
    })
    setCompareFeatureIndices((prev) => {
      if (Array.isArray(prev) && prev.length === 0) return []
      const cleaned = sanitizeFeatureIndices(prev, n)
      if (cleaned.length > 0) return cleaned
      const fallback = coerceFeatureIndex(payload?.defaultFeatureIndex, n)
      if (fallback !== null) return [fallback]
      return featureRanking.length > 0 ? [featureRanking[0]] : []
    })
  }, [payload?.col_labels, payload?.defaultFeatureIndex, featureRanking])

  useEffect(() => {
    if (!Array.isArray(selectedPointIndices) || selectedPointIndices.length > 0) return
    setStaticTrailData([])
  }, [selectedPointIndices])

  useEffect(() => {
    setFeatureColorOverrides({})
  }, [payload?.datasetId])

  useEffect(() => {
    setFeatureColorOverrides((prev) => {
      if (!prev || typeof prev !== 'object') return {}
      const next = {}
      let changed = false
      for (const [rawIdx, rawColor] of Object.entries(prev)) {
        const idx = Number(rawIdx)
        const color = String(rawColor || '').trim().toLowerCase()
        if (!Number.isInteger(idx) || idx < 0 || idx >= featureCount || !allowedFeatureColorSet.has(color)) {
          changed = true
          continue
        }
        next[idx] = color
      }
      if (!changed && Object.keys(next).length === Object.keys(prev).length) return prev
      return next
    })
  }, [featureCount, allowedFeatureColorSet])

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

  function clearSelection() {
    setSelectedCells([])
    setSelectedPointIndices([])
    setStaticTrailData([])
  }

  function handleWindMapToolChange(nextTool) {
    if (nextTool === windMapTool) return
    setSelectedCells([])
    setSelectedPointIndices([])
    setStaticTrailData([])
    setHoverPos(null)
    setWindMapTool(nextTool)
  }

  async function handleExportStaticTrailFigureBundle() {
    if (!payload || staticTrailData.length === 0) return
    setExportBusy(true)
    setError('')
    setExportMessage('')
    try {
      let canvasSnapshotDataUrl = null
      try {
        if (windMapCanvasRef.current) {
          canvasSnapshotDataUrl = windMapCanvasRef.current.toDataURL('image/png')
        }
      } catch {}
      const result = await exportStaticTrailFigures({
        dataset_id: dataset?.datasetId || payload?.datasetId || null,
        datasetName: exportDatasetName,
        activeFeatureIndices,
        canvasSnapshotDataUrl,
        canvasView: windMapView,
        staticTrails: staticTrailData,
        payload: {
          positions: payload.positions,
          feature_values: payload.feature_values,
          col_labels: payload.col_labels,
          colors: [...Array(featureCount).keys()].map((idx) => activeFeatureColorMap[idx] || DEFAULT_FEATURE_HUE),
        },
      })
      const figureCount = Array.isArray(result?.figures) ? result.figures.length : 0
      const suffix = figureCount === 1 ? '' : 's'
      setExportMessage(
        result?.output_dir
          ? `Saved ${figureCount} trail figure${suffix} to ${result.output_dir}`
          : `Saved ${figureCount} trail figure${suffix}.`
      )
    } catch (e) {
      setError(e.message)
    } finally {
      setExportBusy(false)
    }
  }

  useEffect(() => {
    function onKey(e) { if (e.key === 'c' || e.key === 'C') clearSelection() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  function handleModeChange(nextMode) {
    setFeatureMessage('')
    if (nextMode === MODE_AGGREGATE) {
      setCompareFeatureIndices((prev) => {
        const cleaned = sanitizeFeatureIndices(prev, featureCount)
        if (cleaned.length > 0) return cleaned
        return effectiveDefaultFeatureIndex !== null ? [effectiveDefaultFeatureIndex] : []
      })
    } else if (nextMode === MODE_COMPARE) {
      setCompareFeatureIndices((prev) => {
        const cleaned = sanitizeFeatureIndices(prev, featureCount, COMPARE_MAX)
        if (cleaned.length > 0) return cleaned
        return effectiveDefaultFeatureIndex !== null ? [effectiveDefaultFeatureIndex] : []
      })
    } else if (nextMode === MODE_DEFAULT && (mode === MODE_AGGREGATE || mode === MODE_COMPARE)) {
      const nextDefault = effectiveCompareFeatureIndices[0] ?? effectiveDefaultFeatureIndex
      if (nextDefault !== null) setDefaultFeatureIndex(nextDefault)
    }
    setMode(nextMode)
  }

  function handleSelectAllFeatures() {
    setFeatureMessage('')
    setMode(MODE_OVERVIEW)
  }

  function handleSelectFeature(idx) {
    const next = Number(idx)
    if (!Number.isInteger(next) || next < 0 || next >= featureCount) return
    setFeatureMessage('')
    setDefaultFeatureIndex(next)
    setCompareFeatureIndices((prev) => {
      const cleaned = sanitizeFeatureIndices(prev, featureCount)
      return cleaned.length > 0 ? cleaned : [next]
    })
    setMode(MODE_DEFAULT)
  }

  function handleToggleCompareFeature(idx) {
    const nextIdx = Number(idx)
    if (!Number.isInteger(nextIdx) || nextIdx < 0 || nextIdx >= featureCount) return
    setFeatureMessage('')
    setCompareFeatureIndices((prev) => {
      const isCompareMode = mode === MODE_COMPARE
      const cleaned = sanitizeFeatureIndices(prev, featureCount, isCompareMode ? COMPARE_MAX : Infinity)
      const exists = cleaned.includes(nextIdx)
      if (exists) return cleaned.filter((value) => value !== nextIdx)
      if (isCompareMode && cleaned.length >= COMPARE_MAX) {
        setFeatureMessage(`Compare mode supports up to ${COMPARE_MAX} features.`)
        return cleaned
      }
      return [...cleaned, nextIdx]
    })
    setMode((prevMode) => (prevMode === MODE_AGGREGATE ? MODE_AGGREGATE : MODE_COMPARE))
  }

  function handleClearCompareFeatures() {
    setFeatureMessage('')
    setCompareFeatureIndices([])
    setMode((prevMode) => (prevMode === MODE_AGGREGATE ? MODE_AGGREGATE : MODE_COMPARE))
  }

  function handleSetFeatureColor(idx, value) {
    const featureIdx = Number(idx)
    const nextValue = String(value || '').trim().toLowerCase()
    if (!Number.isInteger(featureIdx) || featureIdx < 0 || featureIdx >= featureCount) return
    if (nextValue === FEATURE_COLOR_AUTO) {
      setFeatureColorOverrides((prev) => {
        if (!Object.prototype.hasOwnProperty.call(prev, featureIdx)) return prev
        const next = { ...prev }
        delete next[featureIdx]
        return next
      })
      return
    }
    if (!allowedFeatureColorSet.has(nextValue)) return
    setFeatureColorOverrides((prev) => {
      if (prev[featureIdx] === nextValue) return prev
      return { ...prev, [featureIdx]: nextValue }
    })
  }

  const vaneFocus = (() => {
    const bbox = payload?.bbox || [0, 1, 0, 1]
    const [xmin, xmax, ymin, ymax] = bbox
    const H = payload?.grid_res || 25
    const W = H
    if (selectedCells.length > 0) {
      const { i, j } = selectedCells[selectedCells.length - 1]
      const x = xmin + (j + 0.5) * (xmax - xmin) / W
      const y = ymin + (i + 0.5) * (ymax - ymin) / H
      return { x, y }
    }
    return hoverPos || null
  })()

  const avgActiveFeatures = useMemo(() => {
    try {
      if (!payload || activeFeatureIndices.length === 0) return null
      const { uAll = [], vAll = [], grid_res = 25, unmasked = null } = payload
      const H = grid_res
      const W = grid_res
      let total = 0
      let cells = 0
      const eps = 1e-9
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          if (Array.isArray(unmasked) && unmasked.length === H && unmasked[0].length === W && !unmasked[i][j]) continue
          cells += 1
          let cnt = 0
          for (const fi of activeFeatureIndices) {
            const u = (uAll[fi]?.[i]?.[j] ?? 0)
            const v = (vAll[fi]?.[i]?.[j] ?? 0)
            if (u * u + v * v > eps) cnt += 1
          }
          total += cnt
        }
      }
      if (cells === 0) return 0
      return total / cells
    } catch {
      return null
    }
  }, [payload, activeFeatureIndices])

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
          {busy && <span className="hint" style={{ marginTop: 0 }}>Computing…</span>}
        </div>
      </div>
      <div className="content">
        <div className="workspace">
          <div className="main-column">
            <div className="panel panel-section map-panel">
              <div className="panel-header">
                <p className="panel-title">Wind Map</p>
                <div className="toolbar-group">
                  <button
                    className={`toolbar-btn${windMapTool === WIND_MAP_TOOL_PAN ? ' active' : ''}`}
                    type="button"
                    onClick={() => handleWindMapToolChange(WIND_MAP_TOOL_PAN)}
                    title="Pan and zoom the wind map"
                  >
                    Pan
                  </button>
                  <button
                    className={`toolbar-btn${windMapTool === WIND_MAP_TOOL_BRUSH ? ' active' : ''}`}
                    type="button"
                    onClick={() => handleWindMapToolChange(WIND_MAP_TOOL_BRUSH)}
                    title="Drag a rectangle to select grid cells"
                  >
                    Brush
                  </button>
                  {payload && windMapTool === WIND_MAP_TOOL_BRUSH && (
                    <>
                      <button
                        className={`toolbar-btn${brushSelectionMode === BRUSH_SELECTION_MODE_DRAG ? ' active' : ''}`}
                        type="button"
                        onClick={() => setBrushSelectionMode(BRUSH_SELECTION_MODE_DRAG)}
                        title="Drag across the map to paint selected grid cells"
                      >
                        Drag
                      </button>
                      <button
                        className={`toolbar-btn${brushSelectionMode === BRUSH_SELECTION_MODE_SELECT ? ' active' : ''}`}
                        type="button"
                        onClick={() => setBrushSelectionMode(BRUSH_SELECTION_MODE_SELECT)}
                        title="Click individual grid cells to select them"
                      >
                        Select
                      </button>
                    </>
                  )}
                  {payload && (
                    <button
                      className="btn"
                      title="Save Wind Map as PNG"
                      onClick={() => saveCanvasAsPng(windMapCanvasRef.current, 'wind_map.png')}
                    >Save PNG</button>
                  )}
                  {payload && (
                    <button
                      className="btn"
                      type="button"
                      title="Save one figure per current static trail to output/paper_figures"
                      onClick={handleExportStaticTrailFigureBundle}
                      disabled={exportBusy || staticTrailData.length === 0 || !Array.isArray(payload?.feature_values)}
                    >{exportBusy ? 'Saving…' : 'Save Trail Figures'}</button>
                  )}
                </div>
              </div>
              {payload && windMapTool === WIND_MAP_TOOL_PAN && (
                <div className="panel-note">Pan mode: click points to toggle static trails. Click empty space to clear the current point selection.</div>
              )}
              {payload && windMapTool === WIND_MAP_TOOL_BRUSH && (
                <div className="panel-note">
                  {brushSelectionMode === BRUSH_SELECTION_MODE_DRAG
                    ? 'Brush mode (Drag): drag to paint grid cells. Hold Shift to add to the current selection.'
                    : 'Brush mode (Select): click a grid cell to select it. Hold Shift to add or remove cells from the current selection.'}
                </div>
              )}
              {payload && exportMessage && (
                <div className="panel-note">{exportMessage}</div>
              )}
              {payload ? (
                <div className="wind-map-shell">
                  <CanvasWind
                    payload={payload}
                    mode={mode}
                    interactionMode={windMapTool}
                    brushSelectionMode={brushSelectionMode}
                    featureColorMap={activeFeatureColorMap}
                    neutralColor={OVERVIEW_NEUTRAL}
                    onHover={setHoverPos}
                    onSelectCell={({ i, j, shift }) => shift ? toggleCell(i, j) : setSingleCell(i, j)}
                    onBrushCell={(brushPayload) => {
                      if (brushPayload && Array.isArray(brushPayload.cells)) {
                        const cells = brushPayload.cells
                        const replace = !!brushPayload.replace
                        setSelectedCells((prev) => {
                          const key = (c) => `${c.i}:${c.j}`
                          const base = replace ? [] : prev
                          const setPrev = new Set(base.map(key))
                          const out = base.slice()
                          for (const c of cells) {
                            const k = key(c)
                            if (!setPrev.has(k)) {
                              setPrev.add(k)
                              out.push({ i: c.i, j: c.j })
                            }
                          }
                          return out
                        })
                        return
                      }
                      const { i, j } = brushPayload || {}
                      if (typeof i === 'number' && typeof j === 'number') {
                        setSelectedCells((prev) => {
                          const exists = prev.some((c) => c.i === i && c.j === j)
                          if (exists) return prev
                          return [...prev, { i, j }]
                        })
                      }
                    }}
                    onBrushPoints={(brushPayload) => {
                      const replace = brushPayload?.replace !== false
                      const nextIndices = sanitizePointIndices(brushPayload?.indices, pointCount)
                      setSelectedPointIndices((prev) => {
                        if (replace) return nextIndices
                        const merged = sanitizePointIndices([...(Array.isArray(prev) ? prev : []), ...nextIndices], pointCount)
                        return merged
                      })
                    }}
                    showGrid={showGrid}
                    showMaskOverlay={showMaskOverlay}
                    onSelectPoint={({ idx, indices }) => {
                      if (Array.isArray(indices)) {
                        setSelectedPointIndices(sanitizePointIndices(indices, pointCount))
                        return
                      }
                      if (typeof idx === 'number' && idx >= 0) {
                        setSelectedPointIndices([idx])
                        return
                      }
                      setSelectedPointIndices([])
                    }}
                    selectedPointIndices={selectedPointIndices}
                    particleCount={particleCount}
                    speedScale={speedScale}
                    animatedTrailLengthPx={animatedTrailLengthPx}
                    showParticles={!hideParticles}
                    pointColorFeatureIndex={pointColorFeature !== '' ? Number(pointColorFeature) : null}
                    showPointGradients={showPointGradients}
                    showPointAggregatedGradients={showPointAggGradients}
                    showCellGradients={showCellGradients}
                    showCellAggregatedGradients={showCellAggGradients}
                    showParticleInits={showParticleInits}
                    gradientFeatureIndices={gradientFeatureIndices}
                    selectedCells={selectedCells}
                    featureIndices={activeFeatureIndices}
                    uniformPointShape={uniformPointShape}
                    showParticleArrowheads={showParticleArrowheads}
                    allowGridSelection={restrictSpawnToSelection}
                    restrictSpawnToSelection={restrictSpawnToSelection}
                    autoRespawnRate={autoRespawnEnabled ? 0.3 : 0.0}
                    trailLineWidth={trailLineWidth}
                    pointSize={pointSize}
                    labelColorMap={labelColorMap}
                    datasetId={payload?.datasetId || dataset?.datasetId || file?.name || ''}
                    onStaticTrailsChange={setStaticTrailData}
                    onViewStateChange={setWindMapView}
                    onCanvasElement={(el) => { windMapCanvasRef.current = el }}
                  />
                </div>
              ) : (
                <div className="placeholder wind-map-placeholder">Wind Map</div>
              )}
            </div>
          </div>

          <div className="sidebar-column">
            <div className="panel panel-section sidebar-panel vane-panel">
              <div className="panel-header">
                <p className="panel-title">Wind Vane{selectedCells.length > 0 ? ` (${selectedCells.length} cells)` : ''}</p>
                {payload && (
                  <button
                    className="btn"
                    title="Save Wind Vane as PNG"
                    onClick={() => saveCanvasAsPng(windVaneCanvasRef.current, 'wind_vane.png')}
                  >Save PNG</button>
                )}
              </div>
              {payload ? (
                <div className="wind-vane-shell">
                  <WindVane
                    payload={payload}
                    mode={mode}
                    focus={vaneFocus}
                    selectedCells={selectedCells}
                    useConvexHull={mode === MODE_COMPARE ? !showAllVectors : false}
                    showHull={mode === MODE_COMPARE ? showHull : false}
                    showLabels={mode === MODE_COMPARE ? showVectorLabels : false}
                    featureIndices={vaneFeatureIndices}
                    featureColorMap={activeFeatureColorMap}
                    neutralColor={OVERVIEW_NEUTRAL}
                    onCanvasElement={(el) => { windVaneCanvasRef.current = el }}
                  />
                </div>
              ) : (
                <div className="placeholder wind-vane-placeholder">Wind Vane</div>
              )}
            </div>

            <div className="panel padded panel-section sidebar-panel controls-panel">
              <div className="panel-header">
                <p className="panel-title">Controls</p>
              </div>
              <div className="controls-stack">
                <div className="controls-section">
                  <div className="controls-section-title">Basic</div>
                  <div className="controls-grid controls-grid-compact">
                    <label>Grid Res</label>
                    <div className="slider-row">
                      <input type="range" min={8} max={200} step={1} value={gridRes} onChange={(e) => setGridRes(Number(e.target.value))} />
                      <span className="control-val">{gridRes}</span>
                    </div>

                    <label>Interpolation</label>
                    <select value={interpolationMethod} onChange={(e) => setInterpolationMethod(e.target.value)}>
                      {INTERPOLATION_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>{option.label}</option>
                      ))}
                    </select>

                    <label>Mask Radius</label>
                    <div className="slider-row">
                      <input type="range" min={0} max={MAX_MASK_DILATE_RADIUS_CELLS} step={1} value={maskDilateRadiusCells} onChange={(e) => setMaskDilateRadiusCells(Number(e.target.value))} />
                      <span className="control-val">{maskDilateRadiusCells} cell{maskDilateRadiusCells === 1 ? '' : 's'}</span>
                    </div>

                    <label>Show Grid</label>
                    <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />

                    <label>Show Mask</label>
                    <input type="checkbox" checked={showMaskOverlay} onChange={(e) => setShowMaskOverlay(e.target.checked)} />

                    <label>Hide Particles</label>
                    <input type="checkbox" checked={hideParticles} onChange={(e) => setHideParticles(e.target.checked)} />

                    <label>Restrict Spawns</label>
                    <input type="checkbox" checked={restrictSpawnToSelection} onChange={(e) => setRestrictSpawnToSelection(e.target.checked)} />

                    <label>Selection</label>
                    <div className="selection-row">
                      <span className="selection-count">{selectedCells.length} cells</span>
                      <span className="selection-count">{selectedPointIndices.length} points</span>
                      <button
                        className="btn btn-small"
                        type="button"
                        onClick={clearSelection}
                        disabled={selectedCells.length === 0 && selectedPointIndices.length === 0}
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                </div>

                <div className="controls-section controls-section-advanced">
                  <button
                    className={`section-toggle${advancedControlsOpen ? ' open' : ''}`}
                    type="button"
                    onClick={() => setAdvancedControlsOpen((prev) => !prev)}
                  >
                    <span>Advanced</span>
                    <span>{advancedControlsOpen ? 'Hide' : 'Show'}</span>
                  </button>
                  {advancedControlsOpen && (
                    <div className="controls-grid">
                      <label>Show Convex Hull</label>
                      <input type="checkbox" checked={showHull} onChange={(e) => setShowHull(e.target.checked)} disabled={mode !== MODE_COMPARE} />

                      <label>Show Vector Labels</label>
                      <input type="checkbox" checked={showVectorLabels} onChange={(e) => setShowVectorLabels(e.target.checked)} disabled={mode !== MODE_COMPARE} />

                      <label>Show All Vectors</label>
                      <input type="checkbox" checked={showAllVectors} onChange={(e) => setShowAllVectors(e.target.checked)} disabled={mode !== MODE_COMPARE} />

                      <label>Point Color By</label>
                      <select
                        value={pointColorFeature}
                        onChange={(e) => setPointColorFeature(e.target.value)}
                        disabled={!payload || !Array.isArray(payload.feature_values)}
                      >
                        <option value="">None</option>
                        {alphabeticalFeatureIndices.map((idx) => (
                          <option key={idx} value={String(idx)}>{payload?.col_labels?.[idx]}</option>
                        ))}
                      </select>

                      <label>Particle Arrowheads</label>
                      <input type="checkbox" checked={showParticleArrowheads} onChange={(e) => setShowParticleArrowheads(e.target.checked)} />

                      <label>Show Point Gradients</label>
                      <input type="checkbox" checked={showPointGradients} onChange={(e) => setShowPointGradients(e.target.checked)} disabled={mode === MODE_OVERVIEW} />

                      <label>Show Aggregated Point Gradients</label>
                      <input type="checkbox" checked={showPointAggGradients} onChange={(e) => setShowPointAggGradients(e.target.checked)} />

                      <label>Show Cell Gradients</label>
                      <input type="checkbox" checked={showCellGradients} onChange={(e) => setShowCellGradients(e.target.checked)} disabled={mode === MODE_OVERVIEW} />

                      <label>Show Aggregated Cell Gradients</label>
                      <input type="checkbox" checked={showCellAggGradients} onChange={(e) => setShowCellAggGradients(e.target.checked)} />

                      <label>Show Particle Inits</label>
                      <input type="checkbox" checked={showParticleInits} onChange={(e) => setShowParticleInits(e.target.checked)} />

                      <label>Auto Respawn</label>
                      <input type="checkbox" checked={autoRespawnEnabled} onChange={(e) => setAutoRespawnEnabled(e.target.checked)} />

                      <label>Assess Coverage</label>
                      <div className="slider-row">
                        <input type="checkbox" checked={assessAllCells} onChange={(e) => setAssessAllCells(e.target.checked)} />
                        {assessAllCells && (
                          <span className="control-val">
                            {typeof avgActiveFeatures === 'number' ? avgActiveFeatures.toFixed(2) : '-'}
                          </span>
                        )}
                      </div>

                      <label>Particles</label>
                      <div className="slider-row">
                        <input type="range" min={50} max={5000} step={50} value={particleCount} onChange={(e) => setParticleCount(Number(e.target.value))} />
                        <span className="control-val">{particleCount}</span>
                      </div>

                      <label>Display Speed</label>
                      <div className="slider-row">
                        <input type="range" min={0.5} max={2.5} step={0.1} value={speedScale} onChange={(e) => setSpeedScale(Number(e.target.value))} />
                        <span className="control-val">{Math.round(120 * speedScale)} px/s</span>
                      </div>

                      <label>Animated Trail Length</label>
                      <div className="slider-row">
                        <input type="range" min={40} max={320} step={10} value={animatedTrailLengthPx} onChange={(e) => setAnimatedTrailLengthPx(Number(e.target.value))} />
                        <span className="control-val">{Math.round(animatedTrailLengthPx)} px</span>
                      </div>

                      <label>Trail Width</label>
                      <div className="slider-row">
                        <input type="range" min={1.0} max={6} step={0.1} value={trailLineWidth} onChange={(e) => setTrailLineWidth(Number(e.target.value))} />
                        <span className="control-val">{trailLineWidth.toFixed(1)} px</span>
                      </div>

                      <label>Point Size</label>
                      <div className="slider-row">
                        <input type="range" min={2.5} max={7} step={0.1} value={pointSize} onChange={(e) => setPointSize(Number(e.target.value))} />
                        <span className="control-val">{pointSize.toFixed(1)} px</span>
                      </div>

                    </div>
                  )}
                </div>

                {error && <div className="hint controls-error">{error}</div>}
              </div>
            </div>
          </div>

          <div className="features-column">
            <div className="panel padded panel-section features-panel">
              <div className="panel-header">
                <p className="panel-title">Features</p>
              </div>
              {payload ? (
                <ColorLegend
                  payload={payload}
                  mode={mode}
                  onChangeMode={handleModeChange}
                  defaultFeatureIndex={effectiveDefaultFeatureIndex}
                  compareFeatureIndices={selectedMultiFeatureIndices}
                  onSelectFeature={handleSelectFeature}
                  onSelectAll={handleSelectAllFeatures}
                  onToggleCompareFeature={handleToggleCompareFeature}
                  onClearCompare={handleClearCompareFeatures}
                  compareCap={COMPARE_MAX}
                  message={featureMessage}
                  activeFeatureColorMap={activeFeatureColorMap}
                  featureColorOverrides={featureColorOverrides}
                  featureColorOptions={FEATURE_COLOR_OPTIONS}
                  onSetFeatureColor={handleSetFeatureColor}
                  labelColorMap={labelColorMap}
                />
              ) : (
                <div className="hint">Upload a dataset to browse features</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
