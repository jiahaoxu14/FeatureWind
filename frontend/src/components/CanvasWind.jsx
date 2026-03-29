import React, { useEffect, useRef, useMemo, useState } from 'react'

function sumSelectedGrid(grids, indices) {
  if (!grids || grids.length === 0) return null
  const m = Array.isArray(indices) ? indices : [...Array(grids.length).keys()]
  const H = grids[0].length
  const W = grids[0][0].length
  const out = Array.from({ length: H }, () => new Float32Array(W))
  for (const idx of m) {
    const g = grids[idx]
    for (let i = 0; i < H; i++) {
      const gi = g[i]
      const oi = out[i]
      for (let j = 0; j < W; j++) oi[j] += gi[j]
    }
  }
  return out
}

function bilinearSample(grid, gx, gy) {
  const H = grid.length, W = grid[0].length
  const j0 = Math.floor(gx), i0 = Math.floor(gy)
  const j1 = Math.min(j0 + 1, W - 1), i1 = Math.min(i0 + 1, H - 1)
  const a = Math.min(Math.max(gx - j0, 0), 1)
  const b = Math.min(Math.max(gy - i0, 0), 1)
  const g00 = grid[i0][j0]
  const g01 = grid[i0][j1]
  const g10 = grid[i1][j0]
  const g11 = grid[i1][j1]
  const top = g00 * (1 - a) + g01 * a
  const bot = g10 * (1 - a) + g11 * a
  return top * (1 - b) + bot * b
}

export default function CanvasWind({
  payload,
  mode = 'default',
  interactionMode = 'pan',
  featureColorMap = null,
  neutralColor = '#4b5563',
  particleCount = 1000,
  onHover,
  onSelectCell,
  onBrushCell,
  onCanvasElement = null,
  allowGridSelection = false,
  selectPointsMode = false,
  pointBrushRadiusPx = 14,
  onSelectPoint = null,
  onBrushPoints = null,
  selectedPointIndices = [],
  showGrid = true,
  showMaskOverlay = false,
  showParticles = true,
  showPointGradients = false,
  showPointAggregatedGradients = false,
  showCellGradients = false,
  showCellAggregatedGradients = false,
  showParticleInits = false,
  uniformPointShape = false,
  showParticleArrowheads = false,
  trailLineWidth = 2.0,
  pointSize = 4.4,
  autoRespawnRate = 0.0,
  restrictSpawnToSelection = false,
  brushRadius = 1,
  gradientFeatureIndices = null,
  speedScale = 1.0,
  tailDurationSec = 1.2,
  trailTailMin = 0.10,
  trailTailExp = 2.0,
  lifetimeTailMultiplier = 3.0,
  size = null,
  width = null,
  height = null,
  selectedCells = [],
  featureIndices = null,
  pointColorFeatureIndex = null,
  labelColorMap = null,
}) {
  const containerRef = useRef(null)
  const canvasRef = useRef(null)
  const rafRef = useRef(0)
  const runningRef = useRef(false)
  const brushingRef = useRef(false)
  const lastBrushRef = useRef({ i: -1, j: -1 })
  const showParticlesRef = useRef(!!showParticles)
  const showMaskOverlayRef = useRef(!!showMaskOverlay)
  const showPointGradientsRef = useRef(!!showPointGradients)
  const showPointAggregatedGradientsRef = useRef(!!showPointAggregatedGradients)
  const showCellGradientsRef = useRef(!!showCellGradients)
  const showCellAggregatedGradientsRef = useRef(!!showCellAggregatedGradients)
  const showParticleInitsRef = useRef(!!showParticleInits)
  const uniformPointShapeRef = useRef(!!uniformPointShape)
  const showParticleArrowheadsRef = useRef(!!showParticleArrowheads)
  const allowGridSelectionRef = useRef(!!allowGridSelection)
  const restrictSpawnToSelectionRef = useRef(!!restrictSpawnToSelection)
  const autoRespawnRateRef = useRef(Number(autoRespawnRate) || 0)
  const brushRadiusRef = useRef(Math.max(1, Math.floor(brushRadius || 1)))
  const selectPointsModeRef = useRef(!!selectPointsMode)
  const pointBrushRadiusPxRef = useRef(Math.max(4, Math.floor(pointBrushRadiusPx || 14)))
  const selectedPointIndicesRef = useRef(Array.isArray(selectedPointIndices) ? selectedPointIndices : [])
  const staticTrailSeedsRef = useRef([])
  const staticTrailsRef = useRef([])
  // Point brushing state: only becomes true after movement threshold
  const pointPointerDownRef = useRef(false)
  const pointBrushingRef = useRef(false)
  const pointDownPosRef = useRef({ x: 0, y: 0 })
  const panPointerDownRef = useRef(false)
  const panningRef = useRef(false)
  const panStartPosRef = useRef({ x: 0, y: 0 })
  const panStartViewRef = useRef(null)
  const suppressClickRef = useRef(false)
  const areaBrushActiveRef = useRef(false)
  const areaBrushMovedRef = useRef(false)
  const areaBrushRectRef = useRef(null)
  const hoverCellRef = useRef(null)
  const hoverPointRef = useRef(null)
  const gradientFeatureIndicesRef = useRef(Array.isArray(gradientFeatureIndices) ? gradientFeatureIndices : [])
  const brushCbRef = useRef(onBrushCell)
  // Keep dynamic props in refs to avoid reinitializing particles on toggle
  const showGridRef = useRef(!!showGrid)
  const selectedRef = useRef(selectedCells)
  const explicitWidth = (typeof width === 'number' && width > 0) ? width : null
  const explicitHeight = (typeof height === 'number' && height > 0) ? height : null
  const fixedSize = (typeof size === 'number' && size > 0) ? size : null
  const [canvasSize, setCanvasSize] = useState(fixedSize || 600)
  // keep selection fresh for the draw loop without resetting particles
  useEffect(() => { selectedRef.current = selectedCells || [] }, [selectedCells])
  useEffect(() => { showGridRef.current = !!showGrid }, [showGrid])
  useEffect(() => { showMaskOverlayRef.current = !!showMaskOverlay }, [showMaskOverlay])
  useEffect(() => { brushCbRef.current = onBrushCell }, [onBrushCell])
  useEffect(() => { showParticlesRef.current = !!showParticles }, [showParticles])
  useEffect(() => { showPointGradientsRef.current = !!showPointGradients }, [showPointGradients])
  useEffect(() => { showPointAggregatedGradientsRef.current = !!showPointAggregatedGradients }, [showPointAggregatedGradients])
  useEffect(() => { showCellGradientsRef.current = !!showCellGradients }, [showCellGradients])
  useEffect(() => { showCellAggregatedGradientsRef.current = !!showCellAggregatedGradients }, [showCellAggregatedGradients])
  useEffect(() => { showParticleInitsRef.current = !!showParticleInits }, [showParticleInits])
  useEffect(() => { uniformPointShapeRef.current = !!uniformPointShape }, [uniformPointShape])
  useEffect(() => { showParticleArrowheadsRef.current = !!showParticleArrowheads }, [showParticleArrowheads])
  useEffect(() => { allowGridSelectionRef.current = !!allowGridSelection }, [allowGridSelection])
  useEffect(() => { restrictSpawnToSelectionRef.current = !!restrictSpawnToSelection }, [restrictSpawnToSelection])
  useEffect(() => { autoRespawnRateRef.current = Number(autoRespawnRate) || 0 }, [autoRespawnRate])
  useEffect(() => { brushRadiusRef.current = Math.max(1, Math.floor(brushRadius || 1)) }, [brushRadius])
  useEffect(() => { selectPointsModeRef.current = !!selectPointsMode }, [selectPointsMode])
  useEffect(() => { pointBrushRadiusPxRef.current = Math.max(4, Math.floor(pointBrushRadiusPx || 14)) }, [pointBrushRadiusPx])
  useEffect(() => { selectedPointIndicesRef.current = Array.isArray(selectedPointIndices) ? selectedPointIndices : [] }, [selectedPointIndices])
  useEffect(() => {
    staticTrailSeedsRef.current = []
    staticTrailsRef.current = []
    areaBrushRectRef.current = null
    hoverCellRef.current = null
    hoverPointRef.current = null
    brushingRef.current = false
    lastBrushRef.current = { i: -1, j: -1 }
  }, [interactionMode])

  // Expose canvas element to parent for saving snapshots
  useEffect(() => {
    if (typeof onCanvasElement === 'function') onCanvasElement(canvasRef.current)
    return () => { if (typeof onCanvasElement === 'function') onCanvasElement(null) }
  }, [onCanvasElement])
  useEffect(() => { gradientFeatureIndicesRef.current = Array.isArray(gradientFeatureIndices) ? gradientFeatureIndices : [] }, [gradientFeatureIndices])

  useEffect(() => {
    if (explicitWidth !== null || explicitHeight !== null) return
    if (fixedSize !== null) {
      setCanvasSize(fixedSize)
      return
    }
    const el = containerRef.current
    if (!el) return
    function measure() {
      const next = Math.max(240, Math.floor(el.clientWidth || 0))
      if (next > 0) setCanvasSize(next)
    }
    measure()
    if (typeof ResizeObserver === 'function') {
      const ro = new ResizeObserver(() => measure())
      ro.observe(el)
      return () => ro.disconnect()
    }
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [explicitWidth, explicitHeight, fixedSize])

  const {
    bbox = [0, 1, 0, 1],
    grid_res = 25,
    uAll = [],
    vAll = [],
    positions = [],
    point_labels = null,
    selection = {},
    dominant = null,
    unmasked = null,
    colors = [],
    feature_values = null,
  } = payload || {}

  const indices = useMemo(() => {
    if (Array.isArray(featureIndices)) return featureIndices // honor manual array even if empty
    if (!selection) return []
    if (selection.topKIndices) return selection.topKIndices
    if (selection.featureIndex !== undefined) return [selection.featureIndex]
    return []
  }, [selection, featureIndices])

  const uSum = useMemo(() => sumSelectedGrid(uAll, indices), [uAll, indices])
  const vSum = useMemo(() => sumSelectedGrid(vAll, indices), [vAll, indices])

  // Precompute min/max for selected feature column to color points
  const pointColorStats = useMemo(() => {
    const idx = (typeof pointColorFeatureIndex === 'number') ? pointColorFeatureIndex : null
    if (!Array.isArray(feature_values) || idx === null) return null
    let min = Infinity, max = -Infinity
    for (let r = 0; r < feature_values.length; r++) {
      const row = feature_values[r]
      if (!row || idx < 0 || idx >= row.length) continue
      const v = Number(row[idx])
      if (!Number.isFinite(v)) continue
      if (v < min) min = v
      if (v > max) max = v
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return null
    return { idx, min, max }
  }, [feature_values, pointColorFeatureIndex])

  // Precompute magnitude grid and a robust scale (p95) for alpha mapping
  const magInfo = useMemo(() => {
    if (!uSum || !vSum) return null
    const H = uSum.length, W = uSum[0].length
    const mags = new Float32Array(H * W)
    let k = 0
    for (let i = 0; i < H; i++) {
      const ui = uSum[i]
      const vi = vSum[i]
      for (let j = 0; j < W; j++, k++) {
        const u = ui[j], v = vi[j]
        mags[k] = Math.hypot(u, v)
      }
    }
    // Compute p99 (closer to Python’s 99th percentile usage)
    const arr = Array.from(mags)
    arr.sort((a, b) => a - b)
    const p99 = arr.length ? arr[Math.floor(arr.length * 0.99)] : 1.0
    return { H, W, mags, p99: p99 || 1.0 }
  }, [uSum, vSum])

  // Per-feature robust scale (p99) across the entire grid, independent of selection
  const p99ByFeature = useMemo(() => {
    if (!Array.isArray(uAll) || !Array.isArray(vAll) || uAll.length === 0) return null
    const n = Math.min(uAll.length, vAll.length)
    const H = uAll[0]?.length || 0
    const W = (H && uAll[0][0]) ? uAll[0][0].length : 0
    if (!H || !W) return null
    const out = new Float32Array(n)
    for (let fi = 0; fi < n; fi++) {
      const ui = uAll[fi]
      const vi = vAll[fi]
      if (!ui || !vi) { out[fi] = 1.0; continue }
      const mags = new Float32Array(H * W)
      let k = 0
      for (let i = 0; i < H; i++) {
        const urow = ui[i]
        const vrow = vi[i]
        for (let j = 0; j < W; j++, k++) {
          const u = (urow?.[j] ?? 0)
          const v = (vrow?.[j] ?? 0)
          mags[k] = Math.hypot(u, v)
        }
      }
      const arr = Array.from(mags)
      arr.sort((a, b) => a - b)
      const p99 = arr.length ? arr[Math.floor(arr.length * 0.99)] : 1.0
      out[fi] = p99 || 1.0
    }
    return out
  }, [uAll, vAll])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !uSum || !vSum) return
    const ctx = canvas.getContext('2d')
    const [xmin, xmax, ymin, ymax] = bbox
    const viewRef = { current: { xmin, xmax, ymin, ymax } }
    const Wpx = canvas.width
    const Hpx = canvas.height
    const W = grid_res, H = grid_res

    // Trail configuration (mirrors defaults in featurewind/config.py)
    const FIELD_EPS = 1e-8
    const FIXED_ADVECTION_STEP_SEC = 1 / 60
    const DISPLAY_SPEED_PX = Math.max(1, 120 * (Number(speedScale) || 1))
    const TAIL_DURATION_SEC = Math.max(0.1, Number(tailDurationSec) || 1.2)
    const TRAIL_TAIL_MIN = Math.max(0, Math.min(1, trailTailMin))
    const TRAIL_TAIL_EXP = Math.max(0.5, trailTailExp)
    const LIFETIME_TAIL_MULTIPLIER = Math.max(0.5, Number(lifetimeTailMultiplier) || 3.0)
    const MAX_LIFETIME_SEC = LIFETIME_TAIL_MULTIPLIER * TAIL_DURATION_SEC

    // Optional mask helpers
    let hasMask = Array.isArray(unmasked) && unmasked.length === H && unmasked[0].length === W
    const isOverviewMode = mode === 'overview'
    const isCompareMode = mode === 'compare'

    const dxWorld = (xmax - xmin) / W
    const dyWorld = (ymax - ymin) / H

    function hexToRgb(hex) {
      if (!hex || typeof hex !== 'string') return [20, 20, 20]
      const m = hex.match(/^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i)
      if (!m) return [20, 20, 20]
      return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)]
    }

    function worldToCell(x, y) {
      if (!Number.isFinite(x) || !Number.isFinite(y)) return null
      if (x < xmin || x > xmax || y < ymin || y > ymax) return null
      const j = Math.max(0, Math.min(W - 1, Math.floor((x - xmin) / (xmax - xmin) * W)))
      const i = Math.max(0, Math.min(H - 1, Math.floor((y - ymin) / (ymax - ymin) * H)))
      return { i, j }
    }

    function worldToGrid(x, y) {
      const gx = (x - xmin) / (xmax - xmin) * (W - 1)
      const gy = (y - ymin) / (ymax - ymin) * (H - 1)
      return [gx, gy]
    }

    function worldToScreen(x, y) {
      const v = viewRef.current
      const sx = (x - v.xmin) / (v.xmax - v.xmin) * Wpx
      const sy = Hpx - (y - v.ymin) / (v.ymax - v.ymin) * Hpx
      return [sx, sy]
    }

    function screenToWorld(cx, cy) {
      const v = viewRef.current
      return {
        x: v.xmin + (cx / Wpx) * (v.xmax - v.xmin),
        y: v.ymin + ((Hpx - cy) / Hpx) * (v.ymax - v.ymin),
      }
    }

    function clampView(nextXmin, nextXmax, nextYmin, nextYmax) {
      const fullW = xmax - xmin
      const fullH = ymax - ymin
      const viewW = nextXmax - nextXmin
      const viewH = nextYmax - nextYmin
      return {
        xmin: Math.max(xmin, Math.min(xmax - viewW, nextXmin)),
        xmax: Math.min(xmax, Math.max(xmin + viewW, nextXmax)),
        ymin: Math.max(ymin, Math.min(ymax - viewH, nextYmin)),
        ymax: Math.min(ymax, Math.max(ymin + viewH, nextYmax)),
        fullW,
        fullH,
      }
    }

    function dominantFeatureForCell(i, j) {
      if (Array.isArray(featureIndices) && featureIndices.length > 1) {
        let bestIdx = -1
        let bestMag2 = -1
        for (const fi of featureIndices) {
          const u = (uAll[fi]?.[i]?.[j] ?? 0)
          const v = (vAll[fi]?.[i]?.[j] ?? 0)
          const mag2 = u * u + v * v
          if (mag2 > bestMag2) { bestMag2 = mag2; bestIdx = fi }
        }
        return bestIdx
      }
      if (Array.isArray(featureIndices) && featureIndices.length === 1) return featureIndices[0]
      const fid = dominant?.[i]?.[j]
      return (typeof fid === 'number') ? fid : -1
    }

    function featureHex(fid) {
      if (isOverviewMode) return neutralColor
      if (featureColorMap && typeof featureColorMap === 'object' && typeof featureColorMap[fid] === 'string') {
        return featureColorMap[fid]
      }
      if (Array.isArray(colors) && fid >= 0 && fid < colors.length && typeof colors[fid] === 'string') {
        return colors[fid]
      }
      return neutralColor
    }

    function colorForCell(i, j) {
      if (isOverviewMode) return hexToRgb(neutralColor)
      const fid = dominantFeatureForCell(i, j)
      if (typeof fid === 'number' && fid >= 0) {
        return hexToRgb(featureHex(fid))
      }
      return hexToRgb(neutralColor)
    }

    function resolveFieldContext(featureIndex = null) {
      const useFeatureField = Number.isInteger(featureIndex) && featureIndex >= 0 && featureIndex < uAll.length && featureIndex < vAll.length
      const uGrid = useFeatureField ? uAll[featureIndex] : uSum
      const vGrid = useFeatureField ? vAll[featureIndex] : vSum
      const fieldP99 = (useFeatureField && p99ByFeature && featureIndex < p99ByFeature.length && p99ByFeature[featureIndex] > 0)
        ? p99ByFeature[featureIndex]
        : activeFieldP99
      const fieldWeakThreshold = Math.max(1e-6, 0.015 * fieldP99)
      return { useFeatureField, uGrid, vGrid, fieldP99, fieldWeakThreshold }
    }

    function sampleFieldMagnitudeAt(x, y, featureIndex = null) {
      const [gx, gy] = worldToGrid(x, y)
      const { uGrid, vGrid, fieldP99 } = resolveFieldContext(featureIndex)
      const u = bilinearSample(uGrid, gx, gy)
      const v = bilinearSample(vGrid, gx, gy)
      const mag = Math.hypot(u, v)
      return Math.max(0, Math.min(1, mag / Math.max(fieldP99, 1e-6)))
    }

    function isMaskedCell(i, j) {
      if (i < 0 || i >= H || j < 0 || j >= W) return true
      if (hasMask) return !unmasked[i][j]
      return false
    }

    function isMaskedAt(x, y, featureIndex = null) {
      const cell = worldToCell(x, y)
      if (!cell) return true
      const { i, j } = cell
      if (hasMask) return !unmasked[i][j]
      const [gx, gy] = worldToGrid(x, y)
      const { uGrid, vGrid, fieldWeakThreshold } = resolveFieldContext(featureIndex)
      const u = bilinearSample(uGrid, gx, gy)
      const v = bilinearSample(vGrid, gx, gy)
      if (Math.hypot(u, v) < fieldWeakThreshold) return true
      if (dominant && dominant[i] && typeof dominant[i][j] === 'number') {
        return dominant[i][j] === -1
      }
      return false
    }

    const activeFieldP99 = magInfo?.p99 || 1.0
    const weakThreshold = Math.max(1e-6, 0.015 * activeFieldP99)
    const compareFeatureIndices = (isCompareMode && Array.isArray(featureIndices))
      ? featureIndices.filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < uAll.length && idx < vAll.length)
      : []

    // Precompute list of valid cells for efficient, uniform respawn.
    const validSpawnCells = []
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        if (hasMask && !unmasked[i][j]) continue
        const mag = Math.hypot(uSum?.[i]?.[j] ?? 0, vSum?.[i]?.[j] ?? 0)
        if (mag >= weakThreshold) validSpawnCells.push([i, j])
      }
    }
    const validSpawnCellsByFeature = new Map()
    for (const featureIndex of compareFeatureIndices) {
      const { uGrid, vGrid, fieldWeakThreshold } = resolveFieldContext(featureIndex)
      const cells = []
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          if (hasMask && !unmasked[i][j]) continue
          const mag = Math.hypot(uGrid?.[i]?.[j] ?? 0, vGrid?.[i]?.[j] ?? 0)
          if (mag >= fieldWeakThreshold) cells.push([i, j])
        }
      }
      validSpawnCellsByFeature.set(featureIndex, cells)
    }
    function sampleActiveField(x, y, view = viewRef.current, featureIndex = null) {
      const cell = worldToCell(x, y)
      if (!cell || isMaskedCell(cell.i, cell.j)) {
        return { valid: false, reason: 'masked-or-out-of-bounds', u: 0, v: 0, mag: 0, dirX: 0, dirY: 0, screenDirX: 0, screenDirY: 0 }
      }
      const { uGrid, vGrid, fieldWeakThreshold } = resolveFieldContext(featureIndex)
      const [gx, gy] = worldToGrid(x, y)
      const u = bilinearSample(uGrid, gx, gy)
      const v = bilinearSample(vGrid, gx, gy)
      const mag = Math.hypot(u, v)
      if (!(Number.isFinite(mag) && mag > FIELD_EPS)) {
        return { valid: false, reason: 'zero-or-nonfinite-magnitude', u, v, mag, dirX: 0, dirY: 0, screenDirX: 0, screenDirY: 0 }
      }
      if (mag < fieldWeakThreshold) {
        return { valid: false, reason: 'below-weak-threshold', u, v, mag, dirX: 0, dirY: 0, screenDirX: 0, screenDirY: 0 }
      }
      const dirX = u / (mag + FIELD_EPS)
      const dirY = v / (mag + FIELD_EPS)
      const sxScale = Wpx / (view.xmax - view.xmin)
      const syScale = Hpx / (view.ymax - view.ymin)
      const screenDx = dirX * sxScale
      const screenDy = -dirY * syScale
      const screenMag = Math.hypot(screenDx, screenDy)
      if (!(Number.isFinite(screenMag) && screenMag > FIELD_EPS)) {
        return { valid: false, reason: 'zero-or-nonfinite-screen-direction', u, v, mag, dirX, dirY, screenDirX: 0, screenDirY: 0 }
      }
      return {
        valid: true,
        reason: null,
        u,
        v,
        mag,
        dirX,
        dirY,
        screenDirX: screenDx / screenMag,
        screenDirY: screenDy / screenMag,
      }
    }

    function worldDeltaFromScreenDirection(screenDirX, screenDirY, distancePx, view = viewRef.current) {
      const sxScale = Wpx / (view.xmax - view.xmin)
      const syScale = Hpx / (view.ymax - view.ymin)
      return {
        dx: (screenDirX * distancePx) / sxScale,
        dy: -(screenDirY * distancePx) / syScale,
      }
    }

    function advectAlongFieldRK2(x, y, dtSec, featureIndex = null) {
      const view = viewRef.current
      const start = sampleActiveField(x, y, view, featureIndex)
      if (!start.valid) {
        return {
          valid: false,
          reason: `start-${start.reason || 'invalid'}`,
          sample: start,
          start,
          midpoint: null,
          end: null,
          midX: null,
          midY: null,
          nx: null,
          ny: null,
        }
      }
      const distancePx = DISPLAY_SPEED_PX * dtSec
      const halfDelta = worldDeltaFromScreenDirection(start.screenDirX, start.screenDirY, distancePx * 0.5, view)
      const midX = x + halfDelta.dx
      const midY = y + halfDelta.dy
      const midpoint = sampleActiveField(midX, midY, view, featureIndex)
      if (!midpoint.valid) {
        return {
          valid: false,
          reason: `midpoint-${midpoint.reason || 'invalid'}`,
          sample: midpoint,
          start,
          midpoint,
          end: null,
          midX,
          midY,
          nx: null,
          ny: null,
        }
      }
      const fullDelta = worldDeltaFromScreenDirection(midpoint.screenDirX, midpoint.screenDirY, distancePx, view)
      const nx = x + fullDelta.dx
      const ny = y + fullDelta.dy
      if (!Number.isFinite(nx) || !Number.isFinite(ny)) {
        return {
          valid: false,
          reason: 'nonfinite-endpoint',
          sample: midpoint,
          start,
          midpoint,
          end: null,
          midX,
          midY,
          nx,
          ny,
        }
      }
      const end = sampleActiveField(nx, ny, view, featureIndex)
      if (!end.valid) {
        return {
          valid: false,
          reason: `end-${end.reason || 'invalid'}`,
          sample: end,
          start,
          midpoint,
          end,
          midX,
          midY,
          nx,
          ny,
        }
      }
      return { valid: true, reason: null, x: nx, y: ny, start, midpoint, end, midX, midY, nx, ny }
    }

    function resolveStaticTrailSeed(seed) {
      if (!seed || typeof seed !== 'object') return null
      if (typeof seed.pointIndex === 'number' && seed.pointIndex >= 0 && seed.pointIndex < positions.length) {
        const pt = positions[seed.pointIndex]
        if (Array.isArray(pt) && pt.length >= 2) return { x: pt[0], y: pt[1], pointIndex: seed.pointIndex }
      }
      if (typeof seed.x === 'number' && typeof seed.y === 'number') {
        return {
          x: seed.x,
          y: seed.y,
          pointIndex: Number.isInteger(seed.pointIndex) ? seed.pointIndex : null,
        }
      }
      return null
    }

    function roundDiagnosticValue(value) {
      return Number.isFinite(value) ? Number(value.toFixed(6)) : value
    }

    function snapshotCell(cell) {
      return cell ? { i: cell.i, j: cell.j } : null
    }

    function snapshotView(view) {
      if (!view) return null
      return {
        xmin: roundDiagnosticValue(view.xmin),
        xmax: roundDiagnosticValue(view.xmax),
        ymin: roundDiagnosticValue(view.ymin),
        ymax: roundDiagnosticValue(view.ymax),
      }
    }

    function snapshotFieldSample(sample) {
      if (!sample) return null
      return {
        valid: !!sample.valid,
        reason: sample.reason ?? null,
        u: roundDiagnosticValue(sample.u),
        v: roundDiagnosticValue(sample.v),
        mag: roundDiagnosticValue(sample.mag),
      }
    }

    function flattenStaticTrailStepRecord(record) {
      const cell = record?.cell ? `${record.cell.i},${record.cell.j}` : ''
      const nextCell = record?.nextCell ? `${record.nextCell.i},${record.nextCell.j}` : ''
      return {
        step: record?.step ?? null,
        dt: record?.dt ?? null,
        x: record?.x ?? null,
        y: record?.y ?? null,
        cell,
        visitsBefore: record?.visitsBefore ?? null,
        visitsAfter: record?.visitsAfter ?? null,
        gx: record?.gx ?? null,
        gy: record?.gy ?? null,
        startU: record?.start?.u ?? null,
        startV: record?.start?.v ?? null,
        startMag: record?.start?.mag ?? null,
        startReason: record?.start?.reason ?? null,
        midX: record?.midX ?? null,
        midY: record?.midY ?? null,
        midpointU: record?.midpoint?.u ?? null,
        midpointV: record?.midpoint?.v ?? null,
        midpointMag: record?.midpoint?.mag ?? null,
        midpointReason: record?.midpoint?.reason ?? null,
        nx: record?.nx ?? null,
        ny: record?.ny ?? null,
        nextCell,
        endU: record?.end?.u ?? null,
        endV: record?.end?.v ?? null,
        endMag: record?.end?.mag ?? null,
        endReason: record?.end?.reason ?? null,
        sameCellAsPrev: record?.sameCellAsPrev ?? null,
        sameCellAsNext: record?.sameCellAsNext ?? null,
        sameCellVisitCount: record?.sameCellVisitCount ?? null,
        recentNetDisplacement: record?.recentNetDisplacement ?? null,
        stepStopReason: record?.stepStopReason ?? null,
      }
    }

    function logStaticTrailDiagnostics(trail) {
      if (!trail?.diagnostics) return
      const { globalContext, summary, stepRecords } = trail.diagnostics
      console.groupCollapsed('[Wind Map] Static trail diagnostics', {
        pointIndex: globalContext?.seed?.pointIndex ?? null,
        stopReason: summary?.stopReason ?? trail.stopReason ?? 'unknown',
        stopStep: summary?.stopStep ?? trail.stopStep ?? 0,
        steps: Array.isArray(stepRecords) ? stepRecords.length : 0,
      })
      console.log('globalContext', globalContext)
      console.log('summary', summary)
      console.log('stepRecords', stepRecords)
      if (Array.isArray(stepRecords) && stepRecords.length > 0) {
        console.table(stepRecords.map((record) => flattenStaticTrailStepRecord(record)))
      }
      console.groupEnd()
    }

    function buildStaticTrail(startX, startY, options = {}) {
      const collectDiagnostics = !!options.collectDiagnostics
      const trailFeatureIndex = Number.isInteger(options.featureIndex) ? options.featureIndex : null
      const stepRecords = []
      const visitCounts = new Map()
      const cellPxX = Wpx / W
      const cellPxY = Hpx / H
      const cellDiagPx = Math.hypot(cellPxX, cellPxY)
      const stepPx = DISPLAY_SPEED_PX * FIXED_ADVECTION_STEP_SEC
      const recentWindowSize = 16
      const progressThreshold = Math.max(1e-6, Math.hypot(dxWorld, dyWorld) * 0.35)
      // The old fixed `visits > 24` rule was too coarse: a streamline can make
      // steady progress yet still need many fixed substeps to cross one coarse cell.
      const visitLimit = Math.max(64, Math.ceil((cellDiagPx / Math.max(stepPx, 1e-6)) * 2.0))
      const loopGuard = {
        visitLimit,
        recentWindowSize,
        progressThreshold: roundDiagnosticValue(progressThreshold),
      }
      const makeGlobalContext = () => ({
        seed: {
          x: roundDiagnosticValue(startX),
          y: roundDiagnosticValue(startY),
          pointIndex: Number.isInteger(options.pointIndex) ? options.pointIndex : null,
        },
        featureIndex: trailFeatureIndex,
        grid_res: grid_res,
        canvasSize: { Wpx, Hpx },
        bbox: [xmin, xmax, ymin, ymax].map((value) => roundDiagnosticValue(value)),
        speedScale: Number(speedScale) || 1,
        DISPLAY_SPEED_PX: roundDiagnosticValue(DISPLAY_SPEED_PX),
        weakThreshold: roundDiagnosticValue(weakThreshold),
        viewWindow: snapshotView(viewRef.current),
      })
      const summarizeVisitCounts = () => (
        Array.from(visitCounts.entries())
          .map(([key, visits]) => {
            const [i, j] = key.split(':').map((value) => Number(value))
            return { cell: { i, j }, visits }
          })
          .sort((a, b) => b.visits - a.visits)
          .slice(0, 8)
      )
      const finishTrail = (points, segments, stopReason, stopStep = 0, extraSummary = {}) => {
        const trail = {
          points,
          segments,
          start: { x: startX, y: startY },
          featureIndex: trailFeatureIndex,
          stopReason,
          stopStep,
          loopGuard,
        }
        if (extraSummary.loopGuardState) trail.loopGuardState = extraSummary.loopGuardState
        if (collectDiagnostics) {
          trail.diagnostics = {
            globalContext: makeGlobalContext(),
            summary: {
              stopReason,
              stopStep,
              totalPoints: points.length,
              totalSegments: segments.length,
              topRepeatedCells: summarizeVisitCounts(),
              loopGuard,
              ...extraSummary,
            },
            stepRecords,
          }
        }
        return trail
      }
      const startCell = worldToCell(startX, startY)
      if (!startCell || isMaskedCell(startCell.i, startCell.j)) {
        return finishTrail([], [], 'start-masked-or-out-of-bounds', 0, {
          startCell: snapshotCell(startCell),
          initialSample: null,
        })
      }
      const startSample = sampleActiveField(startX, startY, viewRef.current, trailFeatureIndex)
      if (!startSample.valid) {
        return finishTrail([], [], `start-${startSample.reason || 'invalid'}`, 0, {
          startCell: snapshotCell(startCell),
          initialSample: snapshotFieldSample(startSample),
        })
      }
      const points = [{ x: startX, y: startY }]
      const segments = []
      const maxSteps = Math.max(64, Math.min(4000, W * H * 4))
      const recentPositions = [{ x: startX, y: startY }]
      let sameCellKey = null
      let sameCellVisitCount = 0

      function computeRecentNetDisplacement(candidateX, candidateY) {
        const startIdx = Math.max(0, recentPositions.length - (recentWindowSize - 1))
        const startPos = recentPositions[startIdx] || recentPositions[0] || { x: candidateX, y: candidateY }
        return Math.hypot(candidateX - startPos.x, candidateY - startPos.y)
      }

      let x = startX
      let y = startY
      for (let step = 0; step < maxSteps; step++) {
        const cell = worldToCell(x, y)
        const [gx, gy] = worldToGrid(x, y)
        if (!cell || isMaskedCell(cell.i, cell.j)) {
          if (collectDiagnostics) {
            const prevRecord = stepRecords[stepRecords.length - 1] || null
            stepRecords.push({
              step,
              dt: FIXED_ADVECTION_STEP_SEC,
              x: roundDiagnosticValue(x),
              y: roundDiagnosticValue(y),
              cell: snapshotCell(cell),
              visitsBefore: null,
              visitsAfter: null,
              gx: roundDiagnosticValue(gx),
              gy: roundDiagnosticValue(gy),
              start: null,
              midX: null,
              midY: null,
              midpoint: null,
              nx: null,
              ny: null,
              nextCell: null,
              end: null,
              sameCellAsPrev: !!(prevRecord?.cell && cell && prevRecord.cell.i === cell.i && prevRecord.cell.j === cell.j),
              sameCellAsNext: null,
              stepStopReason: 'current-cell-masked-or-out-of-bounds',
            })
          }
          return finishTrail(points, segments, 'current-cell-masked-or-out-of-bounds', step)
        }
        const key = `${cell.i}:${cell.j}`
        const visitsBefore = visitCounts.get(key) || 0
        const visitsAfter = visitsBefore + 1
        visitCounts.set(key, visitsAfter)
        if (sameCellKey === key) {
          sameCellVisitCount += 1
        } else {
          sameCellKey = key
          sameCellVisitCount = 1
        }
        const prevRecord = stepRecords[stepRecords.length - 1] || null
        const stepRecord = collectDiagnostics ? {
          step,
          dt: FIXED_ADVECTION_STEP_SEC,
          x: roundDiagnosticValue(x),
          y: roundDiagnosticValue(y),
          cell: snapshotCell(cell),
          visitsBefore,
          visitsAfter,
          gx: roundDiagnosticValue(gx),
          gy: roundDiagnosticValue(gy),
          start: null,
          midX: null,
          midY: null,
          midpoint: null,
          nx: null,
          ny: null,
          nextCell: null,
          end: null,
          sameCellAsPrev: !!(prevRecord?.cell && prevRecord.cell.i === cell.i && prevRecord.cell.j === cell.j),
          sameCellAsNext: null,
          sameCellVisitCount,
          stepStopReason: null,
        } : null

        const advected = advectAlongFieldRK2(x, y, FIXED_ADVECTION_STEP_SEC, trailFeatureIndex)
        if (stepRecord) {
          stepRecord.start = snapshotFieldSample(advected.start)
          stepRecord.midX = roundDiagnosticValue(advected.midX)
          stepRecord.midY = roundDiagnosticValue(advected.midY)
          stepRecord.midpoint = snapshotFieldSample(advected.midpoint)
          stepRecord.nx = roundDiagnosticValue(advected.nx)
          stepRecord.ny = roundDiagnosticValue(advected.ny)
          stepRecord.end = snapshotFieldSample(advected.end)
        }
        if (!advected.valid) {
          if (stepRecord) {
            stepRecord.stepStopReason = advected.reason || 'rk2-invalid'
            stepRecords.push(stepRecord)
          }
          return finishTrail(points, segments, advected.reason || 'rk2-invalid', step)
        }
        const { x: nx, y: ny } = advected
        const nextCell = worldToCell(nx, ny)
        if (stepRecord) {
          stepRecord.nextCell = snapshotCell(nextCell)
          stepRecord.sameCellAsNext = !!(nextCell && nextCell.i === cell.i && nextCell.j === cell.j)
        }
        if (!nextCell || isMaskedCell(nextCell.i, nextCell.j)) {
          if (stepRecord) {
            stepRecord.stepStopReason = 'next-cell-masked-or-out-of-bounds'
            stepRecords.push(stepRecord)
          }
          return finishTrail(points, segments, 'next-cell-masked-or-out-of-bounds', step)
        }

        const stayedInSameCell = nextCell.i === cell.i && nextCell.j === cell.j
        const recentNetDisplacement = stayedInSameCell
          ? computeRecentNetDisplacement(nx, ny)
          : null
        if (stepRecord) {
          stepRecord.recentNetDisplacement = roundDiagnosticValue(recentNetDisplacement)
        }
        if (stayedInSameCell && sameCellVisitCount > visitLimit && recentNetDisplacement <= progressThreshold) {
          const loopGuardState = {
            cell: snapshotCell(cell),
            visitCount: sameCellVisitCount,
            recentNetDisplacement: roundDiagnosticValue(recentNetDisplacement),
          }
          if (stepRecord) {
            stepRecord.stepStopReason = 'loop-guard'
            stepRecord.loopGuardState = loopGuardState
            stepRecords.push(stepRecord)
          }
          return finishTrail(points, segments, 'loop-guard', step, { loopGuardState })
        }

        segments.push({
          x0: x,
          y0: y,
          x1: nx,
          y1: ny,
          rgb: trailFeatureIndex !== null ? hexToRgb(featureHex(trailFeatureIndex)) : colorForCell(cell.i, cell.j),
        })
        points.push({ x: nx, y: ny })
        recentPositions.push({ x: nx, y: ny })
        if (recentPositions.length > recentWindowSize) recentPositions.shift()
        if (stepRecord) stepRecords.push(stepRecord)
        x = nx
        y = ny
      }
      return finishTrail(points, segments, 'max-steps', maxSteps)
    }

    function buildStaticTrailsFromSeeds(seeds, options = {}) {
      if (!Array.isArray(seeds) || seeds.length === 0) return []
      const deduped = []
      const seen = new Set()
      for (const rawSeed of seeds) {
        const seed = resolveStaticTrailSeed(rawSeed)
        if (!seed) continue
        const key = Number.isInteger(seed.pointIndex) ? `p:${seed.pointIndex}` : `${seed.x}:${seed.y}`
        if (seen.has(key)) continue
        seen.add(key)
        deduped.push(seed)
      }
      const staticTrailFeatureIndices = (isCompareMode && Array.isArray(featureIndices) && featureIndices.length > 1)
        ? featureIndices
        : [null]
      return deduped.flatMap((seed) => (
        staticTrailFeatureIndices.map((featureIndex) => ({
          ...buildStaticTrail(seed.x, seed.y, { ...options, pointIndex: seed.pointIndex, featureIndex }),
          pointIndex: seed.pointIndex,
          featureIndex,
        }))
      ))
    }

    function setStaticTrailSeeds(seeds, options = {}) {
      const normalized = Array.isArray(seeds)
        ? seeds.map((seed) => resolveStaticTrailSeed(seed)).filter(Boolean)
        : []
      staticTrailSeedsRef.current = normalized
      staticTrailsRef.current = buildStaticTrailsFromSeeds(normalized, options)
      return staticTrailsRef.current
    }

    function shuffleCells(cells) {
      const out = cells.slice()
      for (let i = out.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        const tmp = out[i]
        out[i] = out[j]
        out[j] = tmp
      }
      return out
    }

    const spawnStateByKey = new Map()

    function nextSpawnCell(cells, key = '') {
      if (!Array.isArray(cells) || cells.length === 0) return null
      let state = spawnStateByKey.get(key)
      if (!state) {
        state = { key, order: [], index: 0 }
        spawnStateByKey.set(key, state)
      }
      if (state.index >= state.order.length) {
        state.order = shuffleCells(cells)
        state.index = 0
      }
      const cell = state.order[state.index] || null
      state.index += 1
      return cell
    }

    function spawnPointInCell(cell) {
      const [i, j] = cell
      const dx = (xmax - xmin) / W
      const dy = (ymax - ymin) / H
      return {
        x: xmin + j * dx + Math.random() * dx,
        y: ymin + i * dy + Math.random() * dy,
      }
    }

    function randomSpawn(featureIndex = null) {
      const featurePool = Number.isInteger(featureIndex)
        ? (validSpawnCellsByFeature.get(featureIndex) || [])
        : validSpawnCells
      // If restriction is enabled and we have selected cells, spawn within them (respecting mask)
      if (restrictSpawnToSelectionRef.current && Array.isArray(selectedRef.current) && selectedRef.current.length > 0) {
        const { uGrid, vGrid, fieldWeakThreshold } = resolveFieldContext(featureIndex)
        const valid = []
        for (const c of selectedRef.current) {
          const i = Math.max(0, Math.min(H - 1, (c.i|0)))
          const j = Math.max(0, Math.min(W - 1, (c.j|0)))
          const mag = Math.hypot(uGrid?.[i]?.[j] ?? 0, vGrid?.[i]?.[j] ?? 0)
          if ((!hasMask || unmasked[i][j]) && mag >= fieldWeakThreshold) valid.push([i, j])
        }
        if (valid.length > 0) {
          const key = `selected:${featureIndex ?? 'aggregate'}:${valid.map(([i, j]) => `${i}:${j}`).join('|')}`
          const picked = nextSpawnCell(valid, key) || valid[0]
          return spawnPointInCell(picked)
        }
      }
      if (featurePool.length) {
        const key = Number.isInteger(featureIndex) ? `feature:${featureIndex}` : 'global'
        const picked = nextSpawnCell(featurePool, key) || featurePool[0]
        return spawnPointInCell(picked)
      } else if (hasMask) {
        const fallback = []
        for (let i = 0; i < H; i++) {
          for (let j = 0; j < W; j++) {
            if (unmasked[i][j]) fallback.push([i, j])
          }
        }
        if (fallback.length) {
          return spawnPointInCell(fallback[Math.floor(Math.random() * fallback.length)])
        }
      }
      return {
        x: xmin + Math.random() * (xmax - xmin),
        y: ymin + Math.random() * (ymax - ymin),
      }
    }

    function resetParticle(p, x, y, nowSec) {
      p.x = x
      p.y = y
      p.ageSec = 0
      p.initX = x
      p.initY = y
      p.hist = [{ x, y, t: nowSec }]
    }

    // Particles with trail histories and lifetimes
    let simTimeSec = 0
    let simAccumulatorSec = 0
    const particles = Array.from({ length: particleCount }, (_, particleIndex) => {
      const particleFeatureIndex = compareFeatureIndices.length > 0
        ? compareFeatureIndices[particleIndex % compareFeatureIndices.length]
        : null
      const { x, y } = randomSpawn(particleFeatureIndex)
      return { x, y, ageSec: 0, hist: [{ x, y, t: simTimeSec }], initX: x, initY: y, featureIndex: particleFeatureIndex }
    })

    // Frame counter for periodic behaviors
    let frameCounter = 0

    staticTrailsRef.current = buildStaticTrailsFromSeeds(staticTrailSeedsRef.current)

    function stepFixed() {
      for (const p of particles) {
        const particleFeatureIndex = Number.isInteger(p.featureIndex) ? p.featureIndex : null
        const advected = advectAlongFieldRK2(p.x, p.y, FIXED_ADVECTION_STEP_SEC, particleFeatureIndex)
        if (!advected.valid) {
          const { x: nx, y: ny } = randomSpawn(particleFeatureIndex)
          resetParticle(p, nx, ny, simTimeSec)
          continue
        }

        p.x = advected.x
        p.y = advected.y
        p.ageSec += FIXED_ADVECTION_STEP_SEC
        p.hist.unshift({ x: p.x, y: p.y, t: simTimeSec })
        while (p.hist.length > 2 && (simTimeSec - p.hist[p.hist.length - 1].t) > TAIL_DURATION_SEC) {
          p.hist.pop()
        }

        // Respawn if out of bounds, over-age, or in masked region
        const outOfBounds = p.x < xmin || p.x > xmax || p.y < ymin || p.y > ymax
        const overAge = p.ageSec > MAX_LIFETIME_SEC
        const inMasked = isMaskedAt(p.x, p.y, particleFeatureIndex)
        if (outOfBounds || overAge || inMasked) {
          const { x: nx, y: ny } = randomSpawn(particleFeatureIndex)
          resetParticle(p, nx, ny, simTimeSec)
        }
      }

      // Optional automatic respawn in bursts every N frames: respawn ~fraction of particles
      const frac = Math.max(0, Math.min(1, autoRespawnRateRef.current || 0))
      const EVERY_N_FRAMES = 15
      frameCounter += 1
      if (frac > 0 && (frameCounter % EVERY_N_FRAMES === 0)) {
        const count = Math.max(1, Math.floor(frac * particles.length))
        for (let k = 0; k < count; k++) {
          const idx = Math.floor(Math.random() * particles.length)
          const p = particles[idx]
          const particleFeatureIndex = Number.isInteger(p.featureIndex) ? p.featureIndex : null
          const { x: nx, y: ny } = randomSpawn(particleFeatureIndex)
          resetParticle(p, nx, ny, simTimeSec)
        }
      }
    }

    let last = performance.now()
    function draw() {
      if (!runningRef.current) return
      const now = performance.now()
      const frameDt = Math.min((now - last) / 1000, 0.05)
      last = now
      if (showParticlesRef.current) {
        simAccumulatorSec += frameDt
        while (simAccumulatorSec >= FIXED_ADVECTION_STEP_SEC) {
          simAccumulatorSec -= FIXED_ADVECTION_STEP_SEC
          simTimeSec += FIXED_ADVECTION_STEP_SEC
          stepFixed()
        }
      }
      ctx.clearRect(0, 0, Wpx, Hpx)

      // If aggregated cell gradients are shown, also color each cell by its dominant feature
      // Also apply when showing particle init overlays
      if (!isOverviewMode && (showCellAggregatedGradientsRef.current || showParticleInitsRef.current)) {
        const cellWpx = Wpx / W
        const cellHpx = Hpx / H
        for (let i = 0; i < H; i++) {
          for (let j = 0; j < W; j++) {
            if (hasMask && !unmasked[i][j]) continue
            let fid = -1
            if (Array.isArray(featureIndices) && featureIndices.length > 1) {
              // Dominant among the selected features only
              let bestIdx = -1
              let bestMag2 = -1
              for (const fi of featureIndices) {
                const u = (uAll[fi]?.[i]?.[j] ?? 0)
                const v = (vAll[fi]?.[i]?.[j] ?? 0)
                const mag2 = u*u + v*v
                if (mag2 > bestMag2) { bestMag2 = mag2; bestIdx = fi }
              }
              fid = bestIdx
            } else if (Array.isArray(featureIndices) && featureIndices.length === 1) {
              fid = featureIndices[0]
            } else if (dominant && dominant[i] && typeof dominant[i][j] === 'number') {
              fid = dominant[i][j]
            }
            if (typeof fid !== 'number' || fid < 0) continue
            const rgb = colorForCell(i, j)
            const sx0 = j * cellWpx
            const sy1 = Hpx - (i + 1) * cellHpx
            ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.28)`
            ctx.fillRect(sx0, sy1, cellWpx, cellHpx)
          }
        }
      }

      if (showMaskOverlayRef.current && hasMask) {
        const cellWpx = Wpx / W
        const cellHpx = Hpx / H
        ctx.lineWidth = 0.8
        for (let i = 0; i < H; i++) {
          for (let j = 0; j < W; j++) {
            if (unmasked[i][j]) continue
            const sx0 = j * cellWpx
            const sy1 = Hpx - (i + 1) * cellHpx
            ctx.fillStyle = 'rgba(15,23,42,0.28)'
            ctx.fillRect(sx0, sy1, cellWpx, cellHpx)
            ctx.strokeStyle = 'rgba(239,68,68,0.50)'
            ctx.strokeRect(sx0 + 0.5, sy1 + 0.5, Math.max(0, cellWpx - 1), Math.max(0, cellHpx - 1))
          }
        }
      }

      // Draw grid lines (cell boundaries)
      if (showGridRef.current) {
        ctx.strokeStyle = 'rgba(180,180,180,0.35)'
        ctx.lineWidth = 0.5
        // vertical lines at x boundaries
        for (let k = 0; k <= W; k++) {
          const xk = xmin + (k / W) * (xmax - xmin)
          const v = viewRef.current
          const sx = (xk - v.xmin) / (v.xmax - v.xmin) * Wpx
          ctx.beginPath()
          ctx.moveTo(sx, 0)
          ctx.lineTo(sx, Hpx)
          ctx.stroke()
        }
        // horizontal lines at y boundaries
        for (let k = 0; k <= H; k++) {
          const yk = ymin + (k / H) * (ymax - ymin)
          const v = viewRef.current
          const sy = Hpx - (yk - v.ymin) / (v.ymax - v.ymin) * Hpx
          ctx.beginPath()
          ctx.moveTo(0, sy)
          ctx.lineTo(Wpx, sy)
          ctx.stroke()
        }
      }

      // Highlight selected grid cells (semi-transparent overlay)
      if (selectedRef.current && selectedRef.current.length > 0) {
        ctx.lineWidth = 1.2
        ctx.strokeStyle = 'rgba(37,99,235,0.7)'
        ctx.fillStyle = 'rgba(37,99,235,0.10)'
        for (const cell of selectedRef.current) {
          const i = Math.max(0, Math.min(H - 1, cell.i|0))
          const j = Math.max(0, Math.min(W - 1, cell.j|0))
          const x0 = xmin + (j / W) * (xmax - xmin)
          const x1 = xmin + ((j + 1) / W) * (xmax - xmin)
          const y0 = ymin + (i / H) * (ymax - ymin)
          const y1 = ymin + ((i + 1) / H) * (ymax - ymin)
          const sx0 = (x0 - xmin) / (xmax - xmin) * Wpx
          const sx1 = (x1 - xmin) / (xmax - xmin) * Wpx
          const sy0 = Hpx - (y0 - ymin) / (ymax - ymin) * Hpx
          const sy1 = Hpx - (y1 - ymin) / (ymax - ymin) * Hpx
          const w = sx1 - sx0
          const h = sy0 - sy1
          ctx.beginPath()
          ctx.rect(sx0, sy1, w, h)
          ctx.fill()
          ctx.stroke()
        }
      }

      if (hoverCellRef.current && interactionMode === 'brush') {
        const { i: hi, j: hj } = hoverCellRef.current
        const x0 = xmin + (hj / W) * (xmax - xmin)
        const x1 = xmin + ((hj + 1) / W) * (xmax - xmin)
        const y0 = ymin + (hi / H) * (ymax - ymin)
        const y1 = ymin + ((hi + 1) / H) * (ymax - ymin)
        const sx0 = (x0 - xmin) / (xmax - xmin) * Wpx
        const sx1 = (x1 - xmin) / (xmax - xmin) * Wpx
        const sy0 = Hpx - (y0 - ymin) / (ymax - ymin) * Hpx
        const sy1 = Hpx - (y1 - ymin) / (ymax - ymin) * Hpx
        ctx.fillStyle = 'rgba(37,99,235,0.22)'
        ctx.strokeStyle = 'rgba(37,99,235,0.95)'
        ctx.lineWidth = 1.5
        ctx.setLineDash([])
        ctx.fillRect(sx0, sy1, sx1 - sx0, sy0 - sy1)
        ctx.strokeRect(sx0, sy1, sx1 - sx0, sy0 - sy1)
      }

      if (hoverPointRef.current && !showParticleInitsRef.current) {
        const { sx, sy, label } = hoverPointRef.current
        const text = String(label)
        ctx.fillStyle = 'rgba(255,255,255,0.98)'
        ctx.strokeStyle = '#111827'
        ctx.lineWidth = 1.2
        ctx.beginPath()
        ctx.arc(sx, sy, Math.max(6, Number(pointSize) + 2.8), 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()

        ctx.font = '12px ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif'
        const paddingX = 8
        const boxH = 24
        const textW = Math.ceil(ctx.measureText(text).width)
        const boxW = textW + paddingX * 2
        let boxX = sx - boxW / 2
        let boxY = sy - boxH - 14
        if (boxX < 6) boxX = 6
        if (boxX + boxW > Wpx - 6) boxX = Wpx - boxW - 6
        if (boxY < 6) boxY = sy + 14
        ctx.fillStyle = 'rgba(255,255,255,0.96)'
        ctx.strokeStyle = 'rgba(17,24,39,0.92)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.roundRect(boxX, boxY, boxW, boxH, 8)
        ctx.fill()
        ctx.stroke()
        ctx.fillStyle = '#111827'
        ctx.textBaseline = 'middle'
        ctx.fillText(text, boxX + paddingX, boxY + boxH / 2)
      }

      // Optional: draw base points with shapes per label (like Python)
      // Skip drawing data points when showing particle init overlays
      if (!showParticleInitsRef.current) {
        // Shape cycle mirrors matplotlib-style markers used in the original
        const SHAPES = ['o','s','^', 'x', 'D','v','<','>','p','*','h','H','+']
        const shapeSize = Math.max(2, Number(pointSize) || 4.4)
        const pointHaloRadius = shapeSize + 1.1
        ctx.fillStyle = '#d9d9d9'
        // Dark gray/near-black borders for all points
        ctx.strokeStyle = '#222222'
        ctx.lineWidth = 0.8

      function drawRegularPolygon(cx, cy, radius, sides, rotation = 0) {
        if (sides < 3) return
        ctx.beginPath()
        for (let i = 0; i < sides; i++) {
          const a = rotation + (i * 2 * Math.PI) / sides
          const x = cx + radius * Math.cos(a)
          const y = cy + radius * Math.sin(a)
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
        }
        ctx.closePath()
        ctx.fill()
        ctx.stroke()
      }
      function drawDiamond(cx, cy, r) {
        ctx.beginPath()
        ctx.moveTo(cx, cy - r * 1.2)
        ctx.lineTo(cx - r * 1.2, cy)
        ctx.lineTo(cx, cy + r * 1.2)
        ctx.lineTo(cx + r * 1.2, cy)
        ctx.closePath()
        ctx.fill()
        ctx.stroke()
      }
      function drawTriangle(cx, cy, r, orientation) {
        ctx.beginPath()
        if (orientation === 'up') {
          ctx.moveTo(cx, cy - r * 1.3)
          ctx.lineTo(cx - r * 1.1, cy + r * 0.9)
          ctx.lineTo(cx + r * 1.1, cy + r * 0.9)
        } else if (orientation === 'down') {
          ctx.moveTo(cx, cy + r * 1.3)
          ctx.lineTo(cx - r * 1.1, cy - r * 0.9)
          ctx.lineTo(cx + r * 1.1, cy - r * 0.9)
        } else if (orientation === 'left') {
          ctx.moveTo(cx - r * 1.3, cy)
          ctx.lineTo(cx + r * 0.9, cy - r * 1.1)
          ctx.lineTo(cx + r * 0.9, cy + r * 1.1)
        } else {
          // right
          ctx.moveTo(cx + r * 1.3, cy)
          ctx.lineTo(cx - r * 0.9, cy - r * 1.1)
          ctx.lineTo(cx - r * 0.9, cy + r * 1.1)
        }
        ctx.closePath()
        ctx.fill()
        ctx.stroke()
      }
      function drawStar(cx, cy, r, points = 5) {
        const inner = r * 0.5
        ctx.beginPath()
        for (let i = 0; i < points * 2; i++) {
          const angle = (i * Math.PI) / points - Math.PI / 2
          const rad = (i % 2 === 0) ? r : inner
          const x = cx + Math.cos(angle) * rad
          const y = cy + Math.sin(angle) * rad
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
        }
        ctx.closePath(); ctx.fill(); ctx.stroke()
      }
      function drawPlus(cx, cy, r) {
        ctx.beginPath()
        ctx.moveTo(cx - r, cy)
        ctx.lineTo(cx + r, cy)
        ctx.moveTo(cx, cy - r)
        ctx.lineTo(cx, cy + r)
        ctx.stroke()
      }
      function drawCross(cx, cy, r) {
        ctx.beginPath()
        ctx.moveTo(cx - r, cy - r)
        ctx.lineTo(cx + r, cy + r)
        ctx.moveTo(cx + r, cy - r)
        ctx.lineTo(cx - r, cy + r)
        ctx.stroke()
      }
      function drawMarker(cx, cy, shape, r) {
        switch (shape) {
          case 'o':
            ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); break
          case 's':
            ctx.beginPath(); ctx.rect(cx - r, cy - r, 2 * r, 2 * r); ctx.fill(); ctx.stroke(); break
          case '^': drawTriangle(cx, cy, r, 'up'); break
          case 'v': drawTriangle(cx, cy, r, 'down'); break
          case '<': drawTriangle(cx, cy, r, 'left'); break
          case '>': drawTriangle(cx, cy, r, 'right'); break
          case 'D': drawDiamond(cx, cy, r); break
          case 'p': drawRegularPolygon(cx, cy, r * 1.1, 5); break
          case 'h': drawRegularPolygon(cx, cy, r * 1.1, 6); break
          case 'H': drawRegularPolygon(cx, cy, r * 1.15, 6, Math.PI / 6); break
          case '*': drawStar(cx, cy, r * 1.2); break
          case '+': drawPlus(cx, cy, r * 1.2); break
          case 'x': drawCross(cx, cy, r * 1.2); break
          default:
            ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill(); break
        }
      }

      function drawPointGlyph(cx, cy, fillColor) {
        ctx.fillStyle = 'rgba(255,255,255,0.92)'
        ctx.beginPath()
        ctx.arc(cx, cy, pointHaloRadius, 0, Math.PI * 2)
        ctx.fill()

        ctx.fillStyle = fillColor
        ctx.beginPath()
        ctx.arc(cx, cy, shapeSize, 0, Math.PI * 2)
        ctx.fill()

        ctx.strokeStyle = '#111827'
        ctx.lineWidth = 1.0
        ctx.beginPath()
        ctx.arc(cx, cy, shapeSize, 0, Math.PI * 2)
        ctx.stroke()
      }

      if (!showParticleInitsRef.current) {
        // Resolve point fill color in priority order:
        //   1. pointColorStats (grayscale-by-feature, explicit user selection) → overrides label
        //   2. labelColorMap   (category color from label palette)
        //   3. default gray
        const hasLabels = Array.isArray(point_labels) && point_labels.length === positions.length
        const activeLabelMap = labelColorMap && typeof labelColorMap === 'object' ? labelColorMap : null

        for (let idx = 0; idx < positions.length; idx++) {
          const [x, y] = positions[idx]
          const [sx, sy] = worldToScreen(x, y)
          let pointFill = '#d9d9d9'

          if (pointColorStats && Array.isArray(feature_values)) {
            // Grayscale-by-feature mode
            const col = pointColorStats.idx
            const row = feature_values[idx]
            const v = (row && col >= 0 && col < row.length) ? Number(row[col]) : NaN
            let t = 0.5
            if (Number.isFinite(v) && pointColorStats.max > pointColorStats.min) {
              t = Math.max(0, Math.min(1, (v - pointColorStats.min) / (pointColorStats.max - pointColorStats.min)))
            }
            const g = Math.round(255 * (1 - t))
            pointFill = `rgb(${g},${g},${g})`
          } else if (hasLabels && activeLabelMap) {
            // Label color mode — one consistent circle per label category
            const labelKey = String(point_labels[idx])
            pointFill = activeLabelMap[labelKey] || '#d9d9d9'
          }

          drawPointGlyph(sx, sy, pointFill)
        }
      }

      // Highlight selected data points by filling them in red and enlarging
      const selPts = selectedPointIndicesRef.current
      if (Array.isArray(selPts) && selPts.length > 0 && positions && positions.length) {
        for (const pi of selPts) {
          if (typeof pi !== 'number' || pi < 0 || pi >= positions.length) continue
          const [wx, wy] = positions[pi]
          const [sx, sy] = worldToScreen(wx, wy)
          const r = (typeof shapeSize === 'number' ? shapeSize : 3.2) + 3.0
          ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2)
          ctx.fillStyle = '#dc2626'
          ctx.fill()
          ctx.strokeStyle = '#000000'
          ctx.lineWidth = 1.5
          ctx.stroke()
        }
      }
      }

      // Optional: aggregated gradients attached to each point (sum over selected features)
      if (showPointAggregatedGradientsRef.current && positions && uSum && vSum) {
        const p99 = magInfo?.p99 || 1.0
        const sxScale = Wpx / (xmax - xmin)
        const syScale = Hpx / (ymax - ymin)
        const headLen = Math.max(6, Math.min(12, Math.floor(Math.min(Wpx, Hpx) * 0.015)))
        const baseLen = Math.max(10, Math.min(40, Math.min(Wpx, Hpx) * 0.05))
        const phi = Math.PI / 7
        ctx.lineWidth = 1.0
        ctx.strokeStyle = '#111111'
        for (let pIdx = 0; pIdx < positions.length; pIdx++) {
          const [wx, wy] = positions[pIdx]
          const [sx, sy] = worldToScreen(wx, wy)
          const [gx, gy] = worldToGrid(wx, wy)
          const ux = bilinearSample(uSum, gx, gy)
          const vy = bilinearSample(vSum, gx, gy)
          const m = Math.hypot(ux, vy)
          if (m <= 1e-12) continue
          const aField = Math.max(0, Math.min(1, m / p99))
          const ddx = ux * sxScale
          const ddy = -vy * syScale
          const dnorm = Math.hypot(ddx, ddy)
          if (!Number.isFinite(dnorm) || dnorm <= 0) continue
          const dirx = ddx / dnorm
          const diry = ddy / dnorm
          const len = Math.max(0, Math.min(1, aField)) * baseLen
          const ex = sx + dirx * len
          const ey = sy + diry * len
          ctx.globalAlpha = 0.9
          ctx.beginPath()
          ctx.moveTo(sx, sy)
          ctx.lineTo(ex, ey)
          ctx.stroke()
          const ang = Math.atan2(ey - sy, ex - sx)
          const hx1 = ex - headLen * Math.cos(ang - phi)
          const hy1 = ey - headLen * Math.sin(ang - phi)
          const hx2 = ex - headLen * Math.cos(ang + phi)
          const hy2 = ey - headLen * Math.sin(ang + phi)
          ctx.beginPath()
          ctx.moveTo(ex, ey)
          ctx.lineTo(hx1, hy1)
          ctx.moveTo(ex, ey)
          ctx.lineTo(hx2, hy2)
          ctx.stroke()
          ctx.globalAlpha = 1.0
        }
      }

      // Optional: per-feature gradients attached to each point
      if (showPointGradientsRef.current && positions) {
        // Use manually checked features when provided; otherwise fall back to current selection indices
        const provided = gradientFeatureIndicesRef.current
        const featList = Array.isArray(provided) ? provided : indices
        if (Array.isArray(featList) && featList.length > 0) {
          const p99Sum = magInfo?.p99 || 1.0
          const sxScale = Wpx / (xmax - xmin)
          const syScale = Hpx / (ymax - ymin)
          const headLen = Math.max(6, Math.min(12, Math.floor(Math.min(Wpx, Hpx) * 0.015)))
          const phi = Math.PI / 7 // ~25.7 degrees
          ctx.lineWidth = 1.0
          for (let pIdx = 0; pIdx < positions.length; pIdx++) {
            const [wx, wy] = positions[pIdx]
            const [sx, sy] = worldToScreen(wx, wy)
            const [gx, gy] = worldToGrid(wx, wy)
            for (const fi of featList) {
              const u = bilinearSample(uAll[fi], gx, gy)
              const v = bilinearSample(vAll[fi], gx, gy)
              const m = Math.hypot(u, v)
              if (m <= 1e-12) continue
              const denom = (p99ByFeature && fi >= 0 && fi < p99ByFeature.length && p99ByFeature[fi] > 0)
                ? p99ByFeature[fi]
                : p99Sum
              const aField = Math.max(0, Math.min(1, m / denom))
              // Map world vector to screen direction (account for y flip and axis scales)
              const ddx = u * sxScale
              const ddy = -v * syScale
              const dnorm = Math.hypot(ddx, ddy)
              if (!Number.isFinite(dnorm) || dnorm <= 0) continue
              const dirx = ddx / dnorm
              const diry = ddy / dnorm
              const baseLen = Math.max(10, Math.min(40, Math.min(Wpx, Hpx) * 0.05))
              const len = Math.max(0, Math.min(1, aField)) * baseLen
              const ex = sx + dirx * len
              const ey = sy + diry * len
              // Stroke color with field-strength alpha
              const col = featureHex(fi)
              ctx.strokeStyle = col
              ctx.globalAlpha = 0.9
              // Main shaft
              ctx.beginPath()
              ctx.moveTo(sx, sy)
              ctx.lineTo(ex, ey)
              ctx.stroke()
              // Arrow head
              const ang = Math.atan2(ey - sy, ex - sx)
              const hx1 = ex - headLen * Math.cos(ang - phi)
              const hy1 = ey - headLen * Math.sin(ang - phi)
              const hx2 = ex - headLen * Math.cos(ang + phi)
              const hy2 = ey - headLen * Math.sin(ang + phi)
              ctx.beginPath()
              ctx.moveTo(ex, ey)
              ctx.lineTo(hx1, hy1)
              ctx.moveTo(ex, ey)
              ctx.lineTo(hx2, hy2)
              ctx.stroke()
              ctx.globalAlpha = 1.0
            }
          }
        }
      }

      // Optional: per-feature gradients at each grid cell center
      if (showCellGradientsRef.current) {
        const provided = gradientFeatureIndicesRef.current
        const featList = Array.isArray(provided) ? provided : indices
        if (Array.isArray(featList) && featList.length > 0) {
          const p99Sum = magInfo?.p99 || 1.0
          const sxScale = Wpx / (xmax - xmin)
          const syScale = Hpx / (ymax - ymin)
          const headLen = Math.max(5, Math.min(10, Math.floor(Math.min(Wpx, Hpx) * 0.012)))
          const phi = Math.PI / 7
          const dx = (xmax - xmin) / W
          const dy = (ymax - ymin) / H
          const baseLen = Math.max(6, Math.min(24, Math.min(Wpx, Hpx) * 0.035))
          ctx.lineWidth = 0.9
          for (let i = 0; i < H; i++) {
            for (let j = 0; j < W; j++) {
              if (hasMask && !unmasked[i][j]) continue
              const wx = xmin + (j + 0.5) * dx
              const wy = ymin + (i + 0.5) * dy
              const [sx, sy] = worldToScreen(wx, wy)
              for (const fi of featList) {
                const u = (uAll[fi]?.[i]?.[j] ?? 0)
                const v = (vAll[fi]?.[i]?.[j] ?? 0)
                const m = Math.hypot(u, v)
                if (m <= 1e-12) continue
                const denom = (p99ByFeature && fi >= 0 && fi < p99ByFeature.length && p99ByFeature[fi] > 0)
                  ? p99ByFeature[fi]
                  : p99Sum
                const aField = Math.max(0, Math.min(1, m / denom))
                const ddx = u * sxScale
                const ddy = -v * syScale
                const dnorm = Math.hypot(ddx, ddy)
                if (!Number.isFinite(dnorm) || dnorm <= 0) continue
                const dirx = ddx / dnorm
                const diry = ddy / dnorm
                const len = aField * baseLen
                const ex = sx + dirx * len
                const ey = sy + diry * len
                const col = featureHex(fi)
                ctx.strokeStyle = col
                ctx.globalAlpha = 0.9
                ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke()
                const ang = Math.atan2(ey - sy, ex - sx)
                const hx1 = ex - headLen * Math.cos(ang - phi)
                const hy1 = ey - headLen * Math.sin(ang - phi)
                const hx2 = ex - headLen * Math.cos(ang + phi)
                const hy2 = ey - headLen * Math.sin(ang + phi)
                ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(hx1, hy1); ctx.moveTo(ex, ey); ctx.lineTo(hx2, hy2); ctx.stroke()
                ctx.globalAlpha = 1.0
              }
            }
          }
        }
      }

      // Optional: aggregated (summed) gradients at each grid cell center, colored black
      if (showCellAggregatedGradientsRef.current) {
        const p99 = magInfo?.p99 || 1.0
        const sxScale = Wpx / (xmax - xmin)
        const syScale = Hpx / (ymax - ymin)
        const headLen = Math.max(5, Math.min(10, Math.floor(Math.min(Wpx, Hpx) * 0.012)))
        const phi = Math.PI / 7
        const dx = (xmax - xmin) / W
        const dy = (ymax - ymin) / H
        const baseLen = Math.max(8, Math.min(28, Math.min(Wpx, Hpx) * 0.045))
        ctx.lineWidth = 1.2
        ctx.strokeStyle = '#000000'
        for (let i = 0; i < H; i++) {
          for (let j = 0; j < W; j++) {
            if (hasMask && !unmasked[i][j]) continue
            const ux = (uSum?.[i]?.[j] ?? 0)
            const vy = (vSum?.[i]?.[j] ?? 0)
            const m = Math.hypot(ux, vy)
            if (m <= 1e-12) continue
            const aField = Math.max(0, Math.min(1, m / p99))
            const wx = xmin + (j + 0.5) * dx
            const wy = ymin + (i + 0.5) * dy
            const [sx, sy] = worldToScreen(wx, wy)
            const ddx = ux * sxScale
            const ddy = -vy * syScale
            const dnorm = Math.hypot(ddx, ddy)
            if (!Number.isFinite(dnorm) || dnorm <= 0) continue
            const dirx = ddx / dnorm
            const diry = ddy / dnorm
            const len = aField * baseLen
            const ex = sx + dirx * len
            const ey = sy + diry * len
            ctx.globalAlpha = 1.0
            ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke()
            const ang = Math.atan2(ey - sy, ex - sx)
            const hx1 = ex - headLen * Math.cos(ang - phi)
            const hy1 = ey - headLen * Math.sin(ang - phi)
            const hx2 = ex - headLen * Math.cos(ang + phi)
            const hy2 = ey - headLen * Math.sin(ang + phi)
            ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(hx1, hy1); ctx.moveTo(ex, ey); ctx.lineTo(hx2, hy2); ctx.stroke()
          }
        }
      }

      // Draw trails as fading line segments (colored by the active feature mode)
      ctx.lineWidth = Math.max(0.5, Number(trailLineWidth) || 1.2)
      ctx.lineCap = 'round'
      // If exactly one feature is manually selected, force its color globally
      const singleColorOverride = (Array.isArray(featureIndices) && featureIndices.length === 1)
        ? (function () {
            const idx = featureIndices[0]
            const c = featureHex(idx)
            return hexToRgb(c)
          })()
        : null

      if (showParticlesRef.current) {
        for (const p of particles) {
          // reset cached head style each frame
          p._headAlpha = undefined
          p._headRgb = undefined
          if (!Array.isArray(p.hist) || p.hist.length < 2) continue
          for (let t = p.hist.length - 2; t >= 0; t--) {
          // Fade from tail (low alpha) to head (high alpha)
          const head = p.hist[t]
          const tail = p.hist[t + 1]
          const relHead = Math.max(0, Math.min(1, 1 - ((simTimeSec - head.t) / TAIL_DURATION_SEC)))
          const aTail = TRAIL_TAIL_MIN + (1 - TRAIL_TAIL_MIN) * Math.pow(relHead, TRAIL_TAIL_EXP)
          const x1 = tail.x
          const y1 = tail.y
          const x0 = head.x
          const y0 = head.y
          const particleFeatureIndex = Number.isInteger(p.featureIndex) ? p.featureIndex : null
          const aField = sampleFieldMagnitudeAt(x0, y0, particleFeatureIndex)
          const [gx0, gy0] = worldToGrid(x0, y0)

          // Determine segment color by family
          let rgb = hexToRgb(neutralColor)
          if (isOverviewMode) {
            rgb = hexToRgb(neutralColor)
          } else if (particleFeatureIndex !== null) {
            rgb = hexToRgb(featureHex(particleFeatureIndex))
          } else if (singleColorOverride) {
            rgb = singleColorOverride
          } else {
            const mi = Math.max(0, Math.min(H - 1, Math.round(gy0)))
            const mj = Math.max(0, Math.min(W - 1, Math.round(gx0)))
            if (Array.isArray(featureIndices) && featureIndices.length > 1) {
              // Dominant among the selected features only
              let bestIdx = -1
              let bestMag2 = -1
              for (const fi of featureIndices) {
                const u = (uAll[fi]?.[mi]?.[mj] ?? 0)
                const v = (vAll[fi]?.[mi]?.[mj] ?? 0)
                const mag2 = u*u + v*v
                if (mag2 > bestMag2) { bestMag2 = mag2; bestIdx = fi }
              }
              if (bestIdx >= 0) rgb = hexToRgb(featureHex(bestIdx))
            } else if (dominant) {
              // Fallback to backend-provided dominant feature
              const fid = dominant[mi]?.[mj]
              if (typeof fid === 'number' && fid >= 0) {
                rgb = hexToRgb(featureHex(fid))
              }
            }
          }

          // Respect mask: drop segments fully in masked cells
          if (hasMask) {
            const mi = Math.max(0, Math.min(H - 1, Math.round(gy0)))
            const mj = Math.max(0, Math.min(W - 1, Math.round(gx0)))
            if (!unmasked[mi][mj]) continue
          }

          const alpha = 1.0 * aTail * aField
          // Cache head segment opacity and color for arrowhead rendering
          if (t === 0) { p._headAlpha = alpha; p._headRgb = rgb }
          if (alpha <= 0.01) continue
          const [sx0, sy0] = worldToScreen(x0, y0)
          const [sx1, sy1] = worldToScreen(x1, y1)
          ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha.toFixed(3)})`
          ctx.beginPath()
          ctx.moveTo(sx1, sy1)
          ctx.lineTo(sx0, sy0)
          ctx.stroke()
          }
        }
        // Optional: arrowhead at each particle head
        if (showParticleArrowheadsRef.current) {
          // Slightly larger arrowheads for better visibility
          const headLen = Math.max(8, Math.min(16, Math.floor(Math.min(Wpx, Hpx) * 0.02)))
          const phi = Math.PI / 7
          for (const p of particles) {
            // Use cached head alpha/color to exactly match the head segment's opacity and color
            const alpha = (typeof p._headAlpha === 'number') ? p._headAlpha : 0
            const rgb = Array.isArray(p._headRgb) ? p._headRgb : [20, 20, 20]
            if (!(alpha > 0.01) || !Array.isArray(p.hist) || p.hist.length < 2) continue
            const x0 = p.hist[0].x, y0 = p.hist[0].y
            const x1 = p.hist[1].x, y1 = p.hist[1].y
            // Screen coords + direction from last step
            const [sx0, sy0] = worldToScreen(x0, y0)
            const [sx1, sy1] = worldToScreen(x1, y1)
            const ang = Math.atan2(sy0 - sy1, sx0 - sx1)
            ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha.toFixed(3)})`
            const hx1 = sx0 - headLen * Math.cos(ang - phi)
            const hy1 = sy0 - headLen * Math.sin(ang - phi)
            const hx2 = sx0 - headLen * Math.cos(ang + phi)
            const hy2 = sy0 - headLen * Math.sin(ang + phi)
            ctx.beginPath()
            ctx.moveTo(sx0, sy0)
            ctx.lineTo(hx1, hy1)
            ctx.moveTo(sx0, sy0)
            ctx.lineTo(hx2, hy2)
            ctx.stroke()
          }
        }
      }

      const staticTrails = Array.isArray(staticTrailsRef.current) ? staticTrailsRef.current : []
      for (const staticTrail of staticTrails) {
        if (!staticTrail || !Array.isArray(staticTrail.points) || staticTrail.points.length === 0) continue
        const points = staticTrail.points
        const segments = Array.isArray(staticTrail.segments) ? staticTrail.segments : []
        const trailFeatureIndex = Number.isInteger(staticTrail.featureIndex) ? staticTrail.featureIndex : null
        if (points.length >= 2) {
          ctx.lineJoin = 'round'
          ctx.lineCap = 'round'
          ctx.strokeStyle = 'rgba(255,255,255,0.96)'
          ctx.lineWidth = Math.max(2.5, Number(trailLineWidth) + 2.0)
          ctx.beginPath()
          const [sxStart, syStart] = worldToScreen(points[0].x, points[0].y)
          ctx.moveTo(sxStart, syStart)
          for (let k = 1; k < points.length; k++) {
            const [sx, sy] = worldToScreen(points[k].x, points[k].y)
            ctx.lineTo(sx, sy)
          }
          ctx.stroke()

          const minStaticWidth = Math.max(1.8, Number(trailLineWidth) * 0.95)
          const maxStaticWidth = Math.max(minStaticWidth + 1.2, Number(trailLineWidth) + 2.0)
          const staticWidthExp = 0.7
          let lastVisibleSeg = null
          let lastSegWidth = minStaticWidth
          for (const seg of segments) {
            const rgb = Array.isArray(seg.rgb) ? seg.rgb : [20, 20, 20]
            const aField = sampleFieldMagnitudeAt(seg.x1, seg.y1, trailFeatureIndex)
            const segWidth = minStaticWidth + (maxStaticWidth - minStaticWidth) * Math.pow(aField, staticWidthExp)
            const [sx0, sy0] = worldToScreen(seg.x0, seg.y0)
            const [sx1, sy1] = worldToScreen(seg.x1, seg.y1)
            ctx.lineWidth = segWidth
            ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.98)`
            ctx.beginPath()
            ctx.moveTo(sx0, sy0)
            ctx.lineTo(sx1, sy1)
            ctx.stroke()
            lastVisibleSeg = seg
            lastSegWidth = segWidth
          }

          const lastSeg = lastVisibleSeg
          if (lastSeg) {
            const rgb = Array.isArray(lastSeg.rgb) ? lastSeg.rgb : [20, 20, 20]
            const [sx0, sy0] = worldToScreen(lastSeg.x0, lastSeg.y0)
            const [sx1, sy1] = worldToScreen(lastSeg.x1, lastSeg.y1)
            const ang = Math.atan2(sy1 - sy0, sx1 - sx0)
            const headLen = Math.max(8, Math.min(18, 6 + lastSegWidth * 2.4))
            const phi = Math.PI / 7
            ctx.lineWidth = Math.max(1.6, lastSegWidth * 0.95)
            ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.98)`
            ctx.beginPath()
            ctx.moveTo(sx1, sy1)
            ctx.lineTo(sx1 - headLen * Math.cos(ang - phi), sy1 - headLen * Math.sin(ang - phi))
            ctx.moveTo(sx1, sy1)
            ctx.lineTo(sx1 - headLen * Math.cos(ang + phi), sy1 - headLen * Math.sin(ang + phi))
            ctx.stroke()
          }
        }

        const start = staticTrail.start || points[0]
        if (start) {
          const cell = worldToCell(start.x, start.y)
          const rgb = trailFeatureIndex !== null
            ? hexToRgb(featureHex(trailFeatureIndex))
            : (cell ? colorForCell(cell.i, cell.j) : [20, 20, 20])
          const [sx, sy] = worldToScreen(start.x, start.y)
          ctx.fillStyle = 'rgba(255,255,255,0.98)'
          ctx.beginPath()
          ctx.arc(sx, sy, 5.2, 0, Math.PI * 2)
          ctx.fill()
          ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.98)`
          ctx.lineWidth = 2.0
          ctx.beginPath()
          ctx.arc(sx, sy, 4.2, 0, Math.PI * 2)
          ctx.stroke()
        }
      }

      // Draw particle initialization places and their driving vectors.
      if (showParticleInitsRef.current) {
        const sxScale = Wpx / (xmax - xmin)
        const syScale = Hpx / (ymax - ymin)
        const headLen = Math.max(5, Math.min(10, Math.floor(Math.min(Wpx, Hpx) * 0.012)))
        const phi = Math.PI / 7
        const baseLen = Math.max(8, Math.min(28, Math.min(Wpx, Hpx) * 0.045))
        ctx.lineWidth = 1.0
        for (const p of particles) {
          const x0 = p.initX
          const y0 = p.initY
          const [gx0, gy0] = worldToGrid(x0, y0)
          const particleFeatureIndex = Number.isInteger(p.featureIndex) ? p.featureIndex : null
          const { uGrid, vGrid, fieldP99 } = resolveFieldContext(particleFeatureIndex)
          // Respect mask (skip masked cells)
          if (hasMask) {
            const mi = Math.max(0, Math.min(H - 1, Math.round(gy0)))
            const mj = Math.max(0, Math.min(W - 1, Math.round(gx0)))
            if (!unmasked[mi][mj]) continue
          }
          const u0 = bilinearSample(uGrid, gx0, gy0)
          const v0 = bilinearSample(vGrid, gx0, gy0)
          const m = Math.hypot(u0, v0)
          const aField = Math.max(0, Math.min(1, m / Math.max(fieldP99, 1e-6)))
          const [sx0, sy0] = worldToScreen(x0, y0)
          // Choose color by dominant feature at init cell (respect current selection if provided)
          let colorHex = neutralColor
          const mi = Math.max(0, Math.min(H - 1, Math.round(gy0)))
          const mj = Math.max(0, Math.min(W - 1, Math.round(gx0)))
          if (particleFeatureIndex !== null) {
            colorHex = featureHex(particleFeatureIndex)
          } else if (!isOverviewMode && Array.isArray(featureIndices) && featureIndices.length > 1) {
            let bestIdx = -1, bestMag2 = -1
            for (const fi of featureIndices) {
              const uu = (uAll[fi]?.[mi]?.[mj] ?? 0), vv = (vAll[fi]?.[mi]?.[mj] ?? 0)
              const mag2 = uu*uu + vv*vv
              if (mag2 > bestMag2) { bestMag2 = mag2; bestIdx = fi }
            }
            if (bestIdx >= 0) colorHex = featureHex(bestIdx)
          } else if (!isOverviewMode && Array.isArray(featureIndices) && featureIndices.length === 1) {
            const idx = featureIndices[0]
            colorHex = featureHex(idx)
          } else if (!isOverviewMode && dominant) {
            const fid = dominant[mi]?.[mj]
            if (typeof fid === 'number' && fid >= 0) colorHex = featureHex(fid)
          }
          ctx.fillStyle = colorHex
          ctx.strokeStyle = colorHex
          // Dot at init
          ctx.beginPath(); ctx.arc(sx0, sy0, 2.2, 0, Math.PI * 2); ctx.fill()
          // Arrow from init along driving vector (account for aspect and y-flip)
          const ddx = u0 * sxScale
          const ddy = -v0 * syScale
          const dnorm = Math.hypot(ddx, ddy)
          if (!Number.isFinite(dnorm) || dnorm <= 0) continue
          const dirx = ddx / dnorm
          const diry = ddy / dnorm
          const len = aField * baseLen
          const ex = sx0 + dirx * len
          const ey = sy0 + diry * len
          ctx.beginPath(); ctx.moveTo(sx0, sy0); ctx.lineTo(ex, ey); ctx.stroke()
          const ang = Math.atan2(ey - sy0, ex - sx0)
          const hx1 = ex - headLen * Math.cos(ang - Math.PI / 7)
          const hy1 = ey - headLen * Math.sin(ang - Math.PI / 7)
          const hx2 = ex - headLen * Math.cos(ang + Math.PI / 7)
          const hy2 = ey - headLen * Math.sin(ang + Math.PI / 7)
          ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(hx1, hy1); ctx.moveTo(ex, ey); ctx.lineTo(hx2, hy2); ctx.stroke()
        }
      }
      rafRef.current = requestAnimationFrame(draw)
    }
    runningRef.current = true
    rafRef.current = requestAnimationFrame(draw)

    // Hover handler to report world coords
    function handleMove(e) {
      const rect = canvas.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      const v = viewRef.current
      const x = v.xmin + (cx / Wpx) * (v.xmax - v.xmin)
      const y = v.ymin + ((Hpx - cy) / Hpx) * (v.ymax - v.ymin)
      if (onHover) onHover({ x, y })
      if (interactionMode === 'brush') {
        const j = Math.max(0, Math.min(W - 1, Math.floor((x - xmin) / (xmax - xmin) * W)))
        const i = Math.max(0, Math.min(H - 1, Math.floor((y - ymin) / (ymax - ymin) * H)))
        hoverCellRef.current = { i, j }
      } else {
        hoverCellRef.current = null
      }
      if (!showParticleInitsRef.current && positions.length > 0) {
        const hoverRadius = Math.max(8, (Number(pointSize) || 4.4) + 4)
        const hoverRadius2 = hoverRadius * hoverRadius
        let bestIdx = -1
        let bestD2 = Infinity
        for (let pIdx = 0; pIdx < positions.length; pIdx++) {
          const [px, py] = positions[pIdx]
          const [sx, sy] = worldToScreen(px, py)
          const dx = sx - cx
          const dy = sy - cy
          const d2 = dx * dx + dy * dy
          if (d2 <= hoverRadius2 && d2 < bestD2) {
            bestD2 = d2
            bestIdx = pIdx
          }
        }
        if (bestIdx >= 0) {
          const [px, py] = positions[bestIdx]
          const [sx, sy] = worldToScreen(px, py)
          const label = Array.isArray(point_labels) && bestIdx < point_labels.length
            ? String(point_labels[bestIdx])
            : String(bestIdx)
          hoverPointRef.current = { idx: bestIdx, sx, sy, label }
          if (onHover) onHover({ x, y, pointIndex: bestIdx, pointLabel: label })
        } else {
          hoverPointRef.current = null
        }
      } else {
        hoverPointRef.current = null
      }
      // Brush selection while dragging
      if (selectPointsModeRef.current && typeof onBrushPoints === 'function') {
        // Activate point brushing after small movement threshold from mousedown
        if (pointPointerDownRef.current && !pointBrushingRef.current) {
          const dx0 = cx - pointDownPosRef.current.x
          const dy0 = cy - pointDownPosRef.current.y
          const d2 = dx0*dx0 + dy0*dy0
          if (d2 > 36) { // 6px threshold
            pointBrushingRef.current = true
          }
        }
        if (!pointBrushingRef.current) return
        const R = 14
        const R2 = R * R
        const indices = []
        for (let pIdx = 0; pIdx < positions.length; pIdx++) {
          const [px, py] = positions[pIdx]
          const [sx, sy] = worldToScreen(px, py)
          const dx = sx - cx, dy = sy - cy
          if (dx*dx + dy*dy <= R2) indices.push(pIdx)
        }
        if (indices.length) { try { onBrushPoints({ indices }) } catch {} }
      } else if (brushingRef.current && typeof brushCbRef.current === 'function') {
        const j = Math.max(0, Math.min(W - 1, Math.floor((x - xmin) / (xmax - xmin) * W)))
        const i = Math.max(0, Math.min(H - 1, Math.floor((y - ymin) / (ymax - ymin) * H)))
        const last = lastBrushRef.current
        if (i !== last.i || j !== last.j) {
          lastBrushRef.current = { i, j }
          try { brushCbRef.current({ i, j }) } catch {}
        }
      }
    }
    canvas.addEventListener('mousemove', handleMove)

    function handleLeave() {
      hoverCellRef.current = null
      hoverPointRef.current = null
      if (onHover) onHover(null)
    }
    canvas.addEventListener('mouseleave', handleLeave)

    // Wheel zoom: zoom around mouse position
    function handleWheel(e) {
      try { e.preventDefault() } catch {}
      const rect = canvas.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      const v = viewRef.current
      const wx = v.xmin + (cx / Wpx) * (v.xmax - v.xmin)
      const wy = v.ymin + ((Hpx - cy) / Hpx) * (v.ymax - v.ymin)
      const scale = Math.exp(-e.deltaY * 0.0015) // smooth zoom factor
      const newW = (v.xmax - v.xmin) / scale
      const newH = (v.ymax - v.ymin) / scale
      const minW = (xmax - xmin) / 50
      const minH = (ymax - ymin) / 50
      const clampedW = Math.max(minW, Math.min((xmax - xmin), newW))
      const clampedH = Math.max(minH, Math.min((ymax - ymin), newH))
      // Keep mouse world position fixed
      const tX = (wx - v.xmin) / (v.xmax - v.xmin)
      const tY = (wy - v.ymin) / (v.ymax - v.ymin)
      const nxmin = wx - tX * clampedW
      const nxmax = nxmin + clampedW
      const nymin = wy - tY * clampedH
      const nymax = nymin + clampedH
      // Clamp to global bbox
      const clamped = clampView(nxmin, nxmax, nymin, nymax)
      viewRef.current = {
        xmin: clamped.xmin,
        xmax: clamped.xmax,
        ymin: clamped.ymin,
        ymax: clamped.ymax,
      }
    }
    canvas.addEventListener('wheel', handleWheel, { passive: false })

    // Click handler to select grid cell
    function handleClick(e) {
      if (interactionMode === 'brush') return
      if (suppressClickRef.current) {
        suppressClickRef.current = false
        return
      }
      const rect = canvas.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      const clickRadius = Math.max(6, pointBrushRadiusPxRef.current || 14)
      const clickRadius2 = clickRadius * clickRadius
      let clickedPointIdx = -1
      let bestPointD2 = Infinity
      for (let pIdx = 0; pIdx < positions.length; pIdx++) {
        const [px, py] = positions[pIdx]
        const [sx, sy] = worldToScreen(px, py)
        const dx = sx - cx
        const dy = sy - cy
        const d2 = dx * dx + dy * dy
        if (d2 <= clickRadius2 && d2 < bestPointD2) {
          bestPointD2 = d2
          clickedPointIdx = pIdx
        }
      }
      if (clickedPointIdx >= 0) {
        const [px, py] = positions[clickedPointIdx]
        const builtTrails = setStaticTrailSeeds(
          [{ x: px, y: py, pointIndex: clickedPointIdx }],
          { collectDiagnostics: true }
        )
        const clickedTrail = builtTrails[0] || null
        const stopReason = clickedTrail?.stopReason || 'unknown'
        const stopStep = clickedTrail?.stopStep ?? 0
        const segmentCount = Array.isArray(clickedTrail?.segments) ? clickedTrail.segments.length : 0
        console.info('[Wind Map] Static trail stop', {
          pointIndex: clickedPointIdx,
          stopReason,
          stopStep,
          segmentCount,
        })
        logStaticTrailDiagnostics(clickedTrail)
      } else {
        setStaticTrailSeeds([])
      }
      if (typeof onSelectPoint === 'function') {
        try { onSelectPoint({ idx: clickedPointIdx }) } catch {}
        return
      }
    }
    canvas.addEventListener('click', handleClick)

    // Brush handlers
    function handleDown(e) {
      if (e.button !== 0) return
      const rect = canvas.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      pointPointerDownRef.current = true
      pointBrushingRef.current = false
      pointDownPosRef.current = { x: cx, y: cy }
      panPointerDownRef.current = false
      panningRef.current = false
      panStartViewRef.current = null
      if (selectPointsModeRef.current) {
        // Do not start grid brushing in point-select mode
        return
      }
      if (interactionMode === 'brush') {
        const v2 = viewRef.current
        const x = v2.xmin + (cx / Wpx) * (v2.xmax - v2.xmin)
        const y = v2.ymin + ((Hpx - cy) / Hpx) * (v2.ymax - v2.ymin)
        const j = Math.max(0, Math.min(W - 1, Math.floor((x - xmin) / (xmax - xmin) * W)))
        const i = Math.max(0, Math.min(H - 1, Math.floor((y - ymin) / (ymax - ymin) * H)))
        brushingRef.current = true
        lastBrushRef.current = { i, j }
        if (typeof brushCbRef.current === 'function') {
          try { brushCbRef.current({ cells: [{ i, j }], replace: !e.shiftKey }) } catch {}
        }
        canvas.style.cursor = 'crosshair'
        return
      }
      panPointerDownRef.current = true
      panStartPosRef.current = { x: cx, y: cy }
      panStartViewRef.current = { ...viewRef.current }
      canvas.style.cursor = 'grabbing'
    }
    function handleUp(e) {
      if (brushingRef.current && interactionMode === 'brush') {
        suppressClickRef.current = true
      }
      pointPointerDownRef.current = false
      pointBrushingRef.current = false
      panPointerDownRef.current = false
      panningRef.current = false
      panStartViewRef.current = null
      brushingRef.current = false
      lastBrushRef.current = { i: -1, j: -1 }
      canvas.style.cursor = interactionMode === 'brush' ? 'crosshair' : 'grab'
    }
    canvas.style.cursor = interactionMode === 'brush' ? 'crosshair' : 'grab'
    const PAN_THRESHOLD2 = 36
    const originalHandleMove = handleMove
    function handleMoveWithPan(e) {
      const rect = canvas.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      if (panPointerDownRef.current && panStartViewRef.current) {
        const dx = cx - panStartPosRef.current.x
        const dy = cy - panStartPosRef.current.y
        const d2 = dx * dx + dy * dy
        if (!panningRef.current && d2 > PAN_THRESHOLD2) {
          panningRef.current = true
          suppressClickRef.current = true
        }
        if (panningRef.current) {
          const startView = panStartViewRef.current
          const viewW = startView.xmax - startView.xmin
          const viewH = startView.ymax - startView.ymin
          const nextXmin = startView.xmin - (dx / Wpx) * viewW
          const nextXmax = startView.xmax - (dx / Wpx) * viewW
          const nextYmin = startView.ymin + (dy / Hpx) * viewH
          const nextYmax = startView.ymax + (dy / Hpx) * viewH
          const clamped = clampView(nextXmin, nextXmax, nextYmin, nextYmax)
          viewRef.current = {
            xmin: clamped.xmin,
            xmax: clamped.xmax,
            ymin: clamped.ymin,
            ymax: clamped.ymax,
          }
        }
      }
      originalHandleMove(e)
    }
    canvas.addEventListener('mousedown', handleDown)
    window.addEventListener('mouseup', handleUp)
    canvas.removeEventListener('mousemove', handleMove)
    canvas.addEventListener('mousemove', handleMoveWithPan)

    return () => {
      runningRef.current = false
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      canvas.removeEventListener('mousemove', handleMoveWithPan)
      canvas.removeEventListener('mouseleave', handleLeave)
      canvas.removeEventListener('wheel', handleWheel)
      canvas.removeEventListener('click', handleClick)
      canvas.removeEventListener('mousedown', handleDown)
      window.removeEventListener('mouseup', handleUp)
    }
  }, [
    bbox,
    interactionMode,
    mode,
    grid_res,
    uSum,
    vSum,
    p99ByFeature,
    positions,
    point_labels,
    colors,
    dominant,
    featureColorMap,
    labelColorMap,
    neutralColor,
    featureIndices,
    pointColorStats,
    particleCount,
    tailDurationSec,
    trailTailMin,
    trailTailExp,
    lifetimeTailMultiplier,
    speedScale,
    trailLineWidth,
    pointSize,
    showCellAggregatedGradients,
    showCellGradients,
    showPointGradients,
    showPointAggregatedGradients,
  ])

  const canvasWidth = explicitWidth ?? fixedSize ?? canvasSize
  const canvasHeight = explicitHeight ?? fixedSize ?? canvasSize
  const isResponsiveCanvas = explicitWidth === null && explicitHeight === null && fixedSize === null
  return (
    <div ref={containerRef} className="canvas-surface">
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        style={{
          width: isResponsiveCanvas ? '100%' : `${canvasWidth}px`,
          height: isResponsiveCanvas ? 'auto' : `${canvasHeight}px`,
          border: '1px solid #ddd',
          display: 'block',
        }}
      />
    </div>
  )
}
