import React, { useEffect, useRef, useMemo } from 'react'

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
  showPoints = true,
  showParticles = true,
  showPointGradients = false,
  showCellGradients = false,
  showCellAggregatedGradients = false,
  showParticleInits = false,
  uniformPointShape = false,
  showParticleArrowheads = false,
  trailLineWidth = 2.0,
  autoRespawnRate = 0.0,
  restrictSpawnToSelection = false,
  brushRadius = 1,
  gradientFeatureIndices = null,
  speedScale = 1.0,
  tailLength = 10,
  trailTailMin = 0.10,
  trailTailExp = 2.0,
  maxLifetime = 200,
  size = 600,
  width = null,
  height = null,
  selectedCells = [],
  featureIndices = null,
  pointColorFeatureIndex = null,
  colorCells = true,
  showLIC = false,
  useMask = true,
  pointGradientColor = null,
}) {
  const canvasRef = useRef(null)
  const rafRef = useRef(0)
  const runningRef = useRef(false)
  const brushingRef = useRef(false)
  const lastBrushRef = useRef({ i: -1, j: -1 })
  const showParticlesRef = useRef(!!showParticles)
  const showPointGradientsRef = useRef(!!showPointGradients)
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
  const licCanvasRef = useRef(null)
  // Point brushing state: only becomes true after movement threshold
  const pointPointerDownRef = useRef(false)
  const pointBrushingRef = useRef(false)
  const pointDownPosRef = useRef({ x: 0, y: 0 })
  const gradientFeatureIndicesRef = useRef(Array.isArray(gradientFeatureIndices) ? gradientFeatureIndices : [])
  const brushCbRef = useRef(onBrushCell)
  // Keep dynamic props in refs to avoid reinitializing particles on toggle
  const showGridRef = useRef(!!showGrid)
  const showPointsRef = useRef(!!showPoints)
  const selectedRef = useRef(selectedCells)
  // keep selection fresh for the draw loop without resetting particles
  useEffect(() => { selectedRef.current = selectedCells || [] }, [selectedCells])
  useEffect(() => { showGridRef.current = !!showGrid }, [showGrid])
  useEffect(() => { showPointsRef.current = !!showPoints }, [showPoints])
  useEffect(() => { brushCbRef.current = onBrushCell }, [onBrushCell])
  useEffect(() => { showParticlesRef.current = !!showParticles }, [showParticles])
  useEffect(() => { showPointGradientsRef.current = !!showPointGradients }, [showPointGradients])
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

  // Expose canvas element to parent for saving snapshots
  useEffect(() => {
    if (typeof onCanvasElement === 'function') onCanvasElement(canvasRef.current)
    return () => { if (typeof onCanvasElement === 'function') onCanvasElement(null) }
  }, [onCanvasElement])
  useEffect(() => { gradientFeatureIndicesRef.current = Array.isArray(gradientFeatureIndices) ? gradientFeatureIndices : [] }, [gradientFeatureIndices])

  const {
    bbox = [0, 1, 0, 1],
    grid_res = 600,
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

  // Single-feature selection for LIC (falls back to aggregated when none provided)
  const licFeatureIndex = useMemo(() => {
    if (Array.isArray(featureIndices) && featureIndices.length > 0) return featureIndices[0]
    if (selection && selection.featureIndex !== undefined) return selection.featureIndex
    if (selection && Array.isArray(selection.topKIndices) && selection.topKIndices.length > 0) return selection.topKIndices[0]
    return null
  }, [featureIndices, selection])

  const licField = useMemo(() => {
    const idx = licFeatureIndex
    const u = (typeof idx === 'number' && uAll?.[idx]) ? uAll[idx] : uSum
    const v = (typeof idx === 'number' && vAll?.[idx]) ? vAll[idx] : vSum
    return { u, v }
  }, [licFeatureIndex, uAll, vAll, uSum, vSum])

  // Normalized field for LIC tracing (unit vectors per cell)
  const licFieldNorm = useMemo(() => {
    const u = licField.u
    const v = licField.v
    if (!u || !v || !u.length || !v.length) return { u: null, v: null }
    const H = u.length
    const W = u[0]?.length || 0
    if (!H || !W) return { u: null, v: null }
    const uN = Array.from({ length: H }, () => new Float32Array(W))
    const vN = Array.from({ length: H }, () => new Float32Array(W))
    for (let i = 0; i < H; i++) {
      const ui = u[i]
      const vi = v[i]
      const uo = uN[i]
      const vo = vN[i]
      for (let j = 0; j < W; j++) {
        const uu = ui?.[j] ?? 0
        const vv = vi?.[j] ?? 0
        const m = Math.hypot(uu, vv)
        if (m > 1e-9) {
          uo[j] = uu / m
          vo[j] = vv / m
        } else {
          uo[j] = 0
          vo[j] = 0
        }
      }
    }
    return { u: uN, v: vN }
  }, [licField])

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

  // Build a LIC texture over a normalized single-feature vector field
  useEffect(() => {
    const uRaw = licField.u
    const vRaw = licField.v
    const uField = licFieldNorm.u
    const vField = licFieldNorm.v
    if (!showLIC || !uRaw || !vRaw || !uField || !vField) { licCanvasRef.current = null; return }
    const H = uRaw.length
    const W = uRaw[0]?.length || 0
    if (!H || !W) { licCanvasRef.current = null; return }
    const hasMask = useMask && Array.isArray(unmasked) && unmasked.length === H && unmasked[0].length === W

    // p99 magnitude from the raw (unnormalized) LIC field for alpha scaling
    let magScale = 1.0
    try {
      const mags = new Float32Array(H * W)
      let k = 0
      for (let i = 0; i < H; i++) {
        const ui = uRaw[i]
        const vi = vRaw[i]
        for (let j = 0; j < W; j++, k++) {
          mags[k] = Math.hypot(ui?.[j] ?? 0, vi?.[j] ?? 0)
        }
      }
      const arr = Array.from(mags)
      arr.sort((a, b) => a - b)
      magScale = arr.length ? (arr[Math.floor(arr.length * 0.99)] || 1.0) : 1.0
      if (!(magScale > 0)) magScale = 1.0
    } catch { magScale = 1.0 }

    const noise = new Float32Array(H * W)
    // Zero-mean noise helps the convolution keep visible contrast after averaging.
    for (let k = 0; k < noise.length; k++) noise[k] = (Math.random() * 2 - 1)
    const sampleNoise = (gx, gy) => {
      const j0 = Math.max(0, Math.min(W - 1, Math.floor(gx)))
      const i0 = Math.max(0, Math.min(H - 1, Math.floor(gy)))
      const j1 = Math.min(j0 + 1, W - 1), i1 = Math.min(i0 + 1, H - 1)
      const a = Math.min(Math.max(gx - j0, 0), 1)
      const b = Math.min(Math.max(gy - i0, 0), 1)
      const n00 = noise[i0 * W + j0]
      const n01 = noise[i0 * W + j1]
      const n10 = noise[i1 * W + j0]
      const n11 = noise[i1 * W + j1]
      const top = n00 * (1 - a) + n01 * a
      const bot = n10 * (1 - a) + n11 * a
      return top * (1 - b) + bot * b
    }

    // LIC: trace streamline in both directions for length 2L on normalized field.
    // Use a tapered (Hann) kernel to avoid the overly-blurry “box filter” look.
    const L = 10 // streamline half-length in grid units (smaller = crisper)
    const stepSize = 0.9 // step in grid units

    // Hann window on [0,1] with max at 0 and 0 at 1
    const hann01 = (t) => {
      const x = Math.max(0, Math.min(1, t))
      return 0.5 * (1 + Math.cos(Math.PI * x))
    }
    const data = new Uint8ClampedArray(W * H * 4)

    const traceStream = (gx0, gy0, dir) => {
      const samples = []
      let gx = gx0
      let gy = gy0
      let travelled = 0
      while (travelled < L) {
        // If we ever step out of bounds, stop before sampling.
        if (gx < 0 || gx > W - 1 || gy < 0 || gy > H - 1) break
        const u = bilinearSample(uField, gx, gy)
        const v = bilinearSample(vField, gx, gy)
        const m = Math.hypot(u, v)
        if (m <= 1e-9) break

        const nx = gx + (u / m) * stepSize * dir
        const ny = gy + (v / m) * stepSize * dir
        const segLen = Math.hypot(nx - gx, ny - gy)
        if (segLen <= 1e-6) break

        travelled += segLen
        gx = nx
        gy = ny

        // Record the new sample point and its distance from the seed
        // (used for tapered convolution kernel)
        samples.push({ gx, gy, segLen, dist: travelled })
      }
      return samples
    }

    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const idx = 4 * (i * W + j)
        if (hasMask && !unmasked[i][j]) { data[idx + 3] = 0; continue }

        const centerMag = Math.hypot(uRaw[i]?.[j] ?? 0, vRaw[i]?.[j] ?? 0)
        const gx0 = j + 0.5
        const gy0 = i + 0.5

        // Forward/backward streamlines with path-length weights
        const forward = traceStream(gx0, gy0, 1)
        const backward = traceStream(gx0, gy0, -1)
        // Convolve zero-mean noise along the streamline using a tapered (Hann) kernel.
        let acc = sampleNoise(gx0, gy0) * 1.0
        let wsum = 1.0
        for (const s of forward) {
          const k = hann01(s.dist / L)
          const w = s.segLen * k
          acc += sampleNoise(s.gx, s.gy) * w
          wsum += w
        }
        for (const s of backward) {
          const k = hann01(s.dist / L)
          const w = s.segLen * k
          acc += sampleNoise(s.gx, s.gy) * w
          wsum += w
        }
        const val0 = (wsum > 0) ? (acc / wsum) : 0.0 // roughly in [-1, 1]
        const val = 0.5 + 0.5 * val0

        // Contrast shaping: more aggressive stretch (because convolution is still an average)
        const centered = 0.5 + 2.4 * (val - 0.5)
        const clamped = Math.max(0, Math.min(1, centered))
        const boosted = Math.pow(clamped, 0.85)
        const c = Math.max(0, Math.min(255, Math.round(boosted * 255)))
        const alphaScale = Math.max(0, Math.min(1, centerMag / magScale))
        const a = Math.max(5, Math.round(255 * Math.pow(alphaScale, 0.85)))
        data[idx] = c; data[idx + 1] = c; data[idx + 2] = c; data[idx + 3] = a
      }
    }
    const off = document.createElement('canvas')
    off.width = W; off.height = H
    const ictx = off.getContext('2d')
    const img = new ImageData(data, W, H)
    ictx.putImageData(img, 0, 0)
    licCanvasRef.current = off
  }, [showLIC, licField, licFieldNorm, unmasked, useMask])

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
    const TAIL_LENGTH = Math.max(1, Math.floor(tailLength))
    const TRAIL_TAIL_MIN = Math.max(0, Math.min(1, trailTailMin))
    const TRAIL_TAIL_EXP = Math.max(0.5, trailTailExp)
    const MAX_LIFETIME = Math.max(1, Math.floor(maxLifetime))
    const SPEED_SCALE = speedScale

    // Optional mask helpers
    let hasMask = useMask && Array.isArray(unmasked) && unmasked.length === H && unmasked[0].length === W

    // Precompute list of unmasked cells for efficient respawn
    let unmaskedList = []
    if (hasMask) {
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          if (unmasked[i][j]) unmaskedList.push([i, j])
        }
      }
      if (unmaskedList.length === 0) hasMask = false
    }

    function randomSpawn() {
      // If restriction is enabled and we have selected cells, spawn within them (respecting mask)
      if (restrictSpawnToSelectionRef.current && Array.isArray(selectedRef.current) && selectedRef.current.length > 0) {
        const valid = []
        for (const c of selectedRef.current) {
          const i = Math.max(0, Math.min(H - 1, (c.i|0)))
          const j = Math.max(0, Math.min(W - 1, (c.j|0)))
          if (!hasMask || (unmasked[i][j])) valid.push([i, j])
        }
        if (valid.length > 0) {
          const [i, j] = valid[Math.floor(Math.random() * valid.length)]
          const dx = (xmax - xmin) / W
          const dy = (ymax - ymin) / H
          const x = xmin + j * dx + Math.random() * dx
          const y = ymin + i * dy + Math.random() * dy
          return { x, y }
        }
      }
      if (hasMask && unmaskedList.length) {
        const idx = Math.floor(Math.random() * unmaskedList.length)
        const [i, j] = unmaskedList[idx]
        const dx = (xmax - xmin) / W
        const dy = (ymax - ymin) / H
        const x = xmin + j * dx + Math.random() * dx
        const y = ymin + i * dy + Math.random() * dy
        return { x, y }
      } else {
        return {
          x: xmin + Math.random() * (xmax - xmin),
          y: ymin + Math.random() * (ymax - ymin),
        }
      }
    }

    // Particles with trail histories and lifetimes
    const particles = Array.from({ length: particleCount }, () => {
      const { x, y } = randomSpawn()
      const hist = Array.from({ length: TAIL_LENGTH + 1 }, () => ({ x, y }))
      return { x, y, age: 0, hist, initX: x, initY: y }
    })

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

    // Frame counter for periodic behaviors
    let frameCounter = 0

    function step(dt) {
      const MASK_THRESHOLD = 1e-6
      function isMaskedAt(x, y) {
        // Convert to grid indices
        const gx = (x - xmin) / (xmax - xmin) * (W - 1)
        const gy = (y - ymin) / (ymax - ymin) * (H - 1)
        const mi = Math.max(0, Math.min(H - 1, Math.round(gy)))
        const mj = Math.max(0, Math.min(W - 1, Math.round(gx)))
        // 1) Use explicit unmasked grid when available
        if (hasMask) {
          return !unmasked[mi][mj]
        }
        // 2) Fallback: use field magnitude threshold
        const u = bilinearSample(uSum, gx, gy)
        const v = bilinearSample(vSum, gx, gy)
        if (Math.hypot(u, v) <= MASK_THRESHOLD) return true
        // 3) Last resort: dominance grid -1 means masked
        if (dominant && dominant[mi] && typeof dominant[mi][mj] === 'number') {
          return dominant[mi][mj] === -1
        }
        return false
      }

      for (const p of particles) {
        const [gx, gy] = worldToGrid(p.x, p.y)
        let u = bilinearSample(uSum, gx, gy)
        let v = bilinearSample(vSum, gx, gy)

        // Scale speed directly by user-controlled factor
        u *= SPEED_SCALE
        v *= SPEED_SCALE

        // Integrate position
        p.x += u * dt
        p.y += v * dt
        p.age += 1

        // Shift history
        for (let t = TAIL_LENGTH; t >= 1; t--) {
          p.hist[t].x = p.hist[t - 1].x
          p.hist[t].y = p.hist[t - 1].y
        }
        p.hist[0].x = p.x
        p.hist[0].y = p.y

        // Respawn if out of bounds, over-age, or in masked region
        const outOfBounds = p.x < xmin || p.x > xmax || p.y < ymin || p.y > ymax
        const overAge = p.age > MAX_LIFETIME
        const inMasked = isMaskedAt(p.x, p.y)
        if (outOfBounds || overAge || inMasked) {
          const { x: nx, y: ny } = randomSpawn()
          p.x = nx
          p.y = ny
          p.age = 0
          p.initX = nx
          p.initY = ny
          for (let t = 0; t <= TAIL_LENGTH; t++) { p.hist[t].x = nx; p.hist[t].y = ny }
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
          const { x: nx, y: ny } = randomSpawn()
          p.x = nx
          p.y = ny
          p.age = 0
          p.initX = nx
          p.initY = ny
          for (let t = 0; t <= TAIL_LENGTH; t++) { p.hist[t].x = nx; p.hist[t].y = ny }
        }
      }
    }

    let last = performance.now()
    function draw() {
      if (!runningRef.current) return
      const now = performance.now()
      const dt = Math.min((now - last) / 1000, 0.05)
      last = now
      if (showParticlesRef.current) {
        step(dt)
      }
      ctx.clearRect(0, 0, Wpx, Hpx)

      if (showLIC && licCanvasRef.current) {
        ctx.save()
        ctx.globalAlpha = 0.9
        ctx.imageSmoothingEnabled = false
        // Draw LIC clipped to current view so it pans/zooms with the viewport
        const v = viewRef.current
        const srcX = Math.max(0, Math.min(licCanvasRef.current.width, (v.xmin - xmin) / (xmax - xmin) * licCanvasRef.current.width))
        const srcY = Math.max(0, Math.min(licCanvasRef.current.height, (v.ymin - ymin) / (ymax - ymin) * licCanvasRef.current.height))
        const srcW = Math.max(1, Math.min(licCanvasRef.current.width - srcX, (v.xmax - v.xmin) / (xmax - xmin) * licCanvasRef.current.width))
        const srcH = Math.max(1, Math.min(licCanvasRef.current.height - srcY, (v.ymax - v.ymin) / (ymax - ymin) * licCanvasRef.current.height))
        // Flip vertically via transform so world y-up aligns with screen y-down
        ctx.scale(1, -1)
        ctx.translate(0, -Hpx)
        ctx.drawImage(licCanvasRef.current, srcX, srcY, srcW, srcH, 0, 0, Wpx, Hpx)
        ctx.restore()
      }

      // If aggregated cell gradients are shown, also color each cell by its dominant feature
      // Also apply when showing particle init overlays
      if (colorCells && (showCellAggregatedGradientsRef.current || showParticleInitsRef.current) && colors && colors.length) {
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
            if (typeof fid !== 'number' || fid < 0 || fid >= colors.length) continue
            const rgb = hexToRgb(colors[fid])
            const sx0 = j * cellWpx
            const sy1 = Hpx - (i + 1) * cellHpx
            ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.28)`
            ctx.fillRect(sx0, sy1, cellWpx, cellHpx)
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

      // Optional: draw base points with shapes per label (like Python)
      // Skip drawing data points when showing particle init overlays
      if (showPointsRef.current && !showParticleInitsRef.current) {
        // Shape cycle mirrors matplotlib-style markers used in the original
        const SHAPES = ['o','s','^', 'x', 'D','v','<','>','p','*','h','H','+']
        // Bigger, light-gray markers for better visibility
        const shapeSize = 3.2
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

        if (Array.isArray(point_labels) && point_labels.length === positions.length) {
          // Build stable mapping value→index using sorted unique labels
          const unique = Array.from(new Set(point_labels.map((l) => String(l))))
          unique.sort()
          const map = new Map(unique.map((val, idx) => [val, idx]))
          for (let idx = 0; idx < positions.length; idx++) {
            const [x, y] = positions[idx]
            const [sx, sy] = worldToScreen(x, y)
            const li = map.get(String(point_labels[idx])) || 0
            const shape = uniformPointShapeRef.current ? 'o' : SHAPES[li % SHAPES.length]
            if (pointColorStats && Array.isArray(feature_values)) {
              const col = pointColorStats.idx
              const row = feature_values[idx]
              const v = (row && col >= 0 && col < row.length) ? Number(row[col]) : NaN
              let t = 0.5
              if (Number.isFinite(v) && pointColorStats.max > pointColorStats.min) {
                t = Math.max(0, Math.min(1, (v - pointColorStats.min) / (pointColorStats.max - pointColorStats.min)))
              }
              const g = Math.round(255 * (1 - t))
              ctx.fillStyle = `rgb(${g},${g},${g})`
            } else {
              ctx.fillStyle = '#d9d9d9'
            }
            drawMarker(sx, sy, shape, shapeSize)
          }
        } else {
          // Fallback: simple circles (larger, gray) with border
          for (let pIdx = 0; pIdx < positions.length; pIdx++) {
            const [x, y] = positions[pIdx]
            const [sx, sy] = worldToScreen(x, y)
            if (pointColorStats && Array.isArray(feature_values)) {
              const col = pointColorStats.idx
              const row = feature_values[pIdx]
              const v = (row && col >= 0 && col < row.length) ? Number(row[col]) : NaN
              let t = 0.5
              if (Number.isFinite(v) && pointColorStats.max > pointColorStats.min) {
                t = Math.max(0, Math.min(1, (v - pointColorStats.min) / (pointColorStats.max - pointColorStats.min)))
              }
              const g = Math.round(255 * (1 - t))
              ctx.fillStyle = `rgb(${g},${g},${g})`
            } else {
              ctx.fillStyle = '#d9d9d9'
            }
            ctx.beginPath(); ctx.arc(sx, sy, shapeSize, 0, Math.PI * 2); ctx.fill(); ctx.stroke()
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

      // Optional: per-feature gradients attached to each point
      if (showPointGradientsRef.current && positions) {
        // Use manually checked features when provided; otherwise fall back to current selection indices
        const provided = gradientFeatureIndicesRef.current
        const featList = (Array.isArray(provided) && provided.length > 0) ? provided : indices
        if (Array.isArray(featList) && featList.length > 0) {
          const p99Sum = magInfo?.p99 || 1.0
          const sxScale = Wpx / (xmax - xmin)
          const syScale = Hpx / (ymax - ymin)
          const headLenBase = Math.max(6, Math.min(12, Math.floor(Math.min(Wpx, Hpx) * 0.015)))
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
              const headLen = Math.max(4, Math.min(headLenBase, len * 0.35))
              const ex = sx + dirx * len
              const ey = sy + diry * len
              // Stroke color with field-strength alpha (override if provided)
              const col = (pointGradientColor && typeof pointGradientColor === 'string')
                ? pointGradientColor
                : (colors && fi >= 0 && fi < colors.length) ? colors[fi] : '#222222'
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
        const featList = (Array.isArray(provided) && provided.length > 0) ? provided : indices
        if (Array.isArray(featList) && featList.length > 0) {
          const p99Sum = magInfo?.p99 || 1.0
          const sxScale = Wpx / (xmax - xmin)
          const syScale = Hpx / (ymax - ymin)
          const headLenBase = Math.max(5, Math.min(10, Math.floor(Math.min(Wpx, Hpx) * 0.012)))
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
                const headLen = Math.max(4, Math.min(headLenBase, len * 0.35))
                const ex = sx + dirx * len
                const ey = sy + diry * len
                const col = (colors && fi >= 0 && fi < colors.length) ? colors[fi] : '#222222'
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
        const headLenBase = Math.max(5, Math.min(10, Math.floor(Math.min(Wpx, Hpx) * 0.012)))
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
            const headLen = Math.max(4, Math.min(headLenBase, len * 0.35))
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

      // Draw trails as fading line segments (colored by feature family)
      ctx.lineWidth = Math.max(0.5, Number(trailLineWidth) || 1.2)
      ctx.lineCap = 'round'
      const p99 = magInfo?.p99 || 1.0

      function hexToRgb(hex) {
        if (!hex || typeof hex !== 'string') return [20, 20, 20]
        const m = hex.match(/^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i)
        if (!m) return [20, 20, 20]
        return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)]
      }
      // If exactly one feature is manually selected, force its color globally
      const singleColorOverride = (Array.isArray(featureIndices) && featureIndices.length === 1)
        ? (function () {
            const idx = featureIndices[0]
            function hexToRgb(hex) {
              if (!hex || typeof hex !== 'string') return [20, 20, 20]
              const m = hex.match(/^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i)
              if (!m) return [20, 20, 20]
              return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)]
            }
            const c = (colors && idx >= 0 && idx < colors.length) ? colors[idx] : '#141414'
            return hexToRgb(c)
          })()
        : null

      if (showParticlesRef.current) {
        for (const p of particles) {
          // reset cached head style each frame
          p._headAlpha = undefined
          p._headRgb = undefined
          for (let t = TAIL_LENGTH - 1; t >= 0; t--) {
          // Fade from tail (low alpha) to head (high alpha)
          const relHead = (TAIL_LENGTH - t) / TAIL_LENGTH
          const aTail = TRAIL_TAIL_MIN + (1 - TRAIL_TAIL_MIN) * Math.pow(relHead, TRAIL_TAIL_EXP)
          const x1 = p.hist[t + 1].x
          const y1 = p.hist[t + 1].y
          const x0 = p.hist[t].x
          const y0 = p.hist[t].y
          // Field-strength based alpha at segment head
          const [gx0, gy0] = worldToGrid(x0, y0)
          const u0 = bilinearSample(uSum, gx0, gy0)
          const v0 = bilinearSample(vSum, gx0, gy0)
          const m = Math.hypot(u0, v0)
          const aField = Math.max(0, Math.min(1, m / p99))

          // Determine segment color by family
          let rgb = [20, 20, 20]
          if (singleColorOverride) {
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
              if (bestIdx >= 0) rgb = hexToRgb(colors[bestIdx] || '#141414')
            } else if (dominant && colors && colors.length) {
              // Fallback to backend-provided dominant feature
              const fid = dominant[mi]?.[mj]
              if (typeof fid === 'number' && fid >= 0 && fid < colors.length) {
                rgb = hexToRgb(colors[fid])
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
          const headLenBase = Math.max(8, Math.min(16, Math.floor(Math.min(Wpx, Hpx) * 0.02)))
          const phi = Math.PI / 7
          for (const p of particles) {
            // Use cached head alpha/color to exactly match the head segment's opacity and color
            const alpha = (typeof p._headAlpha === 'number') ? p._headAlpha : 0
            const rgb = Array.isArray(p._headRgb) ? p._headRgb : [20, 20, 20]
            if (!(alpha > 0.01)) continue
            const x0 = p.hist[0].x, y0 = p.hist[0].y
            const x1 = p.hist[1].x, y1 = p.hist[1].y
            // Screen coords + direction from last step
            const [sx0, sy0] = worldToScreen(x0, y0)
            const [sx1, sy1] = worldToScreen(x1, y1)
            const ang = Math.atan2(sy0 - sy1, sx0 - sx1)
            const segLen = Math.hypot(sx0 - sx1, sy0 - sy1)
            const headLen = Math.max(4, Math.min(headLenBase, segLen * 0.35))
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

      // Draw particle initialization places and their driving vectors (from summed field)
      if (showParticleInitsRef.current) {
        const p99 = magInfo?.p99 || 1.0
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
          // Respect mask (skip masked cells)
          if (hasMask) {
            const mi = Math.max(0, Math.min(H - 1, Math.round(gy0)))
            const mj = Math.max(0, Math.min(W - 1, Math.round(gx0)))
            if (!unmasked[mi][mj]) continue
          }
          const u0 = bilinearSample(uSum, gx0, gy0)
          const v0 = bilinearSample(vSum, gx0, gy0)
          const m = Math.hypot(u0, v0)
          const aField = Math.max(0, Math.min(1, m / p99))
          const [sx0, sy0] = worldToScreen(x0, y0)
          // Choose color by dominant feature at init cell (respect current selection if provided)
          let colorHex = '#000'
          const mi = Math.max(0, Math.min(H - 1, Math.round(gy0)))
          const mj = Math.max(0, Math.min(W - 1, Math.round(gx0)))
          if (Array.isArray(featureIndices) && featureIndices.length > 1) {
            let bestIdx = -1, bestMag2 = -1
            for (const fi of featureIndices) {
              const uu = (uAll[fi]?.[mi]?.[mj] ?? 0), vv = (vAll[fi]?.[mi]?.[mj] ?? 0)
              const mag2 = uu*uu + vv*vv
              if (mag2 > bestMag2) { bestMag2 = mag2; bestIdx = fi }
            }
            if (bestIdx >= 0 && colors && bestIdx < colors.length) colorHex = colors[bestIdx]
          } else if (Array.isArray(featureIndices) && featureIndices.length === 1) {
            const idx = featureIndices[0]
            if (colors && idx >= 0 && idx < colors.length) colorHex = colors[idx]
          } else if (dominant && colors && colors.length) {
            const fid = dominant[mi]?.[mj]
            if (typeof fid === 'number' && fid >= 0 && fid < colors.length) colorHex = colors[fid]
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
      const headDyn = Math.max(4, Math.min(headLen, len * 0.35))
      const ex = sx0 + dirx * len
      const ey = sy0 + diry * len
      ctx.beginPath(); ctx.moveTo(sx0, sy0); ctx.lineTo(ex, ey); ctx.stroke()
      const ang = Math.atan2(ey - sy0, ex - sx0)
      const hx1 = ex - headDyn * Math.cos(ang - Math.PI / 7)
      const hy1 = ey - headDyn * Math.sin(ang - Math.PI / 7)
      const hx2 = ex - headDyn * Math.cos(ang + Math.PI / 7)
      const hy2 = ey - headDyn * Math.sin(ang + Math.PI / 7)
      ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(hx1, hy1); ctx.moveTo(ex, ey); ctx.lineTo(hx2, hy2); ctx.stroke()
    }
  }
      rafRef.current = requestAnimationFrame(draw)
    }
    runningRef.current = true
    rafRef.current = requestAnimationFrame(draw)

    // Hover handler to report world coords
    function handleMove(e) {
      if (!onHover) return
      const rect = canvas.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      const v = viewRef.current
      const x = v.xmin + (cx / Wpx) * (v.xmax - v.xmin)
      const y = v.ymin + ((Hpx - cy) / Hpx) * (v.ymax - v.ymin)
      onHover({ x, y })
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
      viewRef.current = {
        xmin: Math.max(xmin, Math.min(xmax - clampedW, nxmin)),
        xmax: Math.min(xmax, Math.max(xmin + clampedW, nxmax)),
        ymin: Math.max(ymin, Math.min(ymax - clampedH, nymin)),
        ymax: Math.min(ymax, Math.max(ymin + clampedH, nymax)),
      }
    }
    canvas.addEventListener('wheel', handleWheel, { passive: false })

    // Click handler to select grid cell
    function handleClick(e) {
      if (selectPointsModeRef.current && typeof onSelectPoint === 'function') {
        const rect = canvas.getBoundingClientRect()
        const cx = e.clientX - rect.left
        const cy = e.clientY - rect.top
        const R = 14
        const R2 = R*R
        let best = -1
        let bestD2 = Infinity
        for (let pIdx = 0; pIdx < positions.length; pIdx++) {
          const [px, py] = positions[pIdx]
          const [sx, sy] = worldToScreen(px, py)
          const dx = sx - cx, dy = sy - cy
          const d2 = dx*dx + dy*dy
          if (d2 <= R2 && d2 < bestD2) { bestD2 = d2; best = pIdx }
        }
        if (best >= 0) { try { onSelectPoint({ idx: best, append: !!e.shiftKey }) } catch {} }
        return
      }
      if (!onSelectCell || selectPointsModeRef.current) return
      const rect = canvas.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      const v = viewRef.current
      const x = v.xmin + (cx / Wpx) * (v.xmax - v.xmin)
      const y = v.ymin + ((Hpx - cy) / Hpx) * (v.ymax - v.ymin)
      // grid indices
      const j = Math.max(0, Math.min(W - 1, Math.floor((x - xmin) / (xmax - xmin) * W)))
      const i = Math.max(0, Math.min(H - 1, Math.floor((y - ymin) / (ymax - ymin) * H)))
      onSelectCell({ i, j, shift: !!e.shiftKey })
    }
    canvas.addEventListener('click', handleClick)

    // Brush handlers
    function handleDown(e) {
      const rect = canvas.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      pointPointerDownRef.current = true
      pointBrushingRef.current = false
      pointDownPosRef.current = { x: cx, y: cy }
      if (selectPointsModeRef.current) {
        // Do not start grid brushing in point-select mode
        return
      }
      const x = xmin + (cx / Wpx) * (xmax - xmin)
      const y = ymin + ((Hpx - cy) / Hpx) * (ymax - ymin)
      const j = Math.max(0, Math.min(W - 1, Math.floor((x - xmin) / (xmax - xmin) * W)))
      const i = Math.max(0, Math.min(H - 1, Math.floor((y - ymin) / (ymax - ymin) * H)))
      lastBrushRef.current = { i, j }
      brushingRef.current = true
      if (typeof brushCbRef.current === 'function') {
        try { brushCbRef.current({ i, j }) } catch {}
      }
    }
    function handleUp() {
      pointPointerDownRef.current = false
      pointBrushingRef.current = false
      brushingRef.current = false
      lastBrushRef.current = { i: -1, j: -1 }
    }
    canvas.addEventListener('mousedown', handleDown)
    window.addEventListener('mouseup', handleUp)
    canvas.addEventListener('mouseleave', handleUp)

    return () => {
      runningRef.current = false
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      canvas.removeEventListener('mousemove', handleMove)
      canvas.removeEventListener('wheel', handleWheel)
      canvas.removeEventListener('click', handleClick)
      canvas.removeEventListener('mousedown', handleDown)
      window.removeEventListener('mouseup', handleUp)
      canvas.removeEventListener('mouseleave', handleUp)
    }
  }, [
    bbox,
    grid_res,
    uSum,
    vSum,
    p99ByFeature,
    positions,
    point_labels,
    colors,
    dominant,
    featureIndices,
    pointColorStats,
    particleCount,
    tailLength,
    trailTailMin,
    trailTailExp,
    maxLifetime,
    speedScale,
    trailLineWidth,
    showCellAggregatedGradients,
    showCellGradients,
    showPointGradients,
    colorCells,
    showLIC,
    useMask,
  ])

  const canvasWidth = (typeof width === 'number' && width > 0) ? width : size
  const canvasHeight = (typeof height === 'number' && height > 0) ? height : size
  return (
    <canvas
      ref={canvasRef}
      width={canvasWidth}
      height={canvasHeight}
      style={{ width: `${canvasWidth}px`, height: `${canvasHeight}px`, border: '1px solid #ddd' }}
    />
  )
}
