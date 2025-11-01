import React, { useEffect, useRef, useMemo } from 'react'

function sumSelectedGrid(grids, indices) {
  if (!grids || grids.length === 0) return null
  const m = indices && indices.length ? indices : [...Array(grids.length).keys()]
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
  particleCount = 600,
  onHover,
  showGrid = true,
  consistentSpeed = true,
  speedConstRel = 0.06,
  speedScale = 1.0,
  tailLength = 10,
  trailTailMin = 0.10,
  trailTailExp = 2.0,
  maxLifetime = 200,
}) {
  const canvasRef = useRef(null)

  const {
    bbox = [0, 1, 0, 1],
    grid_res = 25,
    uAll = [],
    vAll = [],
    positions = [],
    selection = {},
    dominant = null,
    unmasked = null,
    colors = [],
  } = payload || {}

  const indices = useMemo(() => {
    if (!selection) return []
    if (selection.topKIndices) return selection.topKIndices
    if (selection.featureIndex !== undefined) return [selection.featureIndex]
    return []
  }, [selection])

  const uSum = useMemo(() => sumSelectedGrid(uAll, indices), [uAll, indices])
  const vSum = useMemo(() => sumSelectedGrid(vAll, indices), [vAll, indices])

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

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !uSum || !vSum) return
    const ctx = canvas.getContext('2d')
    const [xmin, xmax, ymin, ymax] = bbox
    const Wpx = canvas.width
    const Hpx = canvas.height
    const W = grid_res, H = grid_res

    // Trail configuration (mirrors defaults in featurewind/config.py)
    const TAIL_LENGTH = Math.max(1, Math.floor(tailLength))
    const TRAIL_TAIL_MIN = Math.max(0, Math.min(1, trailTailMin))
    const TRAIL_TAIL_EXP = Math.max(0.5, trailTailExp)
    const MAX_LIFETIME = Math.max(1, Math.floor(maxLifetime))
    const CONSISTENT_SPEED = !!consistentSpeed
    const SPEED_CONST_REL = speedConstRel // fraction of plot width per second
    const SPEED_SCALE = speedScale // used when CONSISTENT_SPEED=false

    // Optional mask helpers
    let hasMask = Array.isArray(unmasked) && unmasked.length === H && unmasked[0].length === W

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
      return { x, y, age: 0, hist }
    })

    function worldToGrid(x, y) {
      const gx = (x - xmin) / (xmax - xmin) * (W - 1)
      const gy = (y - ymin) / (ymax - ymin) * (H - 1)
      return [gx, gy]
    }
    function worldToScreen(x, y) {
      const sx = (x - xmin) / (xmax - xmin) * Wpx
      const sy = Hpx - (y - ymin) / (ymax - ymin) * Hpx
      return [sx, sy]
    }

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

        // Consistent speed option (like Python's CONSISTENT_PARTICLE_SPEED)
        if (CONSISTENT_SPEED) {
          const mag = Math.hypot(u, v)
          if (mag > 1e-9) {
            const width = (xmax - xmin)
            const s = SPEED_CONST_REL * width
            u = (u / mag) * s
            v = (v / mag) * s
          } else {
            u = 0; v = 0
          }
        } else {
          u *= SPEED_SCALE
          v *= SPEED_SCALE
        }

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
          for (let t = 0; t <= TAIL_LENGTH; t++) { p.hist[t].x = nx; p.hist[t].y = ny }
        }
      }
    }

    let last = performance.now()
    function draw() {
      const now = performance.now()
      const dt = Math.min((now - last) / 1000, 0.05)
      last = now
      step(dt)
      ctx.clearRect(0, 0, Wpx, Hpx)

      // Draw grid lines (cell boundaries)
      if (showGrid) {
        ctx.strokeStyle = 'rgba(180,180,180,0.35)'
        ctx.lineWidth = 0.5
        // vertical lines at x boundaries
        for (let k = 0; k <= W; k++) {
          const xk = xmin + (k / W) * (xmax - xmin)
          const sx = (xk - xmin) / (xmax - xmin) * Wpx
          ctx.beginPath()
          ctx.moveTo(sx, 0)
          ctx.lineTo(sx, Hpx)
          ctx.stroke()
        }
        // horizontal lines at y boundaries
        for (let k = 0; k <= H; k++) {
          const yk = ymin + (k / H) * (ymax - ymin)
          const sy = Hpx - (yk - ymin) / (ymax - ymin) * Hpx
          ctx.beginPath()
          ctx.moveTo(0, sy)
          ctx.lineTo(Wpx, sy)
          ctx.stroke()
        }
      }

      // Optional: draw base points lightly
      ctx.fillStyle = 'rgba(120,120,120,0.35)'
      for (const [x, y] of positions) {
        const [sx, sy] = worldToScreen(x, y)
        ctx.beginPath()
        ctx.arc(sx, sy, 1.6, 0, Math.PI * 2)
        ctx.fill()
      }

      // Draw trails as fading line segments (colored by dominant feature)
      ctx.lineWidth = 1.2
      ctx.lineCap = 'round'
      const p99 = magInfo?.p99 || 1.0

      function hexToRgb(hex) {
        if (!hex || typeof hex !== 'string') return [20, 20, 20]
        const m = hex.match(/^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i)
        if (!m) return [20, 20, 20]
        return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)]
      }
      for (const p of particles) {
        for (let t = TAIL_LENGTH - 1; t >= 0; t--) {
          const aTail = TRAIL_TAIL_MIN + (1 - TRAIL_TAIL_MIN) * Math.pow((t + 1) / TAIL_LENGTH, TRAIL_TAIL_EXP)
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

          // Sample dominant feature index (nearest cell) and color
          let rgb = [20, 20, 20]
          if (dominant && colors && colors.length) {
            const mi = Math.max(0, Math.min(H - 1, Math.round(gy0)))
            const mj = Math.max(0, Math.min(W - 1, Math.round(gx0)))
            const fid = dominant[mi]?.[mj]
            if (typeof fid === 'number' && fid >= 0 && fid < colors.length) {
              rgb = hexToRgb(colors[fid])
            }
          }

          // Respect mask: drop segments fully in masked cells
          if (hasMask) {
            const mi = Math.max(0, Math.min(H - 1, Math.round(gy0)))
            const mj = Math.max(0, Math.min(W - 1, Math.round(gx0)))
            if (!unmasked[mi][mj]) continue
          }

          const alpha = 0.9 * aTail * aField
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
      requestAnimationFrame(draw)
    }
    const raf = requestAnimationFrame(draw)

    // Hover handler to report world coords
    function handleMove(e) {
      if (!onHover) return
      const rect = canvas.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      const x = xmin + (cx / Wpx) * (xmax - xmin)
      const y = ymin + ((Hpx - cy) / Hpx) * (ymax - ymin)
      onHover({ x, y })
    }
    canvas.addEventListener('mousemove', handleMove)

    return () => {
      cancelAnimationFrame(raf)
      canvas.removeEventListener('mousemove', handleMove)
    }
  }, [bbox, grid_res, uSum, vSum, positions, particleCount])

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={600}
      style={{ width: '800px', height: '600px', border: '1px solid #ddd' }}
    />
  )
}
