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

export default function CanvasWind({ payload, particleCount = 300, onHover }) {
  const canvasRef = useRef(null)

  const {
    bbox = [0, 1, 0, 1],
    grid_res = 25,
    uAll = [],
    vAll = [],
    positions = [],
    selection = {},
  } = payload || {}

  const indices = useMemo(() => {
    if (!selection) return []
    if (selection.topKIndices) return selection.topKIndices
    if (selection.featureIndex !== undefined) return [selection.featureIndex]
    return []
  }, [selection])

  const uSum = useMemo(() => sumSelectedGrid(uAll, indices), [uAll, indices])
  const vSum = useMemo(() => sumSelectedGrid(vAll, indices), [vAll, indices])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !uSum || !vSum) return
    const ctx = canvas.getContext('2d')
    const [xmin, xmax, ymin, ymax] = bbox
    const Wpx = canvas.width
    const Hpx = canvas.height
    const W = grid_res, H = grid_res

    // Particles
    const particles = Array.from({ length: particleCount }, () => ({
      x: xmin + Math.random() * (xmax - xmin),
      y: ymin + Math.random() * (ymax - ymin),
    }))

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
      for (const p of particles) {
        const [gx, gy] = worldToGrid(p.x, p.y)
        const u = bilinearSample(uSum, gx, gy)
        const v = bilinearSample(vSum, gx, gy)
        p.x += u * dt
        p.y += v * dt
        // respawn if out of bounds
        if (p.x < xmin || p.x > xmax || p.y < ymin || p.y > ymax) {
          p.x = xmin + Math.random() * (xmax - xmin)
          p.y = ymin + Math.random() * (ymax - ymin)
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

      // Draw base positions (for sanity)
      ctx.fillStyle = 'rgba(120,120,120,0.5)'
      for (const [x, y] of positions) {
        const [sx, sy] = worldToScreen(x, y)
        ctx.beginPath()
        ctx.arc(sx, sy, 2, 0, Math.PI * 2)
        ctx.fill()
      }

      // Draw particles
      ctx.fillStyle = 'rgba(20,20,20,0.9)'
      for (const p of particles) {
        const [sx, sy] = worldToScreen(p.x, p.y)
        ctx.beginPath()
        ctx.arc(sx, sy, 1.5, 0, Math.PI * 2)
        ctx.fill()
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
