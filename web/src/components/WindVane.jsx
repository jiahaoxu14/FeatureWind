import React, { useEffect, useRef, useMemo, useState } from 'react'

function bilinear(grid, gx, gy) {
  const H = grid.length, W = grid[0].length
  if (H === 0 || W === 0) return 0
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

export default function WindVane({ payload, focus, selectedCells = [], size = null, useConvexHull = true, showHull = false, showLabels = false, featureIndices = null }) {
  const containerRef = useRef(null)
  const canvasRef = useRef(null)
  const [canvasSize, setCanvasSize] = useState(typeof size === 'number' && size > 0 ? size : 220)
  const [hoverPt, setHoverPt] = useState(null)

  // Responsive sizing: if size prop not provided, fill container width
  useEffect(() => {
    if (typeof size === 'number' && size > 0) {
      setCanvasSize(size)
      return
    }
    function measure() {
      const el = containerRef.current
      if (!el) return
      const w = Math.max(0, el.clientWidth)
      const s = Math.max(120, Math.floor(w))
      setCanvasSize(s)
    }
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [size])

  const { bbox = [0, 1, 0, 1], grid_res = 25, uAll = [], vAll = [], colors = [], selection = {}, unmasked = null, dominant = null, col_labels = [] } = payload || {}
  const [xmin, xmax, ymin, ymax] = bbox
  const H = grid_res, W = grid_res

  const indices = useMemo(() => {
    if (Array.isArray(featureIndices) && featureIndices.length > 0) return featureIndices
    if (!selection) return []
    if (selection.topKIndices) return selection.topKIndices
    if (selection.featureIndex !== undefined) return [selection.featureIndex]
    return []
  }, [selection, featureIndices])

  // Resolve focus (x,y) → (gx,gy)
  const [gx, gy] = useMemo(() => {
    let x = (xmin + xmax) / 2
    let y = (ymin + ymax) / 2
    if (focus && typeof focus.x === 'number' && typeof focus.y === 'number') {
      x = Math.max(xmin, Math.min(xmax, focus.x))
      y = Math.max(ymin, Math.min(ymax, focus.y))
    }
    const gxf = (x - xmin) / (xmax - xmin) * (W - 1)
    const gyf = (y - ymin) / (ymax - ymin) * (H - 1)
    return [gxf, gyf]
  }, [focus, xmin, xmax, ymin, ymax, W, H])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !uAll || !vAll || indices.length === 0) {
      const ctx = canvas?.getContext?.('2d')
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
      }
      return
    }
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height
    ctx.clearRect(0, 0, w, h)

    const cx = w / 2
    const cy = h / 2
    const ringR = Math.min(w, h) * 0.35

    const hasUnmasked = Array.isArray(unmasked) && unmasked.length === H && unmasked[0].length === W
    const hasDominant = Array.isArray(dominant) && dominant.length === H && dominant[0].length === W
    const useSelection = Array.isArray(selectedCells) && selectedCells.length > 0
    // Compute grid indices using Python's floor mapping from world coordinates
    let xCoord = (xmin + xmax) / 2
    let yCoord = (ymin + ymax) / 2
    if (focus && typeof focus.x === 'number' && typeof focus.y === 'number') {
      xCoord = Math.max(xmin, Math.min(xmax, focus.x))
      yCoord = Math.max(ymin, Math.min(ymax, focus.y))
    }
    const cj = Math.max(0, Math.min(W - 1, Math.floor(((xCoord - xmin) / (xmax - xmin)) * W)))
    const ci = Math.max(0, Math.min(H - 1, Math.floor(((yCoord - ymin) / (ymax - ymin)) * H)))
    // Determine masked state
    let isMasked = false
    let validCells = []
    if (useSelection) {
      if (hasUnmasked) {
        for (const c of selectedCells) {
          const ii = Math.max(0, Math.min(H - 1, (c.i|0)))
          const jj = Math.max(0, Math.min(W - 1, (c.j|0)))
          if (unmasked[ii][jj]) validCells.push({ i: ii, j: jj })
        }
      } else {
        validCells = selectedCells.map(c => ({ i: Math.max(0, Math.min(H - 1, (c.i|0))), j: Math.max(0, Math.min(W - 1, (c.j|0))) }))
      }
      if (validCells.length === 0) isMasked = true
    } else {
      if (hasUnmasked) {
        isMasked = !unmasked[ci][cj]
      } else if (hasDominant) {
        isMasked = (dominant[ci][cj] === -1)
      }
    }

    // Sample per-feature vectors at (gx,gy) and accumulate
    const featureVectors = []
    let maxMag = 1e-12
    let sumUx = 0, sumVy = 0
    for (const idx of indices) {
      let u = 0, v = 0
      if (Array.isArray(selectedCells) && selectedCells.length > 0) {
        // Aggregate across selected valid cells (cell-center values)
        for (const c of validCells) {
          u += (uAll[idx]?.[c.i]?.[c.j] ?? 0)
          v += (vAll[idx]?.[c.i]?.[c.j] ?? 0)
        }
      } else {
        // Hover-only: snap to cell center values (no bilinear)
        u = (uAll[idx]?.[ci]?.[cj] ?? 0)
        v = (vAll[idx]?.[ci]?.[cj] ?? 0)
      }
      const mag = Math.hypot(u, v)
      maxMag = Math.max(maxMag, mag)
      sumUx += u
      sumVy += v
      const color = colors[idx % colors.length] || '#888888'
      featureVectors.push({ u, v, mag, color })
    }
    // Fallback mask check by magnitude if we didn't have mask grids
    if (!isMasked && !(Array.isArray(selectedCells) && selectedCells.length > 0)) {
      const sumMag = Math.hypot(sumUx, sumVy)
      if (!(hasUnmasked || hasDominant) && sumMag <= 1e-6) {
        isMasked = true
      }
    }

    // Draw ring
    ctx.strokeStyle = '#999'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.arc(cx, cy, ringR, 0, Math.PI * 2)
    ctx.stroke()

    if (!isMasked) {
      // Compute endpoints in canvas coordinates for convex-hull filtering and hover labeling
      const endpoints = featureVectors.map((fv, i) => {
        const scale = fv.mag > 0 ? (ringR * 0.9 * (fv.mag / (maxMag || 1))) : 0
        const ux = fv.mag > 0 ? (fv.u / fv.mag) : 0
        const uy = fv.mag > 0 ? (fv.v / fv.mag) : 0
        const x2 = cx + ux * scale
        const y2 = cy - uy * scale
        return { x2, y2, ux, uy, idx: indices[i], color: fv.color }
      })

      // Convex hull (Monotone chain) over endpoints (skip near-duplicate points)
      function convexHullIdx(pts) {
        if (!pts || pts.length < 3) return pts.map((_, i) => i)
        // Build array of [x,y,idx]
        const arr = pts.map((p, i) => [p.x2, p.y2, i])
        // Sort by x then y
        arr.sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]))
        const cross = (o, a, b) => (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        const lower = []
        for (const p of arr) {
          while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0) lower.pop()
          lower.push(p)
        }
        const upper = []
        for (let k = arr.length - 1; k >= 0; k--) {
          const p = arr[k]
          while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0) upper.pop()
          upper.push(p)
        }
        // Concatenate and drop last of each (duplicate endpoints)
        const hull = lower.slice(0, -1).concat(upper.slice(0, -1))
        // Return unique indices in order
        const seen = new Set()
        const idxs = []
        for (const h of hull) { if (!seen.has(h[2])) { seen.add(h[2]); idxs.push(h[2]) } }
        return idxs
      }

      let hullSet = null
      if (useConvexHull && endpoints.length >= 3) {
        try {
          const hullIdx = convexHullIdx(endpoints)
          hullSet = new Set(hullIdx)
          if (showHull && hullIdx.length >= 3) {
            ctx.strokeStyle = 'rgba(114,114,114,0.7)'
            ctx.lineWidth = 1.0
            ctx.beginPath()
            for (let k = 0; k < hullIdx.length; k++) {
              const p = endpoints[hullIdx[k]]
              if (k === 0) ctx.moveTo(p.x2, p.y2); else ctx.lineTo(p.x2, p.y2)
            }
            // close
            const p0 = endpoints[hullIdx[0]]
            ctx.lineTo(p0.x2, p0.y2)
            ctx.stroke()
          }
        } catch (e) { hullSet = null }
      }

      // Draw per-feature arrows (filtered by hull if enabled) with arrow heads
      featureVectors.forEach((fv, i) => {
        if (hullSet && !hullSet.has(i)) return
        const scale = fv.mag > 0 ? (ringR * 0.9 * (fv.mag / (maxMag || 1))) : 0
        const ux = fv.mag > 0 ? (fv.u / fv.mag) : 0
        const uy = fv.mag > 0 ? (fv.v / fv.mag) : 0
        const x2 = cx + ux * scale
        const y2 = cy - uy * scale
        ctx.strokeStyle = fv.color
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(cx, cy)
        ctx.lineTo(x2, y2)
        ctx.stroke()
        // Arrow head at endpoint (simple V-shape)
        try {
          const angle = Math.atan2(y2 - cy, x2 - cx)
          const headLen = Math.max(6, ringR * 0.06) // pixels
          const phi = Math.PI / 8 // 22.5 degrees
          const xh1 = x2 - headLen * Math.cos(angle - phi)
          const yh1 = y2 - headLen * Math.sin(angle - phi)
          const xh2 = x2 - headLen * Math.cos(angle + phi)
          const yh2 = y2 - headLen * Math.sin(angle + phi)
          ctx.beginPath()
          ctx.moveTo(x2, y2)
          ctx.lineTo(xh1, yh1)
          ctx.moveTo(x2, y2)
          ctx.lineTo(xh2, yh2)
          ctx.stroke()
        } catch (e) { /* ignore arrow head errors */ }

        // Optional: feature label at the end of the vector
        if (showLabels) {
          try {
            const featIdx = indices[i]
            const label = (Array.isArray(col_labels) && featIdx >= 0 && featIdx < col_labels.length) ? String(col_labels[featIdx]) : String(featIdx)
            // Offset label forward along the arrow and slightly perpendicular to avoid overlap
            const forward = Math.max(14, ringR * 0.10)
            const normalOff = Math.max(6, ringR * 0.04)
            // On-screen radial dir is (ux, -uy); a left normal is (uy, ux)
            const lx = x2 + ux * forward + (uy) * normalOff
            const ly = y2 - uy * forward + (ux) * normalOff
            ctx.font = `${Math.max(10, Math.floor(ringR * 0.10))}px sans-serif`
            ctx.textAlign = 'left'
            ctx.textBaseline = 'middle'
            // White halo for readability
            ctx.lineWidth = 3
            ctx.strokeStyle = '#ffffff'
            ctx.strokeText(label, lx, ly)
            // Colored text matching the vector color
            ctx.fillStyle = fv.color || '#111'
            ctx.fillText(label, lx, ly)
          } catch (e) { /* ignore */ }
        }
      })

      // Hover label (single): show nearest endpoint within threshold
      if (hoverPt && endpoints && endpoints.length) {
        let best = -1
        let bestD2 = Infinity
        for (let i = 0; i < endpoints.length; i++) {
          if (hullSet && !hullSet.has(i)) continue
          const ep = endpoints[i]
          const dx = ep.x2 - hoverPt.x
          const dy = ep.y2 - hoverPt.y
          const d2 = dx*dx + dy*dy
          if (d2 < bestD2) { bestD2 = d2; best = i }
        }
        if (best >= 0) {
          const T = Math.max(10, ringR * 0.08)
          if (Math.sqrt(bestD2) <= T) {
            const ep = endpoints[best]
            const featIdx = ep.idx
            const label = (Array.isArray(col_labels) && featIdx >= 0 && featIdx < col_labels.length) ? String(col_labels[featIdx]) : String(featIdx)
            // Offset past the head and a bit perpendicular to avoid overlap
            const forward = Math.max(14, ringR * 0.10)
            const normalOff = Math.max(6, ringR * 0.04)
            const nux = ep.ux || 0
            const nuy = ep.uy || 0
            const lx = ep.x2 + nux * forward + (nuy) * normalOff
            const ly = ep.y2 - nuy * forward + (nux) * normalOff
            ctx.font = `${Math.max(10, Math.floor(ringR * 0.10))}px sans-serif`
            ctx.textAlign = 'left'
            ctx.textBaseline = 'middle'
            ctx.lineWidth = 3
            ctx.strokeStyle = '#ffffff'
            ctx.strokeText(label, lx, ly)
            ctx.fillStyle = ep.color || '#111'
            ctx.fillText(label, lx, ly)
          }
        }
      }

      // Draw direction dot on ring for summed vector with original color/alpha method
      const sumMag = Math.hypot(sumUx, sumVy)
      if (sumMag > 1e-12) {
        const dirX = sumUx / sumMag
        const dirY = sumVy / sumMag
        const dx = dirX * ringR
        const dy = dirY * ringR
        const dotX = cx + dx
        const dotY = cy - dy

        // Choose color: feature with highest projection along flow direction
        let bestIdx = null
        let bestProj = -Infinity
        featureVectors.forEach((fv, i) => {
          const proj = fv.u * dirX + fv.v * dirY
          if (proj > bestProj) { bestProj = proj; bestIdx = indices[i] }
        })
        let dotColor = '#333'
        if (bestIdx !== null && bestIdx >= 0 && bestIdx < colors.length) {
          dotColor = colors[bestIdx]
        }

        // Opacity: field-based relative to global maximum sum magnitude
        let dotAlpha = 0.9
        try {
          const gmax = payload?.global_sum_magnitude_max || 0
          const base = gmax > 1e-12 ? (sumMag / gmax) : 0
          const ratio = Math.tanh(base)
          dotAlpha = Math.max(0.15, Math.min(1.0, 0.15 + 0.85 * ratio))
        } catch (e) { /* keep default */ }

        ctx.fillStyle = dotColor
        ctx.globalAlpha = dotAlpha
        ctx.beginPath()
        ctx.arc(dotX, dotY, Math.max(4, ringR * 0.06), 0, Math.PI * 2)
        ctx.fill()
        // optional guide
        ctx.globalAlpha = dotAlpha * 0.5
        ctx.strokeStyle = dotColor
        ctx.beginPath()
        ctx.moveTo(cx, cy)
        ctx.lineTo(dotX, dotY)
        ctx.stroke()
        ctx.globalAlpha = 1.0
      }
    }

    // Center
    ctx.fillStyle = '#666'
    ctx.beginPath()
    ctx.arc(cx, cy, 3, 0, Math.PI * 2)
    ctx.fill()
  }, [uAll, vAll, colors, indices, gx, gy, unmasked, dominant, selectedCells, useConvexHull, showHull, canvasSize, showLabels, col_labels, hoverPt])

  // Mouse handlers for hover labeling
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    function handleMove(e) {
      const rect = canvas.getBoundingClientRect()
      setHoverPt({ x: e.clientX - rect.left, y: e.clientY - rect.top })
    }
    function handleLeave() { setHoverPt(null) }
    canvas.addEventListener('mousemove', handleMove)
    canvas.addEventListener('mouseleave', handleLeave)
    return () => {
      canvas.removeEventListener('mousemove', handleMove)
      canvas.removeEventListener('mouseleave', handleLeave)
    }
  }, [])

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <canvas
        ref={canvasRef}
        width={canvasSize}
        height={canvasSize}
        style={{ width: '100%', height: 'auto', border: '1px solid #eee', display: 'block' }}
      />
    </div>
  )
}
