import React, { useMemo } from 'react'

export default function PointSeries({
  indices = [],
  featureValues = [],
  featureIndex = 0,
  onChangeFeatureIndex,
  colLabels = [],
  width = 600,
  height = 220,
}) {
  const { xs, ys, ymin, ymax } = useMemo(() => {
    if (!Array.isArray(indices) || indices.length === 0) return { xs: [], ys: [], ymin: 0, ymax: 1 }
    const xs = indices.map((_, i) => i)
    const ys = indices.map((rowIdx) => {
      const row = featureValues?.[rowIdx]
      if (!row || featureIndex < 0 || featureIndex >= row.length) return NaN
      const v = Number(row[featureIndex])
      return Number.isFinite(v) ? v : NaN
    })
    let ymin = Infinity, ymax = -Infinity
    for (const v of ys) { if (Number.isFinite(v)) { if (v < ymin) ymin = v; if (v > ymax) ymax = v } }
    if (!Number.isFinite(ymin) || !Number.isFinite(ymax) || ymin === ymax) { ymin = 0; ymax = 1 }
    return { xs, ys, ymin, ymax }
  }, [indices, featureValues, featureIndex])

  const padL = 36, padR = 8, padT = 10, padB = 20
  const W = Math.max(100, width), H = Math.max(120, height)
  const plotW = W - padL - padR, plotH = H - padT - padB
  function sx(i) { return padL + (xs.length <= 1 ? 0 : (i / (xs.length - 1)) * plotW) }
  function sy(v) { return padT + (1 - (v - ymin) / (ymax - ymin)) * plotH }

  let pointsAttr = ''
  if (xs.length > 0) {
    pointsAttr = xs.map((_, i) => `${sx(i)},${sy(ys[i])}`).join(' ')
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ fontSize: 13, color: '#6b7280' }}>Selected Points: {indices.length}</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 12, color: '#6b7280' }}>Feature</span>
          <select value={featureIndex} onChange={(e) => onChangeFeatureIndex && onChangeFeatureIndex(Number(e.target.value))} style={{ height: 28 }}>
            {Array.isArray(colLabels) && colLabels.map((n, i) => (
              <option key={i} value={i}>{String(n)}</option>
            ))}
          </select>
        </div>
      </div>
      <svg width={W} height={H} style={{ display: 'block', border: '1px solid #eee', borderRadius: 6 }}>
        <rect x={0} y={0} width={W} height={H} fill="#fff" />
        {/* axes */}
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="#ddd" />
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="#ddd" />
        {/* y ticks */}
        {Array.from({ length: 5 }).map((_, k) => {
          const t = k / 4
          const y = padT + (1 - t) * plotH
          const val = ymin + (ymax - ymin) * t
          return (
            <g key={k}>
              <line x1={padL - 4} y1={y} x2={W - padR} y2={y} stroke="#f2f2f2" />
              <text x={padL - 6} y={y} textAnchor="end" dominantBaseline="middle" fill="#9ca3af" fontSize={11}>{val.toFixed(2)}</text>
            </g>
          )
        })}
        {pointsAttr && (
          <polyline points={pointsAttr} fill="none" stroke="#2563eb" strokeWidth={2} />
        )}
      </svg>
    </div>
  )
}

