import React, { useMemo } from 'react'

export default function PointSeries({
  indices = [],
  featureValues = [],
  featureIndices = [],
  onChangeFeatureIndices,
  colLabels = [],
  colors = [],
  width = 600,
  height = 220,
}) {
  const { xs, lines, ymin, ymax } = useMemo(() => {
    if (!Array.isArray(indices) || indices.length === 0) return { xs: [], lines: [], ymin: 0, ymax: 1 }
    const xs = indices.map((_, i) => i)
    const fids = Array.isArray(featureIndices) && featureIndices.length ? featureIndices : []
    let ymin = Infinity, ymax = -Infinity
    const lines = fids.map((fid) => {
      const ys = indices.map((rowIdx) => {
        const row = featureValues?.[rowIdx]
        if (!row || fid < 0 || fid >= row.length) return NaN
        const v = Number(row[fid])
        return Number.isFinite(v) ? v : NaN
      })
      for (const v of ys) { if (Number.isFinite(v)) { if (v < ymin) ymin = v; if (v > ymax) ymax = v } }
      return { fid, ys }
    })
    if (!Number.isFinite(ymin) || !Number.isFinite(ymax) || ymin === ymax) { ymin = 0; ymax = 1 }
    return { xs, lines, ymin, ymax }
  }, [indices, featureValues, featureIndices])

  const padL = 36, padR = 8, padT = 10, padB = 20
  const W = Math.max(100, width), H = Math.max(120, height)
  const plotW = W - padL - padR, plotH = H - padT - padB
  function sx(i) { return padL + (xs.length <= 1 ? 0 : (i / (xs.length - 1)) * plotW) }
  function sy(v) { return padT + (1 - (v - ymin) / (ymax - ymin)) * plotH }
  function polyAttr(ys){ return xs.map((_, i) => `${sx(i)},${sy(ys[i])}`).join(' ') }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ fontSize: 13, color: '#6b7280' }}>Selected Points: {indices.length}</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 12, color: '#6b7280' }}>Features</span>
          <select
            multiple
            size={Math.min(8, Math.max(3, colLabels.length))}
            value={Array.isArray(featureIndices) ? featureIndices.map(String) : []}
            onChange={(e) => {
              const opts = Array.from(e.target.selectedOptions).map((o) => parseInt(o.value, 10)).filter((x) => Number.isFinite(x))
              onChangeFeatureIndices && onChangeFeatureIndices(opts)
            }}
            style={{ height: 'auto', minHeight: 28 }}
          >
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
        {Array.isArray(lines) && lines.map((ln) => {
          const pts = polyAttr(ln.ys)
          const col = (Array.isArray(colors) && ln.fid >= 0 && ln.fid < colors.length && colors[ln.fid]) ? colors[ln.fid] : '#2563eb'
          return (<polyline key={`ln-${ln.fid}`} points={pts} fill="none" stroke={col} strokeWidth={2} />)
        })}
      </svg>
    </div>
  )
}
