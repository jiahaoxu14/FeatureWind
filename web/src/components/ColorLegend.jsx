import React, { useMemo } from 'react'

export default function ColorLegend({ payload }) {
  const { col_labels = [], colors = [], selection = {}, family_assignments = null } = payload || {}

  const selected = useMemo(() => {
    if (!selection) return new Set()
    if (Array.isArray(selection.topKIndices)) return new Set(selection.topKIndices)
    if (typeof selection.featureIndex === 'number') return new Set([selection.featureIndex])
    return new Set()
  }, [selection])

  if (Array.isArray(family_assignments) && family_assignments.length === col_labels.length) {
    // Group features by family id
    const famMap = new Map()
    family_assignments.forEach((fam, idx) => {
      if (!famMap.has(fam)) famMap.set(fam, [])
      famMap.get(fam).push(idx)
    })
    const famIds = Array.from(famMap.keys()).sort((a, b) => a - b)
    return (
      <div className="legend-list">
        {famIds.map((famId) => {
          const indices = famMap.get(famId)
          const repIdx = indices && indices.length ? indices[0] : 0
          const famColor = colors[repIdx] || '#888'
          return (
            <div key={`fam-${famId}`} style={{ marginBottom: 8 }}>
              <div className="legend-item" style={{ fontWeight: 600 }}>
                <span className="legend-swatch" style={{ background: famColor }} />
                <span>Family {famId}</span>
              </div>
              {indices.map((idx) => (
                <div key={idx} className={`legend-item${selected.has(idx) ? ' selected' : ''}`} style={{ paddingLeft: 22 }}>
                  <span className="legend-swatch" style={{ background: colors[idx] || famColor }} />
                  <span title={col_labels[idx]}>{col_labels[idx]}</span>
                </div>
              ))}
            </div>
          )
        })}
      </div>
    )
  }
  // Fallback: flat list
  return (
    <div className="legend-list">
      {col_labels.map((name, idx) => (
        <div key={idx} className={`legend-item${selected.has(idx) ? ' selected' : ''}`}>
          <span className="legend-swatch" style={{ background: colors[idx] || '#888' }} />
          <span title={name}>{name}</span>
        </div>
      ))}
    </div>
  )
}
