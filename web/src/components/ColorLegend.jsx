import React, { useMemo } from 'react'

import { useState, useEffect } from 'react'
export default function ColorLegend({ payload, dataset, onApplyFamilies, visible }) {
  const { col_labels = [], colors = [], selection = {}, family_assignments = null } = payload || {}

  const visibleSet = useMemo(() => {
    if (visible && typeof visible.has === 'function') return visible
    // Fallback to selection if visible set not provided
    if (!selection) return new Set()
    if (Array.isArray(selection.topKIndices)) return new Set(selection.topKIndices)
    if (typeof selection.featureIndex === 'number') return new Set([selection.featureIndex])
    return new Set()
  }, [visible, selection])

  const [editMode, setEditMode] = useState(false)
  const [families, setFamilies] = useState(family_assignments || [])
  useEffect(() => { setFamilies(family_assignments || []) }, [family_assignments])

  function updateFam(idx, val) {
    const x = parseInt(val, 10)
    if (!Number.isFinite(x)) return
    setFamilies((prev) => {
      const next = [...prev]
      next[idx] = x
      return next
    })
  }

  if (Array.isArray(families) && families.length === col_labels.length) {
    // Group features by family id
    const famMap = new Map()
    families.forEach((fam, idx) => {
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
                <div key={idx} className={`legend-item${visibleSet.has(idx) ? ' visible' : ''}`} style={{ paddingLeft: 22 }}>
                  <span className="legend-swatch" style={{ background: colors[idx] || famColor }} />
                  <span title={col_labels[idx]} style={{ flex: 1 }}>{col_labels[idx]}</span>
                  {editMode && (
                    <input
                      type="number"
                      min={0}
                      step={1}
                      value={families[idx] ?? 0}
                      onChange={(e) => updateFam(idx, e.target.value)}
                      style={{ width: 64, height: 28, border: '1px solid #e5e7eb', borderRadius: 6 }}
                    />
                  )}
                </div>
              ))}
            </div>
          )
        })}
        <div className="legend-item" style={{ justifyContent: 'space-between', marginTop: 8 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={editMode} onChange={(e) => setEditMode(e.target.checked)} />
            Edit families
          </label>
          <button
            disabled={!editMode || !dataset}
            onClick={() => onApplyFamilies && onApplyFamilies(families)}
            style={{ height: 28, padding: '0 10px', borderRadius: 6, border: '1px solid #e5e7eb', background: editMode ? '#fff' : '#f3f4f6' }}
          >Apply</button>
        </div>
      </div>
    )
  }
  // Fallback: flat list
  return (
    <div className="legend-list">
      {col_labels.map((name, idx) => (
        <div key={idx} className={`legend-item${visibleSet.has(idx) ? ' visible' : ''}`}>
          <span className="legend-swatch" style={{ background: colors[idx] || '#888' }} />
          <span title={name}>{name}</span>
        </div>
      ))}
    </div>
  )
}
