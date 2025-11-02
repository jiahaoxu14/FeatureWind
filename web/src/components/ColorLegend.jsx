import React, { useMemo } from 'react'

import { useState, useEffect } from 'react'
export default function ColorLegend({ payload, dataset, onApplyFamilies, visible, selectedFeatures = [], onChangeSelectedFeatures }) {
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
  const [dragFeature, setDragFeature] = useState(null)
  const [dragOverFam, setDragOverFam] = useState(null)

  // Effective selection reflects current visualization: manual selection first,
  // otherwise fall back to backend-provided selection (Top-K or single feature)
  const effectiveSelectedArray = useMemo(() => {
    if (Array.isArray(selectedFeatures) && selectedFeatures.length > 0) return selectedFeatures
    if (selection && Array.isArray(selection.topKIndices)) return selection.topKIndices
    if (selection && typeof selection.featureIndex === 'number') return [selection.featureIndex]
    return []
  }, [selectedFeatures, selection])
  const selectedSet = useMemo(() => new Set(effectiveSelectedArray), [effectiveSelectedArray])

  function setSelected(next) {
    if (typeof onChangeSelectedFeatures === 'function') {
      const arr = Array.from(new Set(next)).sort((a, b) => a - b)
      onChangeSelectedFeatures(arr)
    }
  }

  // Build a ranking of features to support a Top-K selection slider inside the legend
  const ranking = useMemo(() => {
    const n = Array.isArray(col_labels) ? col_labels.length : 0
    const tk = (selection && Array.isArray(selection.topKIndices)) ? selection.topKIndices : null
    if (!n) return []
    if (Array.isArray(tk) && tk.length === n) return tk.slice()
    if (Array.isArray(tk) && tk.length > 0) {
      const set = new Set(tk)
      const rest = []
      for (let i = 0; i < n; i++) if (!set.has(i)) rest.push(i)
      return tk.concat(rest)
    }
    return [...Array(n).keys()]
  }, [selection, col_labels])

  function handleDragStartFeature(idx, e) {
    setDragFeature(idx)
    try { e.dataTransfer.setData('text/plain', String(idx)); e.dataTransfer.effectAllowed = 'move' } catch {}
  }
  function handleDragOverFamily(famId, e) { e.preventDefault(); setDragOverFam(famId); try { e.dataTransfer.dropEffect = 'move' } catch {} }
  function handleDragLeaveFamily() { setDragOverFam(null) }
  function handleDropToFamily(famId, e) {
    e.preventDefault()
    let idx = dragFeature
    try {
      const t = parseInt(e.dataTransfer.getData('text/plain'), 10)
      if (Number.isFinite(t)) idx = t
    } catch {}
    if (idx === null || idx === undefined) return
    setFamilies((prev) => {
      const next = [...prev]
      next[idx] = famId
      return next
    })
    setDragOverFam(null)
    setDragFeature(null)
  }
  function handleDropToNewFamily(e) {
    e.preventDefault()
    let idx = dragFeature
    try {
      const t = parseInt(e.dataTransfer.getData('text/plain'), 10)
      if (Number.isFinite(t)) idx = t
    } catch {}
    if (idx === null || idx === undefined) return
    const nextFam = families.length ? Math.max(...families.map((f) => parseInt(f, 10))) + 1 : 0
    setFamilies((prev) => {
      const next = [...prev]
      next[idx] = nextFam
      return next
    })
    setDragOverFam(null)
    setDragFeature(null)
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
        {/* Top-K selection slider inside color family panel; updates legend checkboxes */}
        <div className="legend-item" style={{ gap: 10 }}>
          <label style={{ fontSize: 12, color: '#6b7280' }}>Top-K</label>
          <div className="slider-row" style={{ flex: 1 }}>
            <input
              type="range"
              min={0}
              max={col_labels.length || 0}
              step={1}
              value={effectiveSelectedArray.length}
              onChange={(e) => {
                const k = Math.max(0, Math.min((col_labels.length || 0), Number(e.target.value)))
                const next = ranking.slice(0, k)
                setSelected(next)
              }}
            />
            <span className="control-val">{effectiveSelectedArray.length}</span>
          </div>
        </div>
        <div className="spacer" />
        {famIds.map((famId) => {
          const indices = famMap.get(famId)
          const repIdx = indices && indices.length ? indices[0] : 0
          const famColor = colors[repIdx] || '#888'
          const selInFam = indices.filter((i) => selectedSet.has(i))
          const allSelected = selInFam.length === indices.length
          return (
            <div key={`fam-${famId}`} style={{ marginBottom: 8 }}
                 onDragOver={editMode ? (e) => handleDragOverFamily(famId, e) : undefined}
                 onDragLeave={editMode ? handleDragLeaveFamily : undefined}
                 onDrop={editMode ? (e) => handleDropToFamily(famId, e) : undefined}
                 className={dragOverFam === famId ? 'drop-target' : ''}
            >
              <div className="legend-item family-header" style={{ fontWeight: 600, gap: 8 }}>
                <span className="legend-swatch" style={{ background: famColor }} />
                <span>Family {famId}</span>
                <span style={{ flex: 1 }} />
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <button title="Add all in family" onClick={() => setSelected([...selectedSet, ...indices])} style={{ height: 22, padding: '0 6px', borderRadius: 4, border: '1px solid #e5e7eb', background: '#fff' }}>All+</button>
                  <button title="Select only this family" onClick={() => setSelected(indices)} style={{ height: 22, padding: '0 6px', borderRadius: 4, border: '1px solid #e5e7eb', background: '#fff' }}>Only</button>
                  <button title="Clear this family" onClick={() => setSelected([...Array.from(selectedSet)].filter(i => !indices.includes(i)))} style={{ height: 22, padding: '0 6px', borderRadius: 4, border: '1px solid #e5e7eb', background: '#fff' }}>Clear</button>
                </div>
              </div>
              {indices.map((idx) => (
                <label key={idx}
                       className={`legend-item feature${visibleSet.has(idx) ? ' visible' : ''} ${editMode ? 'draggable' : ''}`}
                       style={{ paddingLeft: 22, gap: 8 }}
                       draggable={!!editMode}
                       onDragStart={editMode ? (e) => handleDragStartFeature(idx, e) : undefined}
                >
                  <input
                    type="checkbox"
                    checked={selectedSet.has(idx)}
                    onChange={(e) => {
                      const next = new Set(selectedSet)
                      if (e.target.checked) next.add(idx); else next.delete(idx)
                      setSelected(Array.from(next))
                    }}
                    style={{ margin: 0 }}
                  />
                  <span className="legend-swatch" style={{ background: colors[idx] || famColor }} />
                  <span title={col_labels[idx]} style={{ flex: 1 }}>{col_labels[idx]}</span>
                </label>
              ))}
            </div>
          )
        })}
        {editMode && (
          <div className={`new-family ${dragOverFam === '__new__' ? 'drop-target' : ''}`}
               onDragOver={(e) => handleDragOverFamily('__new__', e)}
               onDragLeave={handleDragLeaveFamily}
               onDrop={handleDropToNewFamily}
          >
            + New Family (drop feature here)
          </div>
        )}
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
      {/* Top-K selection slider for fallback list as well */}
      <div className="legend-item" style={{ gap: 10 }}>
        <label style={{ fontSize: 12, color: '#6b7280' }}>Top-K</label>
        <div className="slider-row" style={{ flex: 1 }}>
          <input
            type="range"
            min={0}
            max={col_labels.length || 0}
            step={1}
            value={effectiveSelectedArray.length}
            onChange={(e) => {
              const k = Math.max(0, Math.min((col_labels.length || 0), Number(e.target.value)))
              const next = ranking.slice(0, k)
              setSelected(next)
            }}
          />
          <span className="control-val">{effectiveSelectedArray.length}</span>
        </div>
      </div>
      <div className="spacer" />
      {col_labels.map((name, idx) => (
        <label key={idx} className={`legend-item${visibleSet.has(idx) ? ' visible' : ''}`} style={{ gap: 8 }}>
          <input
            type="checkbox"
            checked={selectedSet.has(idx)}
            onChange={(e) => {
              const next = new Set(selectedSet)
              if (e.target.checked) next.add(idx); else next.delete(idx)
              setSelected(Array.from(next))
            }}
            style={{ margin: 0 }}
          />
          <span className="legend-swatch" style={{ background: colors[idx] || '#888' }} />
          <span title={name}>{name}</span>
        </label>
      ))}
    </div>
  )
}
