import React, { useEffect, useMemo, useState } from 'react'

const MODE_META = {
  default: { label: 'Default', hint: 'Single feature, single hue.' },
  compare: { label: 'Compare', hint: 'Select up to 4 features.' },
}

export default function ColorLegend({
  payload,
  mode = 'default',
  onChangeMode,
  defaultFeatureIndex = null,
  compareFeatureIndices = [],
  onSelectFeature,
  onSelectAll,
  onToggleCompareFeature,
  onClearCompare,
  compareCap = 4,
  message = '',
  activeFeatureColorMap = {},
}) {
  const { col_labels = [], featureRanking = null } = payload || {}
  const [query, setQuery] = useState('')
  const isOverviewMode = mode === 'overview'
  const isCompareMode = mode === 'compare'
  const defaultTabActive = mode === 'default' || isOverviewMode

  useEffect(() => {
    setQuery('')
  }, [payload?.datasetId])

  const compareSet = useMemo(() => new Set(Array.isArray(compareFeatureIndices) ? compareFeatureIndices : []), [compareFeatureIndices])

  const ranking = useMemo(() => {
    const n = Array.isArray(col_labels) ? col_labels.length : 0
    const fallback = [...Array(n).keys()]
    if (!Array.isArray(featureRanking)) return fallback
    const seen = new Set()
    const ordered = []
    for (const raw of featureRanking) {
      const idx = Number(raw)
      if (!Number.isInteger(idx) || idx < 0 || idx >= n || seen.has(idx)) continue
      seen.add(idx)
      ordered.push(idx)
    }
    for (let idx = 0; idx < n; idx++) {
      if (!seen.has(idx)) ordered.push(idx)
    }
    return ordered
  }, [col_labels, featureRanking])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return ranking
    return ranking.filter((idx) => String(col_labels[idx] || '').toLowerCase().includes(q))
  }, [ranking, query, col_labels])

  const compareSelected = useMemo(() => {
    return (Array.isArray(compareFeatureIndices) ? compareFeatureIndices : [])
      .filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < col_labels.length)
  }, [compareFeatureIndices, col_labels.length])

  function handleMode(nextMode) {
    if (typeof onChangeMode === 'function') onChangeMode(nextMode)
  }

  function handleFeatureAction(idx) {
    if (isCompareMode) {
      if (typeof onToggleCompareFeature === 'function') onToggleCompareFeature(idx)
      return
    }
    if (typeof onSelectFeature === 'function') onSelectFeature(idx)
  }

  function isSelected(idx) {
    if (mode === 'default') return idx === defaultFeatureIndex
    if (isCompareMode) return compareSet.has(idx)
    return false
  }

  function rowActionLabel(idx) {
    if (isCompareMode) return compareSet.has(idx) ? 'Selected' : 'Add'
    if (isOverviewMode) return 'Open'
    return 'Select'
  }

  return (
    <div className="feature-panel">
      <div className="mode-tabs" role="tablist" aria-label="Feature view mode">
        <button
          type="button"
          className={`mode-tab${defaultTabActive ? ' active' : ''}`}
          onClick={() => handleMode('default')}
        >
          {MODE_META.default.label}
        </button>
        <button
          type="button"
          className={`mode-tab${isCompareMode ? ' active' : ''}`}
          onClick={() => handleMode('compare')}
        >
          {MODE_META.compare.label}
        </button>
      </div>

      <p className="feature-mode-hint">
        {isCompareMode
          ? MODE_META.compare.hint
          : isOverviewMode
            ? 'All features aggregated in grayscale. Choose a feature to return to single-feature view.'
            : MODE_META.default.hint}
      </p>

      <div className="feature-toolbar">
        {!isCompareMode && (
          <button
            type="button"
            className="btn"
            onClick={() => typeof onSelectAll === 'function' && onSelectAll()}
            disabled={isOverviewMode}
          >
            {isOverviewMode ? 'Showing All' : 'Select All'}
          </button>
        )}
        {isCompareMode && (
          <button
            type="button"
            className="btn"
            onClick={() => typeof onClearCompare === 'function' && onClearCompare()}
            disabled={compareSelected.length === 0}
          >
            Clear
          </button>
        )}
      </div>

      <div className="feature-search-row">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search features"
          className="feature-search"
        />
      </div>

      {isCompareMode && compareSelected.length > 0 && (
        <div className="feature-chip-list">
          {compareSelected.map((idx) => (
            <button
              key={`chip-${idx}`}
              type="button"
              className="feature-chip"
              onClick={() => handleFeatureAction(idx)}
            >
              <span
                className="legend-swatch"
                style={{ background: activeFeatureColorMap[idx] || '#d1d5db' }}
              />
              <span>{col_labels[idx]}</span>
            </button>
          ))}
        </div>
      )}

      {message && <div className="hint" style={{ color: '#b45309', marginTop: 0 }}>{message}</div>}

      <div className="feature-summary">
        {mode === 'default' && defaultFeatureIndex !== null && defaultFeatureIndex >= 0 && defaultFeatureIndex < col_labels.length
          ? `Active feature: ${col_labels[defaultFeatureIndex]}`
          : null}
        {isCompareMode ? `Selected: ${compareSelected.length}/${compareCap}` : null}
        {isOverviewMode ? 'All features selected.' : null}
      </div>

      <div className="legend-list">
        {filtered.map((idx) => {
          const selected = isSelected(idx)
          const disabled = isCompareMode && !selected && compareSelected.length >= compareCap
          const controlType = isCompareMode ? 'checkbox' : 'radio'
          return (
            <button
              key={idx}
              type="button"
              className={`feature-row${selected ? ' active' : ''}`}
              onClick={() => handleFeatureAction(idx)}
              disabled={disabled}
            >
              <span className="feature-row-main">
                {isOverviewMode ? (
                  <span className="feature-open-indicator">Open</span>
                ) : (
                  <input
                    type={controlType}
                    readOnly
                    checked={selected}
                    disabled={disabled}
                    tabIndex={-1}
                  />
                )}
                <span
                  className="legend-swatch"
                  style={{ background: activeFeatureColorMap[idx] || '#f3f4f6' }}
                />
                <span className="feature-label" title={col_labels[idx]}>
                  {col_labels[idx]}
                </span>
              </span>
              <span className="feature-row-action">{rowActionLabel(idx)}</span>
            </button>
          )
        })}
        {filtered.length === 0 && (
          <div className="hint">No features match “{query}”.</div>
        )}
      </div>
    </div>
  )
}
