import React, { useEffect, useMemo, useState } from 'react'

const MODE_META = {
  default: { label: 'Single', hint: 'Single feature, single hue.' },
  aggregate: { label: 'Aggregate', hint: 'Select any number of features. Trails and particles follow the summed field.' },
  compare: { label: 'Compare', hint: 'Select up to 9 features. Each trail and particle stays on one feature.' },
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
  compareCap = 9,
  message = '',
  activeFeatureColorMap = {},
  featureColorOverrides = {},
  featureColorOptions = [],
  onSetFeatureColor,
  labelColorMap = {},
}) {
  const { col_labels = [] } = payload || {}
  const [query, setQuery] = useState('')
  const isOverviewMode = mode === 'overview'
  const isAggregateMode = mode === 'aggregate'
  const isCompareMode = mode === 'compare'
  const isMultiFeatureMode = isAggregateMode || isCompareMode
  const defaultTabActive = mode === 'default'
  const aggregateTabActive = isAggregateMode || isOverviewMode

  useEffect(() => {
    setQuery('')
  }, [payload?.datasetId])

  const compareSet = useMemo(() => new Set(Array.isArray(compareFeatureIndices) ? compareFeatureIndices : []), [compareFeatureIndices])

  const alphabeticalIndices = useMemo(() => {
    const n = Array.isArray(col_labels) ? col_labels.length : 0
    const ordered = [...Array(n).keys()]
    ordered.sort((a, b) => String(col_labels[a] || '').localeCompare(String(col_labels[b] || '')))
    return ordered
  }, [col_labels])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return alphabeticalIndices
    return alphabeticalIndices.filter((idx) => String(col_labels[idx] || '').toLowerCase().includes(q))
  }, [alphabeticalIndices, query, col_labels])

  const compareSelected = useMemo(() => {
    return (Array.isArray(compareFeatureIndices) ? compareFeatureIndices : [])
      .filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < col_labels.length)
  }, [compareFeatureIndices, col_labels.length])
  const compareSelectedAlphabetical = useMemo(() => {
    return [...compareSelected].sort((a, b) => String(col_labels[a] || '').localeCompare(String(col_labels[b] || '')))
  }, [compareSelected, col_labels])
  const compareLimitReached = isCompareMode && compareSelected.length >= compareCap

  function handleMode(nextMode) {
    if (typeof onChangeMode === 'function') onChangeMode(nextMode)
  }

  function handleFeatureAction(idx) {
    if (isMultiFeatureMode) {
      if (typeof onToggleCompareFeature === 'function') onToggleCompareFeature(idx)
      return
    }
    if (typeof onSelectFeature === 'function') onSelectFeature(idx)
  }

  function isSelected(idx) {
    if (mode === 'default') return idx === defaultFeatureIndex
    if (isMultiFeatureMode) return compareSet.has(idx)
    return false
  }

  function rowActionLabel(idx) {
    if (isMultiFeatureMode) return compareSet.has(idx) ? 'Selected' : 'Add'
    if (isOverviewMode) return 'Open'
    return 'Select'
  }

  function handleFeatureColorChange(idx, event) {
    event.stopPropagation()
    if (typeof onSetFeatureColor === 'function') {
      onSetFeatureColor(idx, event.target.value)
    }
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
          className={`mode-tab${aggregateTabActive ? ' active' : ''}`}
          onClick={() => handleMode('aggregate')}
        >
          {MODE_META.aggregate.label}
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
          : aggregateTabActive
            ? (isOverviewMode
                ? 'All features aggregated in grayscale. Choose a feature to return to single-feature view.'
                : MODE_META.aggregate.hint)
          : isOverviewMode
            ? 'All features aggregated in grayscale. Choose a feature to return to single-feature view.'
            : MODE_META.default.hint}
      </p>

      <div className="feature-toolbar">
        {aggregateTabActive && (
          <button
            type="button"
            className="btn"
            onClick={() => typeof onSelectAll === 'function' && onSelectAll()}
            disabled={isOverviewMode}
          >
            {isOverviewMode ? 'Showing All' : 'Select All'}
          </button>
        )}
        {isAggregateMode && (
          <button
            type="button"
            className="btn"
            onClick={() => typeof onClearCompare === 'function' && onClearCompare()}
            disabled={compareSelected.length === 0}
          >
            Clear
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

      {isMultiFeatureMode && compareSelected.length > 0 && (
        <div className="feature-chip-list">
          {compareSelectedAlphabetical.map((idx) => (
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
        {isAggregateMode ? `Selected: ${compareSelected.length}` : null}
        {isCompareMode ? `Selected: ${compareSelected.length}/${compareCap}` : null}
        {isOverviewMode ? 'All features selected.' : null}
      </div>

      <div className="legend-list">
        {filtered.map((idx) => {
          const selected = isSelected(idx)
          const disabled = isCompareMode && !selected && compareLimitReached
          const controlType = isMultiFeatureMode ? 'checkbox' : 'radio'
          const featureColorValue = typeof featureColorOverrides[idx] === 'string'
            ? featureColorOverrides[idx]
            : (featureColorOptions[0]?.value || '')
          return (
            <div key={idx} className="feature-row-shell">
              <button
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
              <label className="feature-color-control" title={`Color for ${col_labels[idx]}`}>
                <span className="feature-color-control-label">Color</span>
                <select
                  className="feature-color-select"
                  value={featureColorValue}
                  onChange={(event) => handleFeatureColorChange(idx, event)}
                  onClick={(event) => event.stopPropagation()}
                  disabled={!featureColorOptions.length}
                >
                  {featureColorOptions.map((option) => (
                    <option key={`${idx}-${option.value}`} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )
        })}
        {filtered.length === 0 && (
          <div className="hint">No features match “{query}”.</div>
        )}
      </div>
    </div>
  )
}
