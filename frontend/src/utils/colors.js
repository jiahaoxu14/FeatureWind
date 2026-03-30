// Feature rendering in the UI defaults to black.
// A separate dark palette is reserved for currently visualized features only,
// so feature hues stay clearly distinct from the light label colors used for points.
export const DEFAULT_FEATURE_HUE = '#000000'
export const FEATURE_PALETTE = [
  '#0f766e', // deep teal
  '#c2410c', // burnt orange
  '#6d28d9', // deep violet
  '#b91c1c', // crimson
  '#166534', // forest green
  '#be185d', // magenta
  '#854d0e', // bronze
  '#1f2937', // charcoal
  '#9f1239', // wine
]

// Label palette — used exclusively for data-point markers.
// Keep these noticeably lighter than the trail palette so points read as a separate layer
// even when a hue family is nearby.
export const LABEL_PALETTE = [
  '#93c5fd',  // light blue
  '#fde68a',  // pale amber
  '#a7f3d0',  // mint
  '#fbcfe8',  // soft pink
  '#bfdbfe',  // powder blue
  '#fef08a',  // light yellow
  '#ddd6fe',  // lavender
  '#bae6fd',  // ice blue
  '#d9f99d',  // soft lime
  '#fecdd3',  // pale rose
]

/**
 * Build a stable { stringLabel → hexColor } map from an array of point labels.
 * Labels are sorted alphabetically so colors are assigned in a deterministic order.
 */
export function buildLabelColorMap(pointLabels) {
  if (!Array.isArray(pointLabels) || pointLabels.length === 0) return {}
  const unique = Array.from(new Set(pointLabels.map(String)))
  unique.sort()
  const map = {}
  unique.forEach((label, i) => {
    map[label] = LABEL_PALETTE[i % LABEL_PALETTE.length]
  })
  return map
}

/**
 * Build a stable { featureIndex -> hexColor } map for feature rendering in the UI.
 * Feature-family coloring is disabled; non-visualized features stay black by default.
 */
export function buildFeatureColorMap(featureCount, familyAssignments) {
  const out = {}
  const count = Number.isInteger(featureCount) ? featureCount : 0
  if (count <= 0) return out

  for (let idx = 0; idx < count; idx++) {
    out[idx] = DEFAULT_FEATURE_HUE
  }
  return out
}
