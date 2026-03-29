// Feature palette — used exclusively for trails, particles, vane vectors, and feature swatches.
// These are dark, saturated hues suited for motion lines on the canvas.
export const FEATURE_PALETTE = ['#0f766e', '#c2410c', '#7c3aed', '#dc2626']
export const DEFAULT_FEATURE_HUE = '#0f766e'

// Label palette — used exclusively for data-point markers.
// Colors are chosen from hue families intentionally disjoint from the feature palette
// (amber, sky-blue, green, pink, yellow, lime, light-orange, fuchsia, cyan, rose).
export const LABEL_PALETTE = [
  '#f59e0b',  // amber        (~38°)
  '#38bdf8',  // sky blue     (~199°)
  '#4ade80',  // green        (~142°)
  '#f472b6',  // pink         (~328°)
  '#facc15',  // yellow       (~48°)
  '#a3e635',  // lime         (~82°)
  '#fb923c',  // light orange (~27°, visually distinct from dark #c2410c)
  '#e879f9',  // fuchsia      (~293°)
  '#67e8f9',  // cyan         (~186°)
  '#fda4af',  // rose         (~351°, much lighter than dark #dc2626)
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
