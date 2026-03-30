// Feature palette — used exclusively for trails, particles, vane vectors, and feature swatches.
// Keep these dark and saturated so they stay far away from the light label palette used for points.
export const FEATURE_PALETTE = [
  '#0f766e', // deep teal
  '#c2410c', // burnt orange
  '#7c3aed', // violet
  '#b91c1c', // deep red
  '#0369a1', // deep cyan-blue
  '#be185d', // magenta
  '#854d0e', // dark brown
  '#1f2937', // charcoal
]
export const DEFAULT_FEATURE_HUE = '#0f766e'

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
 * Prefer family assignments so same-family features share a dark UI hue, independent
 * of any lighter backend palette that may be used elsewhere.
 */
export function buildFeatureColorMap(featureCount, familyAssignments) {
  const out = {}
  const count = Number.isInteger(featureCount) ? featureCount : 0
  if (count <= 0) return out

  const hasFamilies = Array.isArray(familyAssignments) && familyAssignments.length === count
  if (hasFamilies) {
    const normalizedFamilies = familyAssignments.map((raw, idx) => {
      const familyId = Number(raw)
      return Number.isInteger(familyId) ? familyId : idx
    })
    const uniqueFamilies = Array.from(new Set(normalizedFamilies)).sort((a, b) => a - b)
    const familyColorMap = {}
    uniqueFamilies.forEach((familyId, idx) => {
      familyColorMap[familyId] = FEATURE_PALETTE[idx % FEATURE_PALETTE.length]
    })
    normalizedFamilies.forEach((familyId, idx) => {
      out[idx] = familyColorMap[familyId]
    })
    return out
  }

  for (let idx = 0; idx < count; idx++) {
    out[idx] = FEATURE_PALETTE[idx % FEATURE_PALETTE.length]
  }
  return out
}
