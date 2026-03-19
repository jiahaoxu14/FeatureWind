export async function uploadFile(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/api/upload', {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`)
  return res.json()
}

export async function uploadTmapJson(tmap, filename = 'session.tmap') {
  const res = await fetch('/api/upload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tmap, filename }),
  })
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`)
  return res.json()
}

export async function compute(payload) {
  const res = await fetch('/api/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(`Compute failed: ${res.status}`)
  return res.json()
}

export async function validateAnalysis(payload) {
  const res = await fetch('/api/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    let detail = `Validate failed: ${res.status}`
    try {
      const body = await res.json()
      if (body?.error) detail = body.error
    } catch {}
    throw new Error(detail)
  }
  return res.json()
}

export async function recolor(datasetId, familyAssignments) {
  const res = await fetch('/api/colors', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, familyAssignments }),
  })
  if (!res.ok) throw new Error(`Recolor failed: ${res.status}`)
  return res.json()
}

export function downloadJsonFile(payload, filename) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
