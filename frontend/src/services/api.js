async function readErrorMessage(res, fallback) {
  try {
    const contentType = res.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
      const payload = await res.json()
      if (payload && typeof payload.error === 'string' && payload.error.trim()) return payload.error
    } else {
      const text = await res.text()
      if (text && text.trim()) return text
    }
  } catch {
    // Fall through to the fallback message.
  }
  return fallback
}

export async function uploadFile(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/api/upload', {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const msg = await readErrorMessage(res, `Upload failed: ${res.status}`)
    throw new Error(msg)
  }
  return res.json()
}

export async function compute(payload) {
  const res = await fetch('/api/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const msg = await readErrorMessage(res, `Compute failed: ${res.status}`)
    throw new Error(msg)
  }
  return res.json()
}

export async function exportStaticTrailFigures(payload) {
  const res = await fetch('/api/export-static-trail-figures', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const msg = await readErrorMessage(res, `Static trail export failed: ${res.status}`)
    throw new Error(msg)
  }
  return res.json()
}
