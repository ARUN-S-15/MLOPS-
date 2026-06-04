const BASE = "http://localhost:5000"

export const api = {

  async detectEmotion(blob) {
    const fd = new FormData()
    fd.append("image", blob, "frame.jpg")
    const res = await fetch(`${BASE}/predict`, { method: "POST", body: fd })
    if (!res.ok) throw new Error(`predict failed: ${res.status}`)
    return res.json()
  },

  async startInterview(resumeFile) {
    const fd = new FormData()
    if (resumeFile) fd.append("resume", resumeFile)
    const res = await fetch(`${BASE}/interview/start`, { method: "POST", body: fd })
    if (!res.ok) throw new Error(`start interview failed: ${res.status}`)
    return res.json()
  },

  async scoreAnswer(payload) {
    const res = await fetch(`${BASE}/interview/score-answer`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    })
    if (!res.ok) throw new Error(`score answer failed: ${res.status}`)
    return res.json()
  },

  async analyze(videoFile, resumeFile) {
    const fd = new FormData()
    fd.append("video", videoFile)
    if (resumeFile) fd.append("resume", resumeFile)
    const res = await fetch(`${BASE}/analyze`, { method: "POST", body: fd })
    if (!res.ok) throw new Error(`analyze failed: ${res.status}`)
    return res.json()
  },

  async health() {
    const res = await fetch(`${BASE}/health`)
    if (!res.ok) throw new Error("health check failed")
    return res.json()
  },

  // ──────────────────── RAG ENDPOINTS ─────────────────────
  
  async ragStatus() {
    const res = await fetch(`${BASE}/rag/status`)
    if (!res.ok) throw new Error("RAG status check failed")
    return res.json()
  },

  async uploadDocument(file, docType) {
    const fd = new FormData()
    fd.append("document", file)
    fd.append("doc_type", docType)
    const res = await fetch(`${BASE}/rag/upload-document`, { method: "POST", body: fd })
    if (!res.ok) throw new Error(`document upload failed: ${res.status}`)
    return res.json()
  },

  async analyzeWithContext(query, docTypeFilter) {
    const res = await fetch(`${BASE}/rag/analyze-with-context`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        doc_type_filter: docTypeFilter,
      }),
    })
    if (!res.ok) throw new Error(`RAG analysis failed: ${res.status}`)
    return res.json()
  },

  async queryDocuments(query, docTypeFilter, topK = 5) {
    const res = await fetch(`${BASE}/rag/query-documents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        doc_type_filter: docTypeFilter,
        top_k: topK,
      }),
    })
    if (!res.ok) throw new Error(`query failed: ${res.status}`)
    return res.json()
  },

  async fullInterviewAnalysis(analysisData) {
    const res = await fetch(`${BASE}/rag/full-interview-analysis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(analysisData),
    })
    if (!res.ok) throw new Error(`full analysis failed: ${res.status}`)
    return res.json()
  },
}