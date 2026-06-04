import { useState } from "react"
import { api } from "../services/api.js"

export default function DocumentUploadManager() {
  const [documents, setDocuments] = useState({
    resume: null,
    job_description: null,
    transcript: null,
    previous_reports: null,
  })
  
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState({})
  const [error, setError] = useState(null)

  const docTypes = [
    { id: "resume", label: "📄 Resume/CV", description: "Your professional resume" },
    { id: "job_description", label: "💼 Job Description", description: "Target position details" },
    { id: "transcript", label: "📝 Interview Transcript", description: "Interview conversation" },
    { id: "previous_reports", label: "📊 Previous Reports", description: "Past interview reports" },
  ]

  const handleFileSelect = (e, docType) => {
    const file = e.target.files[0]
    if (file) {
      setDocuments(prev => ({ ...prev, [docType]: file }))
      setError(null)
    }
  }

  const handleUpload = async (docType) => {
    const file = documents[docType]
    if (!file) {
      setError(`Please select a file for ${docType}`)
      return
    }

    setUploading(true)
    try {
      const result = await api.uploadDocument(file, docType)
      setUploadStatus(prev => ({
        ...prev,
        [docType]: { success: true, message: result.message, chunks: result.document.chunks }
      }))
      // Reset file after successful upload
      setDocuments(prev => ({ ...prev, [docType]: null }))
    } catch (err) {
      setUploadStatus(prev => ({
        ...prev,
        [docType]: { success: false, message: err.message }
      }))
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const handleUploadAll = async () => {
    const filesToUpload = Object.entries(documents).filter(([_, file]) => file !== null)
    
    if (filesToUpload.length === 0) {
      setError("Please select at least one document to upload")
      return
    }

    setUploading(true)
    try {
      for (const [docType, file] of filesToUpload) {
        await api.uploadDocument(file, docType)
        setUploadStatus(prev => ({
          ...prev,
          [docType]: { success: true, message: "Uploaded successfully", chunks: "..." }
        }))
      }
      setDocuments({
        resume: null,
        job_description: null,
        transcript: null,
        previous_reports: null,
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">📚 Document Management</h2>
        <p className="text-gray-400">Upload your documents for intelligent analysis. The RAG system will use these to provide personalized feedback.</p>
      </div>

      {error && (
        <div className="p-4 bg-red-900/20 border border-red-500 rounded-lg text-red-300">
          ⚠️ {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {docTypes.map(docType => (
          <div
            key={docType.id}
            className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg hover:border-cyan-500/50 transition"
          >
            <div className="mb-3">
              <h3 className="font-semibold text-white mb-1">{docType.label}</h3>
              <p className="text-sm text-gray-400">{docType.description}</p>
            </div>

            <div className="space-y-3">
              {/* File Input */}
              <label className="block">
                <div className="cursor-pointer px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded border border-dashed border-cyan-500/50 text-center transition">
                  <span className="text-gray-300 text-sm">
                    {documents[docType.id]
                      ? `✓ ${documents[docType.id].name}`
                      : "Choose file..."}
                  </span>
                </div>
                <input
                  type="file"
                  className="hidden"
                  onChange={(e) => handleFileSelect(e, docType.id)}
                  accept=".pdf,.txt,.docx,.json,.csv"
                  disabled={uploading}
                />
              </label>

              {/* Upload Status */}
              {uploadStatus[docType.id] && (
                <div className={`text-xs py-2 px-3 rounded ${
                  uploadStatus[docType.id].success
                    ? 'bg-green-900/20 text-green-300 border border-green-500/30'
                    : 'bg-red-900/20 text-red-300 border border-red-500/30'
                }`}>
                  {uploadStatus[docType.id].success ? '✓' : '✗'} {uploadStatus[docType.id].message}
                  {uploadStatus[docType.id].chunks && ` (${uploadStatus[docType.id].chunks} chunks)`}
                </div>
              )}

              {/* Upload Button */}
              <button
                onClick={() => handleUpload(docType.id)}
                disabled={uploading || !documents[docType.id]}
                className="w-full px-3 py-2 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 rounded text-sm text-white transition"
              >
                {uploading ? "⏳ Uploading..." : "Upload"}
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Upload All Button */}
      <div className="flex gap-3">
        <button
          onClick={handleUploadAll}
          disabled={uploading || Object.values(documents).every(d => d === null)}
          className="flex-1 px-4 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-lg text-white font-semibold transition"
        >
          {uploading ? "⏳ Uploading All..." : "🚀 Upload All Documents"}
        </button>
      </div>

      {/* Upload Summary */}
      {Object.values(uploadStatus).length > 0 && (
        <div className="p-4 bg-green-900/10 border border-green-500/30 rounded-lg">
          <h4 className="font-semibold text-green-400 mb-2">✓ Upload Summary</h4>
          <div className="space-y-1 text-sm text-gray-300">
            {Object.entries(uploadStatus).map(([docType, status]) => (
              status && (
                <div key={docType} className="flex items-center">
                  <span className={status.success ? "text-green-400" : "text-red-400"}>
                    {status.success ? "✓" : "✗"}
                  </span>
                  <span className="ml-2 capitalize">{docType.replace(/_/g, " ")}</span>
                </div>
              )
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
