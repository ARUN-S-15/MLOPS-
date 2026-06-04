import { useState, useEffect } from "react"
import { api } from "../services/api.js"
import DocumentUploadManager from "./DocumentUploadManager.jsx"
import AnalysisResults from "./AnalysisResults.jsx"

export default function RAGDashboard() {
  const [activeTab, setActiveTab] = useState("upload") // upload, query, analysis
  const [ragStatus, setRagStatus] = useState(null)
  const [ragLoading, setRagLoading] = useState(true)
  const [ragError, setRagError] = useState(null)

  // Query tab state
  const [queryText, setQueryText] = useState("")
  const [queryResults, setQueryResults] = useState(null)
  const [querying, setQuerying] = useState(false)
  const [selectedDocType, setSelectedDocType] = useState("")

  // Full analysis state
  const [analysisData, setAnalysisData] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisResults, setAnalysisResults] = useState(null)

  // Check RAG status on mount
  useEffect(() => {
    const checkRAGStatus = async () => {
      try {
        const status = await api.ragStatus()
        setRagStatus(status)
        setRagLoading(false)
      } catch (err) {
        setRagError(err.message)
        setRagLoading(false)
      }
    }

    checkRAGStatus()
  }, [])

  const handleQueryDocuments = async () => {
    if (!queryText.trim()) {
      setRagError("Please enter a search query")
      return
    }

    setQuerying(true)
    setRagError(null)
    try {
      const results = await api.queryDocuments(queryText, selectedDocType || null)
      setQueryResults(results)
    } catch (err) {
      setRagError(err.message)
    } finally {
      setQuerying(false)
    }
  }

  const handleFullAnalysis = async () => {
    // For now, show a prompt to gather interview data
    setAnalyzing(true)
    try {
      // This would be populated with actual interview data
      const analysisPayload = {
        resume_content: "Sample resume content",
        job_description_content: "Sample JD",
        interview_transcript: "Interview transcript",
        interview_answers: [],
        emotion_data: {},
        speech_data: {},
      }

      const results = await api.fullInterviewAnalysis(analysisPayload)
      setAnalysisResults(results)
    } catch (err) {
      setRagError(err.message)
    } finally {
      setAnalyzing(false)
    }
  }

  if (ragLoading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-cyan-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Initializing RAG Dashboard...</p>
        </div>
      </div>
    )
  }

  if (ragError && !ragStatus) {
    return (
      <div className="min-h-screen bg-black text-white p-8">
        <div className="max-w-2xl mx-auto">
          <div className="p-6 bg-red-900/20 border border-red-500 rounded-lg">
            <h2 className="text-2xl font-bold text-red-400 mb-2">⚠️ RAG System Error</h2>
            <p className="text-red-300 mb-4">{ragError}</p>
            <p className="text-gray-400 text-sm">
              Please ensure you have:
            </p>
            <ul className="text-gray-300 text-sm space-y-1 mt-2 ml-4">
              <li>✓ Set GROQ_API_KEY environment variable</li>
              <li>✓ Set PINECONE_API_KEY environment variable</li>
              <li>✓ Installed required Python packages: pip install -r requirements.txt</li>
            </ul>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-900/20 to-purple-900/20 border-b border-cyan-500/20 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              🤖 RAG-Enhanced Interview Analyzer
            </h1>
            {ragStatus?.status === "ready" && (
              <div className="flex items-center gap-2 px-3 py-1 bg-green-900/30 border border-green-500/30 rounded-full">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-green-400 text-sm font-medium">RAG Ready</span>
              </div>
            )}
          </div>

          {/* Feature description */}
          <p className="text-gray-400 text-sm mb-4">
            Powered by <span className="text-cyan-400 font-semibold">Groq LLM</span> • 
            <span className="text-purple-400 font-semibold ml-1">Pinecone Vector DB</span> • 
            <span className="text-green-400 font-semibold ml-1">Autonomous Agents</span>
          </p>

          {/* Tabs */}
          <div className="flex gap-2 flex-wrap">
            {[
              { id: "upload", label: "📚 Upload Documents", icon: "upload" },
              { id: "query", label: "🔍 Query Documents", icon: "search" },
              { id: "analysis", label: "🎯 Full Analysis", icon: "analysis" },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg font-medium transition ${
                  activeTab === tab.id
                    ? 'bg-cyan-600 text-white'
                    : 'bg-zinc-800 text-gray-400 hover:bg-zinc-700'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Error Display */}
        {ragError && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-500 rounded-lg text-red-300">
            ⚠️ {ragError}
          </div>
        )}

        {/* Upload Tab */}
        {activeTab === "upload" && (
          <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-8">
            <DocumentUploadManager />
          </div>
        )}

        {/* Query Tab */}
        {activeTab === "query" && (
          <div className="space-y-6">
            <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-8">
              <h2 className="text-2xl font-bold text-white mb-4">🔍 Query Your Documents</h2>
              <p className="text-gray-400 mb-6">Ask questions about your uploaded documents. The RAG system will retrieve relevant information.</p>

              <div className="space-y-4">
                {/* Query Input */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Search Query</label>
                  <textarea
                    value={queryText}
                    onChange={(e) => setQueryText(e.target.value)}
                    placeholder="Ask a question or search for information..."
                    className="w-full px-4 py-3 bg-zinc-800 border border-zinc-700 rounded-lg text-white placeholder-gray-500 focus:border-cyan-500 focus:outline-none"
                    rows={3}
                  />
                </div>

                {/* Document Type Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Filter by Document Type (Optional)</label>
                  <select
                    value={selectedDocType}
                    onChange={(e) => setSelectedDocType(e.target.value)}
                    className="w-full px-4 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
                  >
                    <option value="">All Documents</option>
                    <option value="resume">Resume</option>
                    <option value="job_description">Job Description</option>
                    <option value="transcript">Interview Transcript</option>
                    <option value="previous_reports">Previous Reports</option>
                  </select>
                </div>

                {/* Search Button */}
                <button
                  onClick={handleQueryDocuments}
                  disabled={querying || !queryText.trim()}
                  className="w-full px-4 py-3 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 rounded-lg text-white font-semibold transition"
                >
                  {querying ? "🔄 Searching..." : "🚀 Search Documents"}
                </button>
              </div>
            </div>

            {/* Query Results */}
            {queryResults && (
              <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-8">
                <h3 className="text-xl font-bold text-white mb-4">
                  📋 Results ({queryResults.total_results})
                </h3>
                <div className="space-y-4">
                  {queryResults.results.map((result, i) => (
                    <div key={i} className="p-4 bg-zinc-800/50 border border-zinc-700 rounded-lg hover:border-cyan-500/50 transition">
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h4 className="font-semibold text-cyan-400">{result.source}</h4>
                          <span className="text-xs px-2 py-1 bg-purple-500/20 text-purple-300 rounded capitalize">
                            {result.doc_type}
                          </span>
                        </div>
                        <span className="text-sm text-gray-400">Score: {(result.score * 100).toFixed(1)}%</span>
                      </div>
                      <p className="text-gray-300 text-sm line-clamp-4">{result.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Analysis Tab */}
        {activeTab === "analysis" && (
          <div className="space-y-6">
            {!analysisResults ? (
              <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-8">
                <h2 className="text-2xl font-bold text-white mb-4">🎯 Full Interview Analysis</h2>
                <p className="text-gray-400 mb-6">
                  Run a comprehensive analysis using all AI agents. This will analyze your resume, job description, interview transcript, communication patterns, technical competency, and behavioral assessment.
                </p>

                <button
                  onClick={handleFullAnalysis}
                  disabled={analyzing}
                  className="px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-lg text-white font-semibold text-lg transition"
                >
                  {analyzing ? "⏳ Analyzing..." : "🚀 Start Full Analysis"}
                </button>
              </div>
            ) : (
              <AnalysisResults analysis={analysisResults.analysis} loading={analyzing} />
            )}
          </div>
        )}
      </div>
    </div>
  )
}
