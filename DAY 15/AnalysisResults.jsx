import { useState } from "react"

export default function AnalysisResults({ analysis, loading = false }) {
  const [expandedSection, setExpandedSection] = useState("summary")

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Analyzing interview with RAG-enhanced agents...</p>
        </div>
      </div>
    )
  }

  if (!analysis) {
    return null
  }

  const renderJsonSection = (title, data, color = "cyan") => (
    <div className={`border-l-4 border-${color}-500 p-4 bg-${color}-900/10 rounded mb-4`}>
      <h4 className={`font-semibold text-${color}-400 mb-2`}>{title}</h4>
      <pre className="text-xs text-gray-300 overflow-auto max-h-64 bg-black/30 p-3 rounded">
        {typeof data === 'string' ? data : JSON.stringify(data, null, 2)}
      </pre>
    </div>
  )

  const sections = {
    summary: {
      title: "📋 Interview Summary",
      icon: "summary",
      color: "purple"
    },
    resume: {
      title: "📄 Resume Analysis",
      data: analysis?.resume_analysis,
      color: "blue"
    },
    job: {
      title: "💼 Job Description Analysis",
      data: analysis?.job_analysis,
      color: "green"
    },
    skills: {
      title: "🔍 Skill Gap Analysis",
      data: analysis?.skill_gap_analysis,
      color: "orange"
    },
    communication: {
      title: "🎤 Communication Assessment",
      data: analysis?.communication_analysis,
      color: "cyan"
    },
    technical: {
      title: "🛠️ Technical Competency",
      data: analysis?.technical_analysis,
      color: "yellow"
    },
    behavioral: {
      title: "👥 Behavioral Assessment",
      data: analysis?.behavioral_analysis,
      color: "pink"
    },
    feedback: {
      title: "📊 Final Feedback & Recommendations",
      data: analysis?.final_feedback,
      color: "red"
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">🎯 Comprehensive Interview Analysis</h2>
        <p className="text-gray-400">AI-powered analysis powered by Groq LLM and Pinecone RAG system</p>
      </div>

      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-2 pb-4 border-b border-zinc-800 overflow-x-auto">
        {Object.entries(sections).map(([key, section]) => (
          <button
            key={key}
            onClick={() => setExpandedSection(key)}
            className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition ${
              expandedSection === key
                ? 'bg-cyan-600 text-white'
                : 'bg-zinc-800 text-gray-400 hover:bg-zinc-700'
            }`}
          >
            {section.title.split(" ")[0]} {section.title.split(" ")[1]}
          </button>
        ))}
      </div>

      {/* Content Area */}
      <div className="space-y-4">
        {expandedSection === "summary" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Overall Score Card */}
            {analysis?.final_feedback?.analysis?.['Overall Assessment'] && (
              <div className="p-6 bg-gradient-to-br from-purple-900/20 to-pink-900/20 border border-purple-500/30 rounded-lg">
                <h4 className="text-lg font-semibold text-purple-400 mb-2">Overall Assessment</h4>
                <p className="text-3xl font-bold text-white mb-2">
                  {analysis.final_feedback.analysis['Overall Assessment']}
                </p>
                <div className="flex gap-2 flex-wrap">
                  <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-sm">
                    Interview Ready
                  </span>
                </div>
              </div>
            )}

            {/* Key Strengths */}
            {analysis?.final_feedback?.analysis?.['Key Strengths'] && (
              <div className="p-6 bg-gradient-to-br from-green-900/20 to-cyan-900/20 border border-green-500/30 rounded-lg">
                <h4 className="text-lg font-semibold text-green-400 mb-3">✓ Key Strengths</h4>
                <ul className="space-y-2">
                  {(Array.isArray(analysis.final_feedback.analysis['Key Strengths'])
                    ? analysis.final_feedback.analysis['Key Strengths']
                    : [analysis.final_feedback.analysis['Key Strengths']]
                  ).map((strength, i) => (
                    <li key={i} className="text-green-300 text-sm flex items-start">
                      <span className="mr-2">→</span>
                      {typeof strength === 'string' ? strength : JSON.stringify(strength)}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Areas for Improvement */}
            {analysis?.final_feedback?.analysis?.['Critical Improvement Areas'] && (
              <div className="p-6 bg-gradient-to-br from-orange-900/20 to-red-900/20 border border-orange-500/30 rounded-lg md:col-span-2">
                <h4 className="text-lg font-semibold text-orange-400 mb-3">🚀 Areas for Improvement</h4>
                <ul className="space-y-2 grid grid-cols-1 md:grid-cols-2 gap-4">
                  {(Array.isArray(analysis.final_feedback.analysis['Critical Improvement Areas'])
                    ? analysis.final_feedback.analysis['Critical Improvement Areas']
                    : [analysis.final_feedback.analysis['Critical Improvement Areas']]
                  ).map((area, i) => (
                    <li key={i} className="text-orange-300 text-sm flex items-start">
                      <span className="mr-2">→</span>
                      {typeof area === 'string' ? area : JSON.stringify(area)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Individual Section Analysis */}
        {expandedSection !== "summary" && sections[expandedSection]?.data && (
          <div>
            {renderJsonSection(
              sections[expandedSection].title,
              sections[expandedSection].data,
              sections[expandedSection].color
            )}
          </div>
        )}

        {/* Action Plan */}
        {expandedSection === "feedback" && analysis?.final_feedback?.analysis?.['Personalized Action Plan'] && (
          <div className="mt-6 space-y-4">
            <h4 className="text-lg font-semibold text-red-400">📈 Personalized Action Plan</h4>
            {Object.entries(analysis.final_feedback.analysis['Personalized Action Plan']).map(([period, actions]) => (
              <div key={period} className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                <h5 className="font-semibold text-white mb-2 capitalize">{period}</h5>
                <ul className="space-y-1">
                  {(Array.isArray(actions) ? actions : [actions]).map((action, i) => (
                    <li key={i} className="text-gray-300 text-sm flex items-start">
                      <span className="mr-2">•</span>
                      {typeof action === 'string' ? action : JSON.stringify(action)}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}

        {/* Confidence Score */}
        {analysis?.final_feedback?.analysis?.['Hiring Recommendation with Confidence'] && (
          <div className="p-4 bg-gradient-to-r from-cyan-900/20 to-purple-900/20 border border-cyan-500/30 rounded-lg mt-4">
            <h4 className="font-semibold text-cyan-400 mb-2">🎯 Recommendation Confidence</h4>
            <p className="text-white text-lg">
              {analysis.final_feedback.analysis['Hiring Recommendation with Confidence']}%
            </p>
            <div className="w-full bg-zinc-800 rounded-full h-2 mt-2">
              <div
                className="bg-cyan-500 h-2 rounded-full"
                style={{
                  width: `${Math.min(
                    100,
                    parseInt(analysis.final_feedback.analysis['Hiring Recommendation with Confidence']) || 0
                  )}%`
                }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Raw Data Export */}
      <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
        <details className="cursor-pointer">
          <summary className="font-semibold text-gray-400 hover:text-white">
            📥 Raw Analysis Data (for export/debugging)
          </summary>
          <pre className="mt-3 text-xs text-gray-400 overflow-auto max-h-96 bg-black/50 p-4 rounded">
            {JSON.stringify(analysis, null, 2)}
          </pre>
        </details>
      </div>
    </div>
  )
}
