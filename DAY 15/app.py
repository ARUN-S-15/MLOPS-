from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import tempfile
import os
import cv2
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── CORS: allow React frontend on port 5173 (Vite) and 3000 ──────────────────
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]}})

# ── RAG System Initialization ─────────────────────────────────────────────────
rag_retriever = None
rag_llm = None
agent_orchestrator = None
embedding_manager = None
vector_store = None
document_manager = None

def initialize_rag():
    """Initialize RAG components if credentials are available"""
    global rag_retriever, rag_llm, agent_orchestrator, embedding_manager, vector_store, document_manager
    
    try:
        from rag_config import RAGConfig
        from rag_engine import EmbeddingManager, PineconeVectorStore, RAGRetriever, GroqLLMInference
        from interview_agents import InterviewAgentOrchestrator
        from document_processor import DocumentManager
        
        # Check if credentials are configured
        creds = RAGConfig.validate_credentials()
        if not creds['valid']:
            logger.warning(f"RAG credentials not fully configured. Missing: {creds['missing_keys']}")
            return False
        
        logger.info("Initializing RAG system components...")
        
        # Initialize components
        embedding_manager = EmbeddingManager()
        vector_store = PineconeVectorStore()
        rag_retriever = RAGRetriever(embedding_manager, vector_store)
        rag_llm = GroqLLMInference()
        agent_orchestrator = InterviewAgentOrchestrator(rag_llm, rag_retriever)
        document_manager = DocumentManager()
        
        # Try to create index if it doesn't exist
        try:
            vector_store.create_index_if_not_exists()
        except Exception as e:
            logger.warning(f"Could not ensure index exists: {e}")
        
        logger.info("RAG system initialized successfully!")
        return True
        
    except ImportError as e:
        logger.warning(f"RAG dependencies not installed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        return False

# Try to initialize RAG on startup
rag_ready = initialize_rag()
logger.info(f"RAG System Ready: {rag_ready}")

# ── Load YOLO model once at startup ──────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "../model/best_emotion_model.pt"))
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded! Classes:", model.names)

# ── Face detector using MediaPipe (more reliable than Haar Cascade) ──────────
import mediapipe as mp
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.3
)

# ── Question bank mapped to resume keywords ───────────────────────────────────
QUESTION_BANK = {
    "python": [
        "Walk me through a Python project you built from scratch.",
        "How do you handle errors and exceptions in Python?",
        "Explain the difference between a list and a tuple in Python.",
    ],
    "machine learning": [
        "Explain the difference between supervised and unsupervised learning.",
        "How do you handle overfitting in a machine learning model?",
        "Walk me through how you would approach a new ML problem.",
    ],
    "deep learning": [
        "What is the vanishing gradient problem and how do you solve it?",
        "When would you choose CNN over RNN for a task?",
        "How do you decide on the architecture for a neural network?",
    ],
    "react": [
        "Explain the difference between state and props in React.",
        "How do you optimize performance in a React application?",
        "Describe how you would manage global state in React.",
    ],
    "docker": [
        "Explain the difference between a Docker image and a container.",
        "How would you use Docker in a production deployment?",
        "What is docker-compose and when would you use it?",
    ],
    "sql": [
        "Explain the difference between INNER JOIN and LEFT JOIN.",
        "How would you optimize a slow SQL query?",
        "What is database normalization and why is it important?",
    ],
    "aws": [
        "Which AWS services have you worked with and for what purpose?",
        "How would you design a scalable architecture on AWS?",
        "Explain the difference between EC2 and Lambda.",
    ],
    "javascript": [
        "Explain the event loop in JavaScript.",
        "What is the difference between var, let, and const?",
        "How does asynchronous programming work in JavaScript?",
    ],
    "flask": [
        "How do you handle authentication in a Flask application?",
        "Explain how Flask routing works.",
        "How would you structure a large Flask application?",
    ],
    "opencv": [
        "What computer vision tasks have you solved with OpenCV?",
        "Explain how you would detect objects in a video stream.",
        "How do you preprocess images before feeding them to a model?",
    ],
    "tensorflow": [
        "Explain the difference between eager and graph execution in TensorFlow.",
        "How do you save and load a TensorFlow model?",
        "What is a tensor and how is it different from a NumPy array?",
    ],
    "pytorch": [
        "Explain the difference between PyTorch and TensorFlow.",
        "What is autograd in PyTorch and how does it work?",
        "How do you define a custom dataset in PyTorch?",
    ],
    "git": [
        "Explain the difference between git merge and git rebase.",
        "How do you resolve a merge conflict in Git?",
        "What is your typical Git workflow on a team project?",
    ],
    "java": [
        "Explain the difference between an interface and an abstract class in Java.",
        "What is the Java garbage collector and how does it work?",
        "How does multithreading work in Java?",
    ],
    "data science": [
        "How do you handle missing data in a dataset?",
        "Explain the difference between correlation and causation.",
        "Walk me through your typical data analysis workflow.",
    ],
    "nlp": [
        "Explain the difference between stemming and lemmatization.",
        "What is TF-IDF and when would you use it?",
        "How does a transformer model work at a high level?",
    ],
    "yolo": [
        "Explain how the YOLO object detection algorithm works.",
        "What are the trade-offs between YOLO versions?",
        "How did you train and evaluate your YOLO model?",
    ],
    "api": [
        "Explain the difference between REST and GraphQL.",
        "How do you handle authentication in a REST API?",
        "What are best practices for API versioning?",
    ],
}

DEFAULT_QUESTIONS = [
    "Tell me about yourself and your professional background.",
    "What are your greatest strengths and how have they helped you succeed?",
    "Describe a challenging situation at work and how you resolved it.",
    "Where do you see yourself in 5 years professionally?",
    "Why are you interested in this position and our company?",
    "Tell me about a time you worked in a team under pressure.",
    "What is your biggest weakness and how are you working on it?",
]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "AI Interview Analyzer API",
        "rag_enabled": rag_ready
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": MODEL_PATH,
        "classes": model.names,
        "rag_enabled": rag_ready,
    })


# ────────────────────────────────────── RAG ENDPOINTS ────────────────────────────────────────

@app.route("/rag/status", methods=["GET"])
def rag_status():
    """Check RAG system status and configuration"""
    if not rag_ready:
        return jsonify({
            "status": "unavailable",
            "message": "RAG system not initialized. Check API credentials."
        }), 503
    
    try:
        from rag_config import RAGConfig
        return jsonify({
            "status": "ready",
            "configuration": RAGConfig.get_status(),
            "message": "RAG system is ready for document processing"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/rag/upload-document", methods=["POST"])
def rag_upload_document():
    """Upload and index a document (resume, JD, transcript, etc.)"""
    
    if not rag_ready:
        return jsonify({"error": "RAG system not initialized"}), 503
    
    if "document" not in request.files:
        return jsonify({"error": "No document file provided"}), 400
    
    doc_type = request.form.get("doc_type", "unknown")  # resume, job_description, transcript, etc.
    
    try:
        document_file = request.files["document"]
        
        # Save to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{Path(document_file.filename).suffix}") as tf:
            temp_path = tf.name
            document_file.save(temp_path)
        
        # Process document
        doc, chunks = document_manager.process_file(temp_path, doc_type=doc_type)
        
        logger.info(f"Loaded document: {doc.source}, Type: {doc_type}, Chunks: {len(chunks)}")
        
        # Index chunks to vector store
        rag_retriever.index_documents(chunks)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return jsonify({
            "status": "success",
            "message": f"Document uploaded and indexed successfully",
            "document": {
                "filename": document_file.filename,
                "type": doc_type,
                "chunks": len(chunks),
                "size": doc.metadata.get('file_size', 0)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/rag/analyze-with-context", methods=["POST"])
def rag_analyze_with_context():
    """
    Analyze interview answer with RAG-enhanced context
    Retrieves relevant information from uploaded documents
    """
    
    if not rag_ready:
        return jsonify({"error": "RAG system not initialized"}), 503
    
    data = request.get_json() or {}
    query = data.get("query") or data.get("answer", "")
    doc_type_filter = data.get("doc_type_filter")
    
    if not query:
        return jsonify({"error": "Query/answer is required"}), 400
    
    try:
        # Retrieve context from vector store
        context_results = rag_retriever.retrieve_context(
            query,
            top_k=5,
            doc_type_filter=doc_type_filter
        )
        
        # Generate analysis with context
        result = rag_llm.generate_with_retrieval(
            query,
            rag_retriever,
            doc_type_filter=doc_type_filter
        )
        
        return jsonify({
            "status": "success",
            "analysis": result['response'],
            "sources": result['retrieved_sources'],
            "context_retrieved": len(context_results) > 0
        }), 200
        
    except Exception as e:
        logger.error(f"Error in RAG analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/rag/full-interview-analysis", methods=["POST"])
def rag_full_interview_analysis():
    """
    Comprehensive interview analysis using all agents
    Performs end-to-end analysis with resume, JD, transcript, and interview data
    """
    
    if not rag_ready or agent_orchestrator is None:
        return jsonify({"error": "RAG system or agents not initialized"}), 503
    
    data = request.get_json() or {}
    
    # Extract uploaded document contents (should be processed already)
    resume_content = data.get("resume_content", "")
    jd_content = data.get("job_description_content", "")
    transcript = data.get("interview_transcript", "")
    answers = data.get("interview_answers", [])
    emotion_data = data.get("emotion_data", {})
    speech_data = data.get("speech_data", {})
    
    try:
        logger.info("Starting full interview analysis with agents...")
        
        # Run complete analysis pipeline
        analysis_result = agent_orchestrator.run_full_analysis(
            resume=resume_content,
            job_description=jd_content,
            interview_transcript=transcript,
            emotion_data=emotion_data,
            speech_data=speech_data,
            answers=answers
        )
        
        if analysis_result['status'] == 'success':
            return jsonify({
                "status": "success",
                "analysis": analysis_result['results']
            }), 200
        else:
            return jsonify({
                "status": "error",
                "error": analysis_result.get('error'),
                "partial_results": analysis_result.get('partial_results')
            }), 500
            
    except Exception as e:
        logger.error(f"Error in full interview analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/rag/query-documents", methods=["POST"])
def rag_query_documents():
    """
    Query uploaded documents directly
    Retrieves relevant information based on a search query
    """
    
    if not rag_ready:
        return jsonify({"error": "RAG system not initialized"}), 503
    
    data = request.get_json() or {}
    query = data.get("query", "")
    doc_type_filter = data.get("doc_type_filter")
    top_k = data.get("top_k", 5)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        # Retrieve documents
        results = rag_retriever.retrieve_context(query, top_k=top_k, doc_type_filter=doc_type_filter)
        
        return jsonify({
            "status": "success",
            "query": query,
            "results": [
                {
                    "id": r['id'],
                    "score": r['score'],
                    "content": r['metadata'].get('content', ''),
                    "source": r['metadata'].get('source'),
                    "doc_type": r['metadata'].get('doc_type'),
                }
                for r in results
            ],
            "total_results": len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────── EMOTION & ANALYSIS ENDPOINTS ────────────────────────────────────────


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a single image file.
    Detects faces first, crops each face, runs emotion detection.
    Returns list of detected emotions with confidence.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        img = cv2.imread(tmp_path)
        if img is None:
            return jsonify({"error": "Could not read image"}), 400

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_detector.process(rgb)
        
        faces = []
        if result.detections:
            for det in result.detections:
                bb = det.location_data.relative_bounding_box
                x = max(0, int(bb.xmin * w) - 20)
                y = max(0, int(bb.ymin * h) - 20)
                w_face = int(bb.width * w) + 40
                h_face = int(bb.height * h) + 40
                faces.append((x, y, w_face, h_face))
        
        print(f"[PREDICT] Image size: {w}x{h}, Faces detected: {len(faces)}")

        output = []

        if len(faces) == 0:
            # No face found — run model on full image
            print("[PREDICT] No faces detected, running on full image")
            results = model.predict(source=img, conf=0.2, verbose=False)
            for box in results[0].boxes:
                output.append({
                    "face":       1,
                    "emotion":    model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]) * 100, 2),
                })
            print(f"[PREDICT] Full image emotion results: {output}")
        else:
            # Crop each face and predict emotion
            for i, (x, y, w, h) in enumerate(faces):
                pad  = 20
                crop = img[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
                results = model.predict(source=crop, conf=0.2, verbose=False)
                if results[0].boxes:
                    box = results[0].boxes[0]
                    output.append({
                        "face":       i + 1,
                        "emotion":    model.names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]) * 100, 2),
                        "bbox":       [int(x), int(y), int(w), int(h)],
                    })
            print(f"[PREDICT] Cropped face emotion results: {output}")

        return jsonify({
            "faces_detected": len(faces),
            "detections":     output,
            "top_emotion":    output[0]["emotion"]    if output else "No detection",
            "top_confidence": output[0]["confidence"] if output else 0,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.unlink(tmp_path)


# ── Analyze: full video + resume analysis ─────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts video + optional resume files.
    Returns full interview analysis scores.
    """
    if "video" not in request.files:
        return jsonify({"error": "video file required"}), 400

    video_file = request.files["video"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf:
        vf_path = vf.name
        video_file.save(vf_path)

    try:
        cap    = cv2.VideoCapture(vf_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        sample = max(1, int(fps))
        counts = {n: 0 for n in model.names.values()}
        total  = 0
        idx    = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample == 0:
                h_frame, w_frame = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_detector.process(rgb)
                if result.detections:
                    for det in result.detections:
                        bb = det.location_data.relative_bounding_box
                        x = max(0, int(bb.xmin * w_frame) - 20)
                        y = max(0, int(bb.ymin * h_frame) - 20)
                        x2 = min(w_frame, int((bb.xmin + bb.width) * w_frame) + 20)
                        y2 = min(h_frame, int((bb.ymin + bb.height) * h_frame) + 20)
                        crop = frame[y:y2, x:x2]
                        if crop.size > 0:
                            res = model.predict(source=crop, conf=0.01, verbose=False)[0]
                            if res.boxes:
                                best_conf = float(res.boxes[0].conf[0])
                                emotion = model.names[int(res.boxes[0].cls[0])]
                                print(f"[ANALYZE] Frame {idx}: Detected {emotion} with conf {best_conf:.3f}")
                                counts[emotion] += 1
                                total += 1
                            else:
                                print(f"[ANALYZE] Frame {idx}: Face detected but no emotion boxes (too low conf)")
                else:
                    if idx % (sample * 10) == 0:
                        print(f"[ANALYZE] Frame {idx}: No faces detected")
            idx += 1
        cap.release()

        if total == 0:
            emotion_pct   = {k: 0 for k in counts}
            emotion_score = 50.0
        else:
            emotion_pct   = {k: round(v / total * 100, 1) for k, v in counts.items()}
            pos = emotion_pct.get("Happy", 0) + emotion_pct.get("Neutral", 0)
            neg = (emotion_pct.get("Angry", 0) + emotion_pct.get("Fear", 0) +
                   emotion_pct.get("Sad", 0)   + emotion_pct.get("Discomfort", 0))
            emotion_score = round(min(100, max(0, 50 + pos * 0.5 - neg * 0.3)), 1)

        # Resume analysis if provided
        resume_score   = 70.0
        matched_skills = []
        if "resume" in request.files:
            try:
                import fitz
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as rf:
                    rf_path = rf.name
                    request.files["resume"].save(rf_path)
                doc  = fitz.open(rf_path)
                text = "".join(p.get_text() for p in doc).lower()
                doc.close()
                os.unlink(rf_path)
                kws = list(QUESTION_BANK.keys())
                matched_skills = [k for k in kws if k in text]
                resume_score   = round(min(100, len(matched_skills) / len(kws) * 150), 1)
            except Exception:
                pass

        comm_score  = round(emotion_score * 0.4 + resume_score * 0.6, 1)
        final_score = round(emotion_score * 0.30 + comm_score * 0.35 + resume_score * 0.35, 1)

        if final_score >= 85:   rec = "Highly Recommended"
        elif final_score >= 70: rec = "Recommended"
        elif final_score >= 55: rec = "Neutral"
        else:                   rec = "Not Recommended"

        return jsonify({
            "emotion_score":       emotion_score,
            "communication_score": comm_score,
            "resume_score":        resume_score,
            "final_score":         final_score,
            "recommendation":      f"{rec}. Emotion: {emotion_score}%, Communication: {comm_score}%, Resume: {resume_score}%",
            "emotions":            emotion_pct,
            "matched_skills":      matched_skills,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.unlink(vf_path)


# ── Interview: start — parse resume, return keyword-matched questions ──────────
@app.route("/interview/start", methods=["POST"])
def interview_start():
    """
    Upload resume PDF → keyword matching → returns 7 tailored questions.
    Falls back to default questions if no resume provided.
    """
    resume_text = ""

    if "resume" in request.files:
        try:
            import fitz
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as rf:
                rf_path = rf.name
                request.files["resume"].save(rf_path)
            doc         = fitz.open(rf_path)
            resume_text = "".join(p.get_text() for p in doc).lower()
            doc.close()
            os.unlink(rf_path)
        except Exception as e:
            print(f"Resume read error: {e}")

    if not resume_text.strip():
        return jsonify({"questions": DEFAULT_QUESTIONS, "source": "default"})

    # Match keywords from resume → pick relevant questions
    matched_questions = []
    matched_keywords  = []

    for keyword, qs in QUESTION_BANK.items():
        if keyword in resume_text:
            matched_keywords.append(keyword)
            matched_questions.extend(qs)

    # Deduplicate
    seen             = set()
    unique_questions = []
    for q in matched_questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    # Fill up to 7 with defaults if not enough matched
    final_questions = unique_questions[:7]
    if len(final_questions) < 7:
        for dq in DEFAULT_QUESTIONS:
            if dq not in final_questions:
                final_questions.append(dq)
            if len(final_questions) == 7:
                break

    return jsonify({
        "questions":        final_questions,
        "source":           "resume" if matched_keywords else "default",
        "matched_keywords": matched_keywords,
    })


# ── Interview: score a single answer locally ──────────────────────────────────
@app.route("/interview/score-answer", methods=["POST"])
def score_answer():
    """
    Scores one interview answer using local rule-based logic.
    No external API needed.
    """
    data         = request.get_json() or {}
    answer       = data.get("answer", "")
    emotion      = data.get("emotion", "Neutral")
    word_count   = data.get("word_count", 0)
    filler_count = data.get("filler_count", 0)

    if not answer.strip():
        return jsonify({
            "score":        30,
            "feedback":     "No answer was recorded. Please speak clearly.",
            "strengths":    [],
            "improvements": ["Make sure to speak loud and clearly into the microphone."],
            "emotion":      emotion,
        })

    # 1. Length score — ideal answer is 80-150 words
    if word_count >= 150:   length_score = 100
    elif word_count >= 80:  length_score = 90
    elif word_count >= 50:  length_score = 70
    elif word_count >= 20:  length_score = 50
    else:                   length_score = 25

    # 2. Filler word penalty
    filler_ratio = filler_count / max(word_count, 1)
    filler_score = max(0, 100 - int(filler_ratio * 300))

    # 3. Emotion score
    emotion_scores = {
        "Happy":      100,
        "Neutral":    80,
        "Discomfort": 50,
        "Sad":        40,
        "Fear":       35,
        "Angry":      20,
    }
    emotion_score = emotion_scores.get(emotion, 60)

    # 4. Vocabulary richness — unique meaningful words
    stop_words = {
        "the","a","an","is","it","in","on","and","or","to","of",
        "i","my","we","you","was","be","for","that","this","with","at",
    }
    words             = answer.lower().split()
    unique_meaningful = len(set(w for w in words if w not in stop_words and len(w) > 3))
    richness_score    = min(100, unique_meaningful * 4)

    # 5. Final weighted score
    final_score = int(
        length_score   * 0.30 +
        filler_score   * 0.25 +
        emotion_score  * 0.25 +
        richness_score * 0.20
    )

    # 6. Build human-readable feedback
    strengths    = []
    improvements = []

    if word_count >= 80:
        strengths.append("Good answer length — well elaborated.")
    else:
        improvements.append("Try to elaborate more — aim for at least 80 words.")

    if filler_ratio < 0.05:
        strengths.append("Very few filler words — speech was fluent.")
    elif filler_ratio < 0.10:
        improvements.append("Reduce filler words like 'um', 'uh', 'like'.")
    else:
        improvements.append("Too many filler words — practice speaking more deliberately.")

    if emotion in ["Happy", "Neutral"]:
        strengths.append(f"Good emotional presence — you appeared {emotion.lower()}.")
    else:
        improvements.append("Try to appear more confident and calm during your answer.")

    if richness_score >= 60:
        strengths.append("Good vocabulary — varied and meaningful word choice.")
    else:
        improvements.append("Use more specific and varied vocabulary in your answers.")

    if final_score >= 80:   feedback = "Strong answer with good structure and confidence."
    elif final_score >= 60: feedback = "Decent answer — a few areas to refine."
    else:                   feedback = "Answer needs more depth and confidence."

    return jsonify({
        "score":        final_score,
        "feedback":     feedback,
        "strengths":    strengths,
        "improvements": improvements,
        "emotion":      emotion,
        "breakdown": {
            "length_score":   length_score,
            "filler_score":   filler_score,
            "emotion_score":  emotion_score,
            "richness_score": richness_score,
        },
    })


if __name__ == "__main__":
    # host=0.0.0.0 is required for Docker to expose the port
    app.run(host="0.0.0.0", port=5000, debug=False)