from ultralytics import YOLO
import cv2
import mediapipe as mp

BASE          = r"E:\AI_Interview_Communication_Analyzer"
emotion_model = YOLO(rf"{BASE}\best_emotion_model.pt")

print("Model loaded!")
print("Classes:", emotion_model.names)

# MediaPipe face detector — much more accurate than Haar cascade
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(
    model_selection=0,        # 0 = short range (webcam), 1 = long range
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)
print("Webcam started. Press Q to quit.")
print("Make sure your face is clearly visible and well lit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    if result.detections:
        for det in result.detections:
            bb   = det.location_data.relative_bounding_box
            x1   = max(0, int(bb.xmin * w) - 20)
            y1   = max(0, int(bb.ymin * h) - 20)
            x2   = min(w, int((bb.xmin + bb.width)  * w) + 20)
            y2   = min(h, int((bb.ymin + bb.height) * h) + 20)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            emo_result = emotion_model.predict(
                source=crop, conf=0.2, verbose=False
            )[0]

            if emo_result.boxes:
                emo  = emotion_model.names[int(emo_result.boxes[0].cls)]
                conf = round(float(emo_result.boxes[0].conf)*100, 1)

                color = {
                    "Happy":      (0,255,0),
                    "Neutral":    (200,200,0),
                    "Angry":      (0,0,255),
                    "Sad":        (255,0,0),
                    "Fear":       (0,165,255),
                    "Discomfort": (128,0,128),
                }.get(emo, (0,255,0))

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"{emo}  {conf}%",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.85, color, 2)
            else:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(180,180,180),1)
    else:
        cv2.putText(frame, "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (100,100,100), 1)

    cv2.imshow("AI Interview Emotion Analyzer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed.")