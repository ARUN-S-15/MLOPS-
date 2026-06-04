# Save this as a .py file and run in VS Code
# Uses OpenCV's built-in face detector — no extra model needed

code = '''
from ultralytics import YOLO
import cv2

BASE          = r"E:\\AI Interview & Communication Analyzer"
emotion_model = YOLO(f"{BASE}/best_emotion_model.pt")

# OpenCV built-in face detector — no extra download needed
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60)
    )

    for (x, y, w, h) in faces:
        pad  = 20
        crop = frame[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]

        result = emotion_model.predict(
            source=crop, conf=0.1, verbose=False
        )[0]

        if result.boxes:
            emo   = emotion_model.names[int(result.boxes[0].cls)]
            conf  = round(float(result.boxes[0].conf)*100, 1)
            color = {
                "Happy":      (0,255,0),
                "Neutral":    (255,255,0),
                "Angry":      (0,0,255),
                "Sad":        (255,0,0),
                "Fear":       (0,165,255),
                "Discomfort": (0,0,200),
            }.get(emo, (0,255,0))

            cv2.rectangle(frame,(x,y),(x+w,y+h), color, 2)
            cv2.putText(frame, f"{emo} {conf}%",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)

    cv2.imshow("AI Interview Emotion Analyzer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
'''

BASE = '/content/drive/MyDrive/AI_Interview_Communication_Analyzer/AI_Interview_Communication_Analyzer'

with open(f'{BASE}/realtime_emotion.py', 'w') as f:
    f.write(code)

print("realtime_emotion.py saved to Drive!")
print("Download it and run: python realtime_emotion.py")