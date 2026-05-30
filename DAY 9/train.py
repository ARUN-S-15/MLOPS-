from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)
model.train(data="DATASET/data.yaml", epochs=20, imgsz=640)
