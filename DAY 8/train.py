from ultralytics import YOLO

# Train YOLO model using DATASET/data.yaml in this workspace.
model = YOLO("yolov8n.pt")
model.train(data="DATASET/data.yaml", epochs=50, imgsz=640)
