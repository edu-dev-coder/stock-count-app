from ultralytics import YOLO

# Load the smallest YOLOv8 model.
# This is best for phone because it is lighter and faster.
model = YOLO("yolov8n.pt")

# Export to ONNX format for mobile deployment.
# imgsz=320 makes it lighter for mobile.
# For TFLite, run this with Python 3.11+ with TensorFlow installed:
# pip install tensorflow
# model.export(format="tflite", imgsz=320)
model.export(format="tflite", imgsz=320)