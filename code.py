!pip install ultralytics

from ultralytics import YOLO

import os

# Load the YOLOv8 model - you can start with a pre-trained model like 'yolov8n.pt' (nano) or 'yolov8s.pt' (small)
model = YOLO('yolov8s.pt')  # You can use other versions like 'yolov8n.pt' for the nano version

# Train the model
model.train(data=os.path.join(dataset.location, "/content/Ather-id-2/data.yaml"), epochs=10, imgsz=640, batch=16, name="FinalTrain")

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="uOjF7eVc6NYXFt1P8ANR")
project = rf.workspace("ather-mye9c").project("ather-id")
version = project.version(2)
dataset = version.download("yolov11")

results = model.predict(source="/content/MultiTest.jpg", save=True)

from ultralytics import YOLO

# Load the trained model using the best.pt file
model = YOLO('/content/best.pt')

# Run inference on a new image
results = model.predict(source="/content/1739727031266.jpg", save=True)
