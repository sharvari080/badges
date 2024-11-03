# Import required libraries
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained YOLOv5 model (YOLOv5s - small model, for faster inference)
model = YOLO("yolov5s")  # use 'yolov5l' or 'yolov5x' for larger models

# Load an image from file
img_path = "your_image.jpg"  # Replace with the path to your image
img = cv2.imread(img_path)

# Run inference
results = model(img)

# Process results
for result in results:
    boxes = result.boxes  # Get bounding boxes
    for box in boxes:
        # Extract bounding box coordinates and class names
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # top-left and bottom-right coordinates
        label = model.names[int(box.cls)]
        confidence = box.conf[0].item()

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{label} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

# Convert color for Matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
