# Smart Interaction using YOLOv8 and OpenCV
# ---------------------------------------------------
# Step 1: Install Dependencies
# pip install ultralytics opencv-python flask

# Step 2: Import Libraries
from ultralytics import YOLO
import cv2
import time
import threading

# Step 3: Load Model
model = YOLO("yolov8n.pt")  # 'n' = nano version for low-resource use

# Step 4: Define AI-triggered interaction (BONUS)
def smart_interaction(detected_classes):
    if "person" in detected_classes:
        print("Welcome! Lights ON")
    if "bottle" in detected_classes:
        print("Reminder: Stay hydrated!")

# Step 5: Real-Time Detection from Webcam
def run_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)
        detected_classes = set()

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                detected_classes.add(label)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        smart_interaction(detected_classes)
        cv2.imshow("Smart Interaction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Step 6: Execute
run_detection()

