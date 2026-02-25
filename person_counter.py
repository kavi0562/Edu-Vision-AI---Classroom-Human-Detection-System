from ultralytics import YOLO
import cv2
import os

# Load trained model (mee best model use chey) - using the high accuracy model
model_path = "classroom_ai/max_accuracy_run/weights/best.pt"
if not os.path.exists(model_path):
    print(f"Trained model not found at {model_path}. Using base yolov8x.pt for high accuracy.")
    model_path = "yolov8x.pt"

model = YOLO(model_path)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open avvaledu \u274c")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO prediction (added high confidence and NMS IoU threshold for accurate counting)
    results = model(frame, conf=0.5, iou=0.45)

    person_count = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # Only person class (0)
            if cls == 0:
                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, "Person", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Show count
    cv2.putText(frame, f"Count: {person_count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # Display
    cv2.imshow("Person Counter", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()