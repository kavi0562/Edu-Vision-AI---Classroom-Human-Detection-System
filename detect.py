from ultralytics import YOLO
import cv2

# Load your trained model (later best.pt replace cheyyi)
model = YOLO("runs/detect/train*/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Draw results
    annotated_frame = results[0].plot()

    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()