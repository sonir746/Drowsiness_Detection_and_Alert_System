import cv2
import pyttsx3
import time as t
import torch
from ultralytics import YOLO

# Load the custom-trained YOLOv8 model
model = YOLO('Source\Model\model.pt')

# Start video capture from default webcam (0)
cap = cv2.VideoCapture(0)

# Initialize counter for drowsiness timing
count = 0

# Create a named window once to prevent flickering
cv2.namedWindow("Drowsiness Detection", cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame horizontally (mirror view)
    frame = cv2.flip(frame, 1)

    # Run detection on the current frame
    results = model.predict(
        source=frame,
        show=False,  # Don't auto-display the result — we'll handle display manually
        save=False,
        classes=[0, 1, 2],  # Limit prediction to relevant classes
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Extract detected class names from results
    names = model.names
    check = []
    for r in results:
        for c in r.boxes.cls:
            check.append(names[int(c)])

    # Drowsiness detection logic
    if "face" in check:
        if "close eyes" in check:
            if count == 0:
                start_time = float(t.time())
                count = 1
            end_time = float(t.time())
            run_time = end_time - start_time

            # Trigger alert if eyes are closed for more than 1 second
            if run_time > 1:
                print("alert")
                alert = pyttsx3.init()
                alert.say("Wake up")
                alert.runAndWait()
                alert = None
                print(run_time)
        elif "open eyes" in check:
            alert = None
            print("no alarm")
            count = 0

    # Display annotated frame (bounding boxes, labels)
    annotated_frame = results[0].plot()
    cv2.imshow("Drowsiness Detection", annotated_frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(2) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
