import cv2
import math
from ultralytics import YOLO
import cvzone

# Load your YOLO model
model = YOLO('fire.pt')  # Make sure 'fire.pt' is the correct path to your model

# Class names for detection (assuming 'fire' is the class you want to detect)
classnames = ['fire']

# Main detection function
def stream_and_detect():
    cv2.namedWindow("Live Stream with Detection", cv2.WINDOW_AUTOSIZE)

    # Open the system's camera (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Resize the frame
            frame = cv2.resize(frame, (640, 480))

            # Perform object detection using YOLO
            results = model(frame, stream=True)

            # Getting bbox, confidence, and class name information
            for info in results:
                boxes = info.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    Class = int(box.cls[0])

                    if confidence > 70:  # Only detect if confidence is above 70%
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 30], scale=1.5, thickness=2)

            # Display the frame with detection
            cv2.imshow('Live Stream with Detection', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(5) == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    stream_and_detect()
