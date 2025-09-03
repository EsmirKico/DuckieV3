import requests
import cv2
import numpy as np
import base64

API_URL = "http://localhost:8000/detect"

cap = cv2.VideoCapture(0)  # 0 is default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Encode frame as JPEG for sending to API
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

    try:
        response = requests.post(API_URL, files=files, timeout=5)
        if response.status_code == 200:
            result = response.json()
            # Draw green boxes for all detections
            if result.get("detections"):
                for det in result["detections"]:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Optionally, display debug image from API if available
            # (Uncomment next lines to overwrite with server debug image)
            # if result.get("debug_image_base64"):
            #     img_bytes = base64.b64decode(result["debug_image_base64"])
            #     nparr = np.frombuffer(img_bytes, np.uint8)
            #     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            print(f"API call failed: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Exception calling API: {e}")

    cv2.imshow("Webcam Tennis Ball Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
