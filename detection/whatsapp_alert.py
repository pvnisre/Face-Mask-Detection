import os
import cv2
import time
import numpy as np
import datetime
from flask import session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils.compliance_logger import log_compliance


# ===== Set up paths =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_PROTO = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
FACE_MODEL = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
MASK_MODEL = os.path.join(BASE_DIR, "models", "mask_detector.model")
STATIC_IMAGE_PATH = os.path.join(BASE_DIR, "static", "Whatsapp_Images")
os.makedirs(STATIC_IMAGE_PATH, exist_ok=True)

def detect_and_alert():
    # Load models
    faceNet = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
    maskNet = load_model(MASK_MODEL)

    # Capture frame from webcam
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)  # small delay to avoid blurry capture
    ret, frame = cap.read()
    cap.release()

    if not ret:
        session["last_result"] = "No Frame"
        session["last_image_path"] = None
        return "No Frame", None, None
    
    frame = cv2.flip(frame, 1)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    result = "No Face"
    label = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (224, 224))
            face_rgb = img_to_array(face_rgb)
            face_rgb = preprocess_input(face_rgb)
            face_rgb = np.expand_dims(face_rgb, axis=0)

            (mask, withoutMask) = maskNet.predict(face_rgb)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            confidence_score = max(mask, withoutMask) * 100
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, f"{label}: {confidence_score:.2f}%", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            result = label
            break

    # Save image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alert_image_{timestamp}.jpg"
    full_path = os.path.join(STATIC_IMAGE_PATH, filename)
    web_path = f"/static/Whatsapp_Images/{filename}"
    cv2.imwrite(full_path, frame)

    # Save to session
    session["last_result"] = result
    session["last_image_path"] = web_path
    session["last_confidence"] = f"{confidence_score:.2f}%" if label else None
    session["whatsapp_sent"] = False

    # Log compliance result if a face was detected
    if result in ["Mask", "No Mask"]:
        log_compliance(result, detection_type="WhatsApp Alert")


    return result, web_path, f"{confidence_score:.2f}%" if label else None

#if __name__ == "__main__":
#   print("Running WhatsApp alert detection...")
#   result, image_path, confidence = detect_and_alert()
#    print("Result:", result)
#    print("Image Path:", image_path)
#    print("Confidence:", confidence)
