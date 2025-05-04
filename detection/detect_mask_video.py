# detect_mask_video.py
import cv2
import numpy as np
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def detect_mask_video_stream():
    face_detector_path = "face_detector"
    faceNet = cv2.dnn.readNet(
        os.path.join(face_detector_path, "deploy.prototxt"),
        os.path.join(face_detector_path, "res10_300x300_ssd_iter_140000.caffemodel")
    )

    maskNet = load_model("models/mask_detector.model")

    def detect_and_predict_mask(frame, faceNet, maskNet, confidence_threshold=0.5):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces, locs, preds = [], [], []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < confidence_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

        if faces:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)
        else:
            preds = []

        return locs, preds

    # Use OpenCV's VideoCapture for smoother real-time handling
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    try:
        while True:
            ret, frame = vs.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)  # Fix mirrored video
            frame = cv2.resize(frame, (400, int(frame.shape[0] * 400 / frame.shape[1])))

            locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)

            if not locs:
                cv2.putText(frame, "No Face Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                confidence = max(mask, withoutMask) * 100
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label_text = f"{label}: {confidence:.2f}%"

                cv2.putText(frame, label_text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        vs.release()
