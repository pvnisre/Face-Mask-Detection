from utils.compliance_logger import log_compliance  # <-- Added import

def detect_mask_image(image_path, model_type="mobilenet", confidence_threshold=0.5):
    import os
    import cv2
    import numpy as np
    import time
    from tensorflow.keras.models import load_model

    faceNet = cv2.dnn.readNet(
        "face_detector/deploy.prototxt",
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    )

    model_path = "models/ResNet50_mask_detector.model" if model_type == "resnet" else "models/mask_detector.model"
    model = load_model(model_path)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w - 1, endX), min(h - 1, endY)

        if endX - startX < 10 or endY - startY < 10:
            continue

        face = image[startY:endY, startX:endX]
        if face.size == 0:
            continue

        face = cv2.resize(face, (224, 224))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)

        (mask, withoutMask) = model.predict(face)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
        cv2.putText(image, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        results.append({
            "label": label,
            "confidence": float(max(mask, withoutMask)),
            "box": [int(startX), int(startY), int(endX), int(endY)]
        })

    # Save output image in the desired directory
    output_dir = "static/Image_Detection"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{int(time.time())}_{os.path.basename(image_path)}"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)

    if results:
        first_result = results[0]['label']
        if first_result in ["Mask", "No Mask"]:
            log_compliance(first_result, detection_type="Image Detection")

    return {"output_path": output_path.replace("\\", "/"), "results": results}
