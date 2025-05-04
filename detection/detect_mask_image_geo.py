def detect_mask_image_with_geo(image_path, model_type="mobilenet", confidence_threshold=0.5, latitude=None, longitude=None):
    import os
    import cv2
    import numpy as np
    import time
    import geocoder
    import json
    from tensorflow.keras.models import load_model

    # Load the face detection model
    faceNet = cv2.dnn.readNet(
        "face_detector/deploy.prototxt",
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    )

    # Load the mask detection model
    model_path = "models/ResNet50_mask_detector.model" if model_type == "resnet" else "models/mask_detector.model"
    model = load_model(model_path)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    results = []

    # Loop through the detections and classify faces as 'Mask' or 'No Mask'
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

        # Resize face for mask detection
        face = cv2.resize(face, (224, 224))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)

        # Predict mask or no mask
        (mask, withoutMask) = model.predict(face)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        confidence_value = float(max(mask, withoutMask))
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw the label and bounding box on the image
        label_text = f"{label}: {confidence_value * 100:.2f}%"
        cv2.putText(image, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        # Append result to list
        results.append({
            "label": label,
            "confidence": confidence_value,
            "box": [int(startX), int(startY), int(endX), int(endY)]
        })

    # If no results, add a dummy result to log the violation
    if not results:
        results.append({
            "label": "No Mask Detected",
            "confidence": 0.0,
            "box": [0, 0, 0, 0]  # No bounding box
        })

    # Save the image with bounding boxes and labels
    output_dir = "static/Geo_Image_Detection"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{int(time.time())}_{os.path.basename(image_path)}"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)

    # Get latitude and longitude
    if latitude is None or longitude is None:
        g = geocoder.ip('me')
        lat, lng = g.latlng if g.latlng else ("Unknown", "Unknown")
    else:
        lat, lng = latitude, longitude

    # Log the results and save them to a JSON file
    detection_time = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "latitude": lat,
        "longitude": lng,
        "timestamp": detection_time,
        "image": output_path.replace("static/", ""),  # Relative path for web use
        "results": results
    }

    json_log_dir = os.path.join("static", "logs")
    os.makedirs(json_log_dir, exist_ok=True)
    json_log_file = os.path.join(json_log_dir, "mask_violations.json")

    geo_data = []
    if os.path.exists(json_log_file):
        with open(json_log_file, 'r') as f:
            try:
                geo_data = json.load(f)
            except json.JSONDecodeError:
                geo_data = []

    geo_data.append({
        "latitude": lat,
        "longitude": lng,
        "weight": 1.0 if any(r["label"] == "No Mask" for r in results) else 0.5,
        "timestamp": detection_time
    })

    with open(json_log_file, 'w') as f:
        json.dump(geo_data, f, indent=4)

    return output_path.replace("static/", ""), lat, lng, results
