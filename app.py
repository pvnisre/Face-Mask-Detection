from flask import Flask, Response, render_template, request, jsonify, redirect, url_for, session, flash
import json
import cv2
import sqlite3
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pyttsx3
from collections import defaultdict
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# -------------------- Config --------------------
USER_FILE = 'users.json'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------- Helper Functions --------------------
def load_users():
    if not os.path.exists(USER_FILE):
        return []
    with open(USER_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=2)

# -------------------- Routes --------------------
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    show_login = True
    remembered_email = ''
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        remember = 'remember' in request.form

        users = load_users()
        user = next((u for u in users if u['email'] == email), None)
        if user and check_password_hash(user['password'], password):
            session['user'] = user['username']
            if remember:
                remembered_email = email
            return redirect(url_for('dashboard'))
        else:
            return render_template('login_signup.html', error='Invalid credentials', show_login=True, remembered_email=email)

    return render_template('login_signup.html', show_login=True, remembered_email=remembered_email)

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    users = load_users()
    if any(u['email'] == email for u in users):
        return render_template('login_signup.html', error='Email already exists', show_login=False)

    hashed_password = generate_password_hash(password)
    users.append({'username': username, 'email': email, 'password': hashed_password})
    save_users(users)
    return redirect(url_for('login'))


@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    email = request.form['email']
    new_password = request.form['new_password']

    users = load_users()
    user = next((u for u in users if u['email'] == email), None)

    if user:
        user['password'] = generate_password_hash(new_password)
        save_users(users)
        return redirect(url_for('login'))
    else:
        return render_template('login_signup.html', error='Email not found', show_login=True)

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['user'])

@app.route('/image_detection')
def image_detection():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('image_detection.html')

@app.route('/run_image_detection', methods=['POST'])
def run_image_detection():
    if 'user' not in session:
        return redirect(url_for('login'))

    image = request.files.get('image')
    model_choice = request.form.get('model_choice')

    if not image or not model_choice:
        flash("Please select a model and upload an image", "danger")
        return redirect(url_for('image_detection'))

    filename = f"{int(time.time())}_{image.filename}"
    image_path = os.path.join('static/uploads', filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path)

    from detection.detect_mask_image import detect_mask_image
    try:
        result_data = detect_mask_image(image_path, model_type=model_choice)
        results = result_data.get("results", [])

        if not results:
            result_label = "No Face"
            confidence = None
        else:
            result_label = results[0]['label']
            confidence = f"{results[0]['confidence'] * 100:.2f}%"

        output_image = result_data.get("output_path", "")
        return render_template("detection_result.html",
                               result=result_label,
                               confidence=confidence,
                               output_image=output_image)
    except Exception as e:
        print("Detection Error:", e)
        flash("An error occurred during detection. Please try again.", "danger")
        return redirect(url_for('image_detection'))


@app.route("/live_webcam")
def live_webcam():
    return render_template("live_webcam.html")

@app.route("/video_feed")
def video_feed():
    from detection.detect_mask_video import detect_mask_video_stream  # lazy import here
    return Response(detect_mask_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/whatsapp_alert')
def whatsapp_alert():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('whatsapp_alert.html')


@app.route('/run_whatsapp_detection', methods=['POST'])
def run_whatsapp_detection():
    if 'user' not in session:
        return redirect(url_for('login'))

    from detection.whatsapp_alert import detect_and_alert
    result, image_path, confidence = detect_and_alert()

    session["last_result"] = result
    session["last_image_path"] = image_path
    session["last_confidence"] = confidence
    session["whatsapp_sent"] = False  # Reset flag for new detection

    return redirect(url_for('whatsapp_result'))


@app.route('/whatsapp_result')
def whatsapp_result():
    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template(
        'whatsapp_result.html',
        result=session.get("last_result", "No data"),
        image_path=session.get("last_image_path", ""),
        confidence=session.get("last_confidence", "")
    )


@app.route("/send_whatsapp_alert")
def send_whatsapp_alert():
    if 'user' not in session:
        return redirect(url_for('login'))

    
    try:
        import pywhatkit
    except:
        print("⚠️ pywhatkit could not be loaded (likely no internet). WhatsApp features disabled.")
        pywhatkit = None

    image_path = session.get("last_image_path", "").lstrip("/")
    full_path = os.path.join(os.getcwd(), image_path)

 
    if not session.get("whatsapp_sent") and os.path.exists(full_path):
        if pywhatkit:
            try:
                pywhatkit.sendwhats_image(
                    receiver="+919876543210",
                    img_path=full_path,
                    caption="🚨😷 No mask detected! Here's a captured image 📸.",
                    wait_time=30,
                    tab_close=True
                )
                session["whatsapp_sent"] = True
            except Exception as e:
                print("⚠️ WhatsApp sending failed:", e)
        else:
            print("⚠️ Skipping WhatsApp alert due to no internet.")

    return redirect(url_for("whatsapp_alert_result"))



@app.route("/whatsapp_alert_result")
def whatsapp_alert_result():
    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template(
        "whatsapp_result.html",
        result=session.get("last_result", "No data"),
        image_path=session.get("last_image_path", ""),
        confidence=session.get("last_confidence", "")
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mask_compliance.db")


@app.route('/compliance-tracker')
def compliance_tracker():
    return render_template('mask_compliance_tracker.html')


@app.route('/get-compliance-data')
def get_compliance_data():
    time_range = request.args.get('range', 'daily')  # daily, weekly, monthly

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Build date filter
    now = datetime.now()
    if time_range == 'daily':
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_range == 'weekly':
        start_time = now - timedelta(days=now.weekday())  # Monday this week
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_range == 'monthly':
        start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        start_time = datetime.min  # fallback to all data

    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    # Query the database
    cursor.execute("""
        SELECT compliance_status FROM compliance_logs
        WHERE timestamp >= ?
    """, (start_time_str,))
    results = cursor.fetchall()
    conn.close()

    compliant = len([r for r in results if r[0] == 'Compliant'])
    violation = len([r for r in results if r[0] == 'Violation'])

    return jsonify({
        "compliant": compliant,
        "violation": violation
    })

def capture_image(save_path, delay=1):
    import time
    cam = cv2.VideoCapture(0)
    time.sleep(delay)  # Allow auto-exposure and focus
    ret, frame = cam.read()
    if ret:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)
        cam.release()
        return save_path
    else:
        cam.release()
        raise Exception("Failed to capture image from webcam.")

# Route to select location for detection
@app.route('/select_location')
def select_location():
    return render_template('select_location.html')

@app.route('/detect_mask', methods=['POST'])
def detect_mask():

    from tensorflow.keras.models import load_model

    location = request.form.get('location')

    # Define location coordinates
    location_coords = {
        "Zudio": (13.614120809262024, 79.41638060959853),
        "SV University": (13.628916018115785, 79.39783745314385),
        "CMR Shopping": (13.636998051667492, 79.41984189347363),
        "Khazana Jewellery": (13.63685771554307, 79.42224142986781),
        "Park": (13.618920765641986, 79.42504136491911),
        "SVIMS Hospital": (13.635769852711523, 79.40245213956884),
        "DMart":(13.622678707625463, 79.42039709521795)
    }
    lat, lon = location_coords.get(location, (13.6288, 79.4192))
    print(f"📍 Selected Location: {location} | Coordinates: {lat}, {lon}")

    try:
        image_path = capture_image('static/Geo_Image_Detection/captured_image.jpg')

        # Load mask detection model (ResNet50 or MobileNetV2)
        model = load_model("models/ResNet50_mask_detector.model")

        # Read the captured image
        image = cv2.imread(image_path)
        image = cv2.flip(image, 1) 
        if image is None:
            raise ValueError(f"Image not found: {image_path}")

        (h, w) = image.shape[:2]
        faceNet = cv2.dnn.readNet(
            "face_detector/deploy.prototxt",
            "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        )

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        results = []

        # Loop through detections and classify faces as 'Mask' or 'No Mask'
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
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

            label_text = f"{label}: {confidence_value * 100:.2f}%"
            cv2.putText(image, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

            # Calculate weight based on label and confidence
            weight = 0.5  # Default weight
            if label == "No Mask" and confidence_value >= 0.7:
                weight = 1.0  # High weight for "No Mask Detected" with high confidence
            elif label == "Mask":
                weight = 0.0  # Low weight for "Mask"

            # Append result to list with weight
            results.append({
                "label": label,
                "confidence": confidence_value,
                "box": [int(startX), int(startY), int(endX), int(endY)],
                "weight": weight
            })

        # If no results, add a dummy result to log the violation
        if not results:
            results.append({
                "label": "No Mask Detected",
                "confidence": 0.0,
                "box": [0, 0, 0, 0],  # No bounding box
                "weight": 1.0  # Highest weight when no mask detected
            })

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = 'static/images/detections/'
        os.makedirs(save_dir, exist_ok=True)
        final_path = os.path.join(save_dir, f"{timestamp}_{location}.jpg")
        cv2.imwrite(final_path, image)

        # Save results and path in session
        session['last_image_path'] = final_path.replace('static/', '')
        session['latitude'] = lat
        session['longitude'] = lon
        session['location'] = location
        session['results'] = results

        # Log the violation data
        log_data = {
            'latitude': lat,
            'longitude': lon,
            'location': location,
            'label': results[0]["label"],
            'confidence': results[0]["confidence"],
            'weight': results[0]["weight"],  # Use the weight from results
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Ensure the log file exists
        log_file_path = os.path.join(app.root_path, 'static', 'logs', 'mask_violations.json')
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        try:
            with open(log_file_path, 'r') as file:
                try:
                    logs = json.load(file)
                except json.JSONDecodeError:
                    logs = []  # In case of malformed JSON, reset to an empty list
        except FileNotFoundError:
            logs = [] 

        # Append new log entry
        logs.append(log_data)

        # Save the updated logs back to the file
        with open(log_file_path, 'w') as file:
            json.dump(logs, file, indent=4)

        return render_template(
            'geo_detection_result.html',
            image=session['last_image_path'],
            latitude=lat,
            longitude=lon,
            location=location,
            results=results
        )

    except Exception as e:
        return f"⚠️ Error during detection: {str(e)}"


# Route for real-time geo-location mask detection
@app.route('/detect_geoimage')
def detect_geoimage():
    try:
        from detection.detect_mask_image_geo import detect_mask_image_with_geo

        # Capture image for geo-detection
        image_path = capture_image('static/Geo_Image_Detection/captured_image.jpg')

        # Perform mask detection and geo-tagging
        output_path, latitude, longitude, results = detect_mask_image_with_geo(image_path)

        print("Detection Results:", results)  # Debugging line to check structure

        # Prepare relative path for image
        image_relative_path = output_path.replace('static\\', '').replace('\\', '/')

        # Save results in session
        session['last_result'] = results
        session['last_image_path'] = image_relative_path
        session['latitude'] = latitude
        session['longitude'] = longitude

        # Ensure results are populated
        if not results:
            results.append({
                "label": "No Mask Detected",
                "confidence": 0.0,
                "box": [0, 0, 0, 0]  # No bounding box
            })

        # Render template with results and confidence scores
        return render_template(
            'geo_detection_result.html',
            image=image_relative_path,
            latitude=latitude,
            longitude=longitude,
            results=results
        )

    except Exception as e:
        return f"⚠️ Error during detection: {str(e)}"

# Route to view heatmap (mask violation incidents)
@app.route('/show_heatmap')
def show_heatmap():
    json_path = r'C:\Users\pavan\Desktop\mask-detection-project\static\logs\mask_violations.json'
    with open(json_path, 'r') as f:
        violation_data = json.load(f)

        grouped = defaultdict(list)

    for entry in violation_data:
        key = (round(entry['latitude'], 5), round(entry['longitude'], 5))  # precision control
        grouped[key].append(entry['weight'])

    averaged_data = []
    for (lat, lon), weights in grouped.items():
        avg_weight = sum(weights) / len(weights)
        averaged_data.append({
            'latitude': lat,
            'longitude': lon,
            'weight': avg_weight
        })
    return render_template('heatmap.html', data=json.dumps(averaged_data))

# Route for heatmap data

# Function to run mask detection and return results
def run_mask_detection(image_path):
    from detection.detect_mask_image_geo import detect_mask_image_with_geo
    output_path, latitude, longitude, results = detect_mask_image_with_geo(image_path)
    return output_path, latitude, longitude, results


@app.route('/voice_alert')
def voice_alert():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('voice_alert.html')


# Step 2: Handle Detection Logic and Save Results in Session
@app.route('/run_voice_detection', methods=['POST'])
def run_voice_detection():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Import the detection logic from your detection module
    from detection.voice_alert import detect_and_prepare_voice_alert

    # Run the detection: will return (Mask/No Mask/No Face), image path, confidence score
    result, image_path, confidence = detect_and_prepare_voice_alert()

    # Store results in session to be accessible in the result page
    session["last_result"] = result
    session["last_image_path"] = image_path
    session["last_confidence"] = confidence

    return redirect(url_for('voice_result'))


# Step 3: Show Result Page with Voice Alert Options and Captured Image
@app.route('/voice_result')
def voice_result():
    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template(
        'voice_result.html',
        result=session.get("last_result", "No data"),
        image_path=session.get("last_image_path", ""),
        confidence=session.get("last_confidence", "")
    )


# -------------------- Run Server --------------------
if __name__ == '__main__':

    app.run(debug=True)
