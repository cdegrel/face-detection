from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import pickle
import os
import threading

app = Flask(__name__)

REFERENCES_FILE = "face_references.pkl"
camera = cv2.VideoCapture(0)
lock = threading.Lock()
current_camera_index = 0
last_detected_count = 0
last_face_encoding = None

mp_face_detection = mp.solutions.face_detection

def get_available_cameras(max_index=5):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def switch_camera(camera_index):
    global camera, current_camera_index
    with lock:
        camera.release()
        current_camera_index = camera_index
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera = cv2.VideoCapture(0)
            current_camera_index = 0
            return False
    return True

def load_references():
    if os.path.exists(REFERENCES_FILE):
        with open(REFERENCES_FILE, 'rb') as f:
            refs = pickle.load(f)
            new_refs = {}
            for name, ref_data in refs.items():
                if isinstance(ref_data, dict):
                    new_refs[name] = ref_data
                else:
                    new_refs[name] = {'type': 'human', 'encoding': ref_data.tolist() if hasattr(ref_data, 'tolist') else ref_data}
            return new_refs
    return {}

def save_references(references):
    with open(REFERENCES_FILE, 'wb') as f:
        pickle.dump(references, f)

def generate_frames():
    global last_detected_count, last_face_encoding
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        references = load_references()

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame.shape

            results = face_detection.process(rgb_frame)

            small_rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            face_encodings_in_frame = face_recognition.face_encodings(small_rgb_frame)

            encoding_idx = 0
            detected_count = 0

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box

                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    padding_x = int(width * 0.1)
                    padding_y = int(height * 0.2)

                    x1 = max(0, x - padding_x)
                    y1 = max(0, y - padding_y)
                    x2 = min(w, x + width + padding_x)
                    y2 = min(h, y + height + padding_y)

                    best_match = None
                    best_distance = 0.6

                    if encoding_idx < len(face_encodings_in_frame):
                        current_encoding = face_encodings_in_frame[encoding_idx]
                        last_face_encoding = current_encoding

                        for name, ref_data in references.items():
                            if ref_data.get('type') == 'human':
                                ref_encoding = np.array(ref_data['encoding']) if isinstance(ref_data['encoding'], list) else ref_data['encoding']
                                distance = face_recognition.face_distance([ref_encoding], current_encoding)[0]
                                if distance < best_distance:
                                    best_distance = distance
                                    best_match = name

                    if best_match:
                        color = (0, 255, 0)
                        text = best_match
                    else:
                        color = (0, 0, 255)
                        text = "Inconnu"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    encoding_idx += 1
                    detected_count += 1

            last_detected_count = detected_count

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_references')
def get_references():
    return jsonify(list(load_references().keys()))

@app.route('/save_reference', methods=['POST'])
def save_reference():
    global last_face_encoding
    data = request.json
    name = data.get('name')

    if not name or not name.strip():
        return jsonify({'status': 'error', 'message': 'Le nom est requis'})

    if last_face_encoding is None:
        return jsonify({'status': 'error', 'message': 'Aucun visage détecté. Positionnez-vous devant la caméra.'})

    references = load_references()
    if name in references:
        return jsonify({'status': 'error', 'message': f'Le visage "{name}" existe déjà'})

    references[name] = {
        'type': 'human',
        'encoding': last_face_encoding.tolist()
    }
    save_references(references)

    return jsonify({'status': 'success', 'message': f'Visage "{name}" ajouté avec succès !'})

@app.route('/delete_reference', methods=['POST'])
def delete_reference():
    data = request.json
    name = data.get('name')
    references = load_references()
    if name in references:
        del references[name]
        save_references(references)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Face not found'})

@app.route('/get_cameras')
def get_cameras():
    available = get_available_cameras()
    return jsonify({'cameras': available, 'current': current_camera_index})

@app.route('/set_camera', methods=['POST'])
def set_camera():
    data = request.json
    camera_index = data.get('index')
    if camera_index is not None and isinstance(camera_index, int):
        if switch_camera(camera_index):
            return jsonify({'status': 'success', 'camera': camera_index})
    return jsonify({'status': 'error', 'message': 'Invalid camera index'})

@app.route('/get_detection_count')
def get_detection_count():
    return jsonify({'count': last_detected_count})

if __name__ == '__main__':
    app.run(debug=True, port=5005, host='0.0.0.0')
