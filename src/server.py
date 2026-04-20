from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import threading
from io import BytesIO
from PIL import Image
import base64

from src.config import Config
from src.core.detection import detect_faces, get_available_cameras
from src.core.recognition import load_references, save_references, get_face_encodings, recognize_face

app = Flask(__name__, template_folder='../templates')

camera = None
lock = threading.Lock()
current_camera_index = 0
last_detected_count = 0
last_face_encoding = None
rtsp_url = None
rtsp_camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def switch_camera(camera_index):
    global camera, current_camera_index
    with lock:
        if camera is not None:
            camera.release()
        current_camera_index = camera_index
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera = cv2.VideoCapture(0)
            current_camera_index = 0
            return False
    return True

def set_rtsp_source(url):
    global rtsp_camera, rtsp_url
    with lock:
        if rtsp_camera is not None:
            rtsp_camera.release()
        rtsp_camera = cv2.VideoCapture(url)
        if not rtsp_camera.isOpened():
            rtsp_camera = None
            rtsp_url = None
            return False
        rtsp_url = url
    return True

def draw_detections(frame, results):
    global last_detected_count, last_face_encoding
    references = load_references()
    h, w, c = frame.shape
    face_encodings_in_frame = get_face_encodings(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

            if encoding_idx < len(face_encodings_in_frame):
                current_encoding = face_encodings_in_frame[encoding_idx]
                last_face_encoding = current_encoding

                best_match = recognize_face(current_encoding, references)

                if best_match:
                    color = (0, 255, 0)
                    text = best_match
                else:
                    color = (0, 0, 255)
                    text = "Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                encoding_idx += 1
                detected_count += 1

    last_detected_count = detected_count
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        frame_data = data.get('frame')

        if not frame_data:
            print("No frame data received")
            return jsonify({'status': 'error', 'message': 'No frame data'}), 400

        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        try:
            img_data = base64.b64decode(frame_data)
        except Exception as e:
            print(f"Base64 decode error: {e}")
            return jsonify({'status': 'error', 'message': f'Invalid base64: {e}'}), 400

        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results, rgb_frame = detect_faces(frame)
        frame = draw_detections(frame, results)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        return Response(frame_bytes, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

def generate_frames():
    while True:
        with lock:
            if rtsp_camera is not None:
                ret, frame = rtsp_camera.read()
            else:
                cam = get_camera()
                ret, frame = cam.read()

        if not ret:
            break

        try:
            results, rgb_frame = detect_faces(frame)
            frame = draw_detections(frame, results)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in frame generation: {e}")
            continue

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rtsp_feed')
def rtsp_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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

@app.route('/set_rtsp_source', methods=['POST'])
def set_rtsp_source_endpoint():
    data = request.json
    url = data.get('url')
    if url and set_rtsp_source(url):
        return jsonify({'status': 'success', 'url': url})
    return jsonify({'status': 'error', 'message': 'Failed to connect to RTSP source'})

@app.route('/get_references')
def get_references():
    return jsonify(list(load_references().keys()))

@app.route('/save_reference', methods=['POST'])
def save_reference():
    global last_face_encoding
    data = request.json
    name = data.get('name')

    if not name or not name.strip():
        return jsonify({'status': 'error', 'message': 'Name is required'})

    if last_face_encoding is None:
        return jsonify({'status': 'error', 'message': 'No face detected. Position yourself in front of the camera.'})

    references = load_references()
    if name in references:
        return jsonify({'status': 'error', 'message': f'Face "{name}" already exists'})

    references[name] = {
        'type': 'human',
        'encoding': last_face_encoding.tolist()
    }
    save_references(references)

    return jsonify({'status': 'success', 'message': f'Face "{name}" added successfully!'})

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

def run(debug=False, host='0.0.0.0', port=5005):
    app.run(debug=debug, host=host, port=port)
