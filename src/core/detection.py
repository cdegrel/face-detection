import cv2
import mediapipe as mp
from src.config import Config

mp_face_detection = mp.solutions.face_detection

def detect_faces(frame):
    with mp_face_detection.FaceDetection(
        min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE
    ) as face_detection:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        return results, rgb_frame

def get_available_cameras(max_index=5):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available
