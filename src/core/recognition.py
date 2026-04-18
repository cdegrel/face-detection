import face_recognition
import cv2
import pickle
import os
from src.config import Config

def load_references():
    if os.path.exists(Config.REFERENCES_FILE):
        with open(Config.REFERENCES_FILE, 'rb') as f:
            refs = pickle.load(f)
            new_refs = {}
            for name, ref_data in refs.items():
                if isinstance(ref_data, dict):
                    new_refs[name] = ref_data
                else:
                    new_refs[name] = {
                        'type': 'human',
                        'encoding': ref_data.tolist() if hasattr(ref_data, 'tolist') else ref_data
                    }
            return new_refs
    return {}

def save_references(references):
    with open(Config.REFERENCES_FILE, 'wb') as f:
        pickle.dump(references, f)

def get_face_encodings(rgb_frame, scale=0.25):
    small_rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale)
    return face_recognition.face_encodings(small_rgb_frame)

def recognize_face(current_encoding, references):
    best_match = None
    best_distance = Config.RECOGNITION_THRESHOLD

    for name, ref_data in references.items():
        if ref_data.get('type') == 'human':
            import numpy as np
            ref_encoding = np.array(ref_data['encoding']) if isinstance(ref_data['encoding'], list) else ref_data['encoding']
            distance = face_recognition.face_distance([ref_encoding], current_encoding)[0]
            if distance < best_distance:
                best_distance = distance
                best_match = name

    return best_match
