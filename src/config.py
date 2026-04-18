import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Server
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    PORT = int(os.getenv('PORT', '5005'))
    HOST = os.getenv('HOST', '0.0.0.0')

    # Face detection
    MIN_DETECTION_CONFIDENCE = float(os.getenv('MIN_DETECTION_CONFIDENCE', '0.5'))

    # Face recognition
    RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', '0.6'))

    # Files
    REFERENCES_FILE = os.getenv('REFERENCES_FILE', 'face_references.pkl')
