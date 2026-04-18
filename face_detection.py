import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import pickle
import os
# from ultralytics import YOLO

mp_face_detection = mp.solutions.face_detection

REFERENCES_FILE = "face_references.pkl"

def load_references():
    if os.path.exists(REFERENCES_FILE):
        with open(REFERENCES_FILE, 'rb') as f:
            refs = pickle.load(f)
            # Convertir les anciennes références au nouveau format
            new_refs = {}
            for name, ref_data in refs.items():
                if isinstance(ref_data, dict):
                    new_refs[name] = ref_data
                else:
                    # Ancienne format: juste l'encodage
                    new_refs[name] = {'type': 'human', 'encoding': ref_data.tolist() if hasattr(ref_data, 'tolist') else ref_data}
            return new_refs
    return {}

def save_references(references):
    with open(REFERENCES_FILE, 'wb') as f:
        pickle.dump(references, f)

cap = cv2.VideoCapture(0)
references = load_references()
# yolo_model = YOLO('yolov8n.pt')  # Nano model pour rapidité

print("Commandes:")
print("  's' - Sauvegarder un visage/animal détecté")
print("  'q' - Quitter")
print(f"\nVisages/animaux stockés: {list(references.keys())}")

last_detected_encoding = None
# last_detected_hash = None

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape

        # Détection des visages humains
        results = face_detection.process(rgb_frame)

        small_rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        face_encodings_in_frame = face_recognition.face_encodings(small_rgb_frame)

        encoding_idx = 0

        # Traiter les visages humains
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
                    last_detected_encoding = current_encoding

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
                    text = "Humain inconnu"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                encoding_idx += 1

        # # Détection des animaux avec YOLO
        # yolo_results = yolo_model(frame, verbose=False)

        # for result in yolo_results:
        #     for box in result.boxes:
        #         class_id = int(box.cls)
        #         class_name = result.names[class_id]

        #         # Filtrer pour les animaux
        #         if class_name in ['cat', 'dog', 'bird', 'rabbit']:
        #             x1, y1, x2, y2 = map(int, box.xyxy[0])

        #             # Créer un hash simple du visage pour la reconnaissance
        #             face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        #             if face_crop.size > 0:
        #                 face_hash = hash(face_crop.tobytes()) % ((2**31) - 1)
        #                 last_detected_hash = face_hash

        #                 best_match = None
        #                 for name, ref_data in references.items():
        #                     if ref_data['type'] == 'animal' and ref_data.get('species') == class_name:
        #                         if ref_data.get('hash') == face_hash:
        #                             best_match = name
        #                             break

        #                 if best_match:
        #                     color = (255, 0, 0)  # Bleu pour les animaux reconnus
        #                     text = best_match
        #                 else:
        #                     color = (255, 165, 0)  # Orange pour animaux inconnus
        #                     text = f"{class_name} inconnu"

        #                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        #                 cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Face Detection', frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if last_detected_encoding is not None:
                name = input("Entrez le nom du visage: ")
                if name:
                    references[name] = {
                        'type': 'human',
                        'encoding': last_detected_encoding.tolist()
                    }
                    save_references(references)
                    print(f"✓ Visage '{name}' sauvegardé!")
                    print(f"Visages stockés: {list(references.keys())}")
            # elif last_detected_hash is not None:
            #     name = input("Entrez le nom de l'animal: ")
            #     species = input("Espèce (cat/dog/bird/rabbit): ").lower()
            #     if name and species in ['cat', 'dog', 'bird', 'rabbit']:
            #         references[name] = {
            #             'type': 'animal',
            #             'hash': last_detected_hash,
            #             'species': species
            #         }
            #         save_references(references)
            #         print(f"✓ Animal '{name}' sauvegardé!")
            #         print(f"Visages/animaux stockés: {list(references.keys())}")
            #     else:
            #         print("✗ Espèce invalide")
            else:
                print("✗ Aucun visage détecté")

cap.release()
cv2.destroyAllWindows()
