# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import cv2
# import numpy as np
# from deepface import DeepFace
# from numpy.linalg import norm


# MODEL_NAME = "ArcFace"
# DETECTOR = "retinaface"
# THRESHOLD = 0.68   


# FACE_DB = {
#     "Prashant": {
#         "images": [
#             "backend/images/prashant.jpeg",
            
#         ],
#         "flat": "A-101",
#         "building": "Krishna Heights"
#     },
#     "Virat Kohli": {
#         "images": [
#             "backend/images/virat.png"
#         ],
#         "flat": "B-202",
#         "building": "Royal Residency"
#     }
# }


# def cosine_distance(a, b):
#     return 1 - np.dot(a, b) / (norm(a) * norm(b))


# print("⚙️ Generating face embeddings...")
# FACE_EMBEDDINGS = []

# for name, data in FACE_DB.items():
#     for img in data["images"]:
#         emb = DeepFace.represent(
#             img_path=img,
#             model_name=MODEL_NAME,
#             detector_backend=DETECTOR,
#             enforce_detection=True
#         )[0]["embedding"]

#         FACE_EMBEDDINGS.append({
#             "name": name,
#             "flat": data["flat"],
#             "building": data["building"],
#             "embedding": emb
#         })

# print("✅ Face database ready")


# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

# print("📸 SMART AI CAMERA LIVE | Press 'q' to exit")


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     display = frame.copy()

#     try:
#         faces = DeepFace.extract_faces(
#             img_path=frame,
#             detector_backend=DETECTOR,
#             enforce_detection=False
#         )
#     except:
#         faces = []

#     for face in faces:
#         x = face["facial_area"]["x"]
#         y = face["facial_area"]["y"]
#         w = face["facial_area"]["w"]
#         h = face["facial_area"]["h"]
#         live_face = face["face"]

       
#         live_embedding = DeepFace.represent(
#             img_path=live_face,
#             model_name=MODEL_NAME,
#             detector_backend="skip",
#             enforce_detection=False
#         )[0]["embedding"]

#         best_match = None
#         best_distance = 1.0

#         for ref in FACE_EMBEDDINGS:
#             dist = cosine_distance(live_embedding, ref["embedding"])
#             if dist < best_distance:
#                 best_distance = dist
#                 best_match = ref

#         if best_match and best_distance < THRESHOLD:
#             label = f"✅ {best_match['name']} | Flat {best_match['flat']}"
#             color = (0, 255, 0)
#         else:
#             label = "❌ UNKNOWN FACE"
#             color = (0, 0, 255)

#         cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(display, label, (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.imshow("SMART AI GATE SYSTEM", display)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from deepface import DeepFace
from numpy.linalg import norm
import pyttsx3
import time

# -------------------- SETTINGS --------------------
MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
THRESHOLD = 0.68
ANNOUNCE_DELAY = 1.2  # seconds between announcements

# -------------------- FACE DATABASE --------------------
FACE_DB = {
    "Prashant": {
        "images": ["backend/images/prashant.jpeg"],
        "flat": "A-101",
        "building": "Krishna Heights"
    },
    "Virat Kohli": {
        "images": ["backend/images/virat.png"],
        "flat": "B-202",
        "building": "Royal Residency"
    }
}

# -------------------- UTILITY --------------------
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

# -------------------- VOICE ENGINE --------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

# -------------------- GENERATE FACE EMBEDDINGS --------------------
print("⚙️ Generating face embeddings...")
FACE_EMBEDDINGS = []

for name, data in FACE_DB.items():
    for img in data["images"]:
        emb = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True
        )[0]["embedding"]

        FACE_EMBEDDINGS.append({
            "name": name,
            "flat": data["flat"],
            "building": data["building"],
            "embedding": emb
        })

print("✅ Face database ready")

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("📸 SMART AI CAMERA LIVE | Press 'q' to exit")

# -------------------- ANNOUNCEMENT CONTROL --------------------
announced_identities = set()
announcement_queue = []
last_announce_time = 0

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()

    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR,
            enforce_detection=False
        )
    except:
        faces = []

    current_identities = set()

    for face in faces:
        x = face["facial_area"]["x"]
        y = face["facial_area"]["y"]
        w = face["facial_area"]["w"]
        h = face["facial_area"]["h"]
        live_face = face["face"]

        live_embedding = DeepFace.represent(
            img_path=live_face,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False
        )[0]["embedding"]

        best_match = None
        best_distance = 1.0

        for ref in FACE_EMBEDDINGS:
            dist = cosine_distance(live_embedding, ref["embedding"])
            if dist < best_distance:
                best_distance = dist
                best_match = ref

        # -------------------- IDENTIFICATION --------------------
        if best_match and best_distance < THRESHOLD:
            identity = best_match["name"]
            label = f"{best_match['name']} | Flat {best_match['flat']}"
            color = (0, 255, 0)
            announcement = f"This is {best_match['name']}"
        else:
            identity = "unknown"
            label = "❌ UNKNOWN FACE"
            color = (0, 0, 255)
            announcement = "Unknown face detected"

        current_identities.add(identity)

        # -------------------- QUEUE ANNOUNCEMENT --------------------
        if identity not in announced_identities:
            announcement_queue.append(announcement)
            announced_identities.add(identity)

        # -------------------- DRAW --------------------
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        cv2.putText(display, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # -------------------- VOICE QUEUE (NO OVERLAP) --------------------
    current_time = time.time()
    if announcement_queue and (current_time - last_announce_time) > ANNOUNCE_DELAY:
        msg = announcement_queue.pop(0)
        engine.say(msg)
        engine.runAndWait()
        last_announce_time = current_time

    # -------------------- RESET WHEN FACE LEAVES --------------------
    if len(faces) == 0:
        announced_identities.clear()
        announcement_queue.clear()

    cv2.imshow("SMART AI GATE SYSTEM", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- CLEANUP --------------------
cap.release()
cv2.destroyAllWindows()
