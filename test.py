import sqlite3
from datetime import datetime
import face_recognition
import cv2
import numpy as np
import os
import pickle

# Paths
dataset_path = "faces/Faces"
dataset_csv = "faces/Dataset.csv"
cache_file = "faces/encodings_cache.pkl"
db_file = "attendance.db"

# Initialize the database
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create the attendance table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
''')
conn.commit()

# Initialize known encodings and names
known_encodings = []
known_face_names = []

# Load from cache if available
if os.path.exists(cache_file):
    print("Loading encodings from cache...")
    with open(cache_file, "rb") as f:
        cache_data = pickle.load(f)
        known_encodings = cache_data["encodings"]
        known_face_names = cache_data["names"]
else:
    print("Cache not found. Generating encodings...")
    if not os.path.exists(dataset_csv):
        print(f"Error: Dataset CSV file not found: {dataset_csv}")
        exit()

    dataset_df = pd.read_csv(dataset_csv)
    for index, row in dataset_df.iterrows():
        image_file = row["id"]
        person_name = row["label"]
        image_path = os.path.join(dataset_path, image_file)

        if os.path.exists(image_path):
            try:
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_encodings.append(encoding)
                known_face_names.append(person_name)
            except IndexError:
                print(f"Face not detected in image: {image_path}")
        else:
            print(f"Image file not found: {image_path}")

    with open(cache_file, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_face_names}, f)
    print("Encodings saved to cache.")

# Start video capture
video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Default name for unknown faces
        name = "Unknown"

        # Match the face encoding with known encodings
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < 0.4:
                name = known_face_names[best_match_index]

        # Mark attendance
        if name != "Unknown":
            now = datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")

            # Check if the name is already marked for today
            cursor.execute("""
                SELECT 1 FROM attendance 
                WHERE name = ? AND timestamp LIKE ? 
            """, (name, f"{current_date}%"))
            record_exists = cursor.fetchone()

            if not record_exists:
                # Insert new attendance record
                cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, current_time))
                conn.commit()
                print(f"{name} marked present at {current_time}")

        # Draw a rectangle around the face (red color)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face (white text)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
conn.close()