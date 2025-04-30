import os
import face_recognition
import sqlite3
import numpy as np

# Constants
FACE_FOLDER = "./student_faces"
DB_FILE = "students.db"

# Connect to SQLite and create the student table
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    class_name TEXT NOT NULL,
    image_file TEXT NOT NULL,
    encoding BLOB NOT NULL,
    absence_count INTEGER DEFAULT 0,
    dropped INTEGER DEFAULT 0
)
''')
conn.commit()

# Process each .jpg file
for filename in os.listdir(FACE_FOLDER):
    if filename.lower().endswith(".jpg"):
        parts = filename[:-4].split("_")
        if len(parts) >= 2:
            name = parts[0].capitalize() + " " + parts[1].capitalize()
            class_name = parts[2] if len(parts) > 2 else "Unknown"
        else:
            continue  # Skip improperly named files

        image_path = os.path.join(FACE_FOLDER, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            print(f"⚠️ No face found in {filename}. Skipping.")
            continue

        # Save the first detected face
        encoding_blob = np.array(encodings[0]).tobytes()

        cursor.execute('''
        INSERT INTO students (name, class_name, image_file, encoding)
        VALUES (?, ?, ?, ?)
        ''', (name, class_name, filename, encoding_blob))

        print(f"✅ Registered: {name} ({class_name})")

conn.commit()
conn.close()
