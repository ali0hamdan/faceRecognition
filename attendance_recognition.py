import cv2
import face_recognition
import sqlite3
import numpy as np
import os
import datetime
import csv
import time
import pyttsx3
import mediapipe as mp
import threading

# === CONFIG ===
CURRENT_CLASS = input("📚 Enter class name for this session: ").strip().lower()
DB_FILE = "students.db"
LOG_FOLDER = "attendance_logs"
TIMEOUT_SECONDS = 15
ENABLE_PRIVACY_MASK = False  # 🔒 Disable blurring

engine = pyttsx3.init()

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Ensure logs folder exists
os.makedirs(LOG_FOLDER, exist_ok=True)

def speak(text):
    threading.Thread(target=lambda: [engine.say(text), engine.runAndWait()]).start()

def load_students_from_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, class_name, encoding, absence_count, dropped FROM students")
    students = []
    for row in cursor.fetchall():
        student_id, name, class_name, encoding_blob, absences, dropped = row
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        students.append({
            'id': student_id,
            'name': name,
            'class': class_name,
            'encoding': encoding,
            'absences': absences,
            'dropped': dropped
        })
    conn.close()
    return students

def mark_attendance(student_id, name, status):
    date_str = datetime.date.today().isoformat()
    log_file = os.path.join(LOG_FOLDER, f"attendance_{date_str}.csv")

    try:
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([student_id, name, status, datetime.datetime.now().strftime("%H:%M:%S")])
    except PermissionError:
        print(f"❌ Cannot write to {log_file}. Close it if it's open in Excel.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if status == "Absent":
        cursor.execute("UPDATE students SET absence_count = absence_count + 1 WHERE id = ?", (student_id,))
        cursor.execute("UPDATE students SET dropped = 1 WHERE id = ? AND absence_count >= 9", (student_id,))
    conn.commit()
    conn.close()

def recognize_faces():
    known_students = load_students_from_db()
    known_encodings = [s['encoding'] for s in known_students if s['class'] == CURRENT_CLASS and not s['dropped']]
    target_students = [s for s in known_students if s['class'] == CURRENT_CLASS and not s['dropped']]

    print("📷 Starting webcam. Press 'q' to quit.")
    video = cv2.VideoCapture(0)

    scanned_ids = set()

    for student in target_students:
        print(f"🎤 Calling: {student['name']}... Please come scan your face!")
        speak(f"{student['name']}, please come scan your face.")

        start_time = time.time()
        match_found = False

        while time.time() - start_time < TIMEOUT_SECONDS:
            ret, frame = video.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_rgb = cv2.resize(rgb_frame, (320, 240))
            mesh_results = face_mesh.process(small_rgb)
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if not face_encodings:
                cv2.imshow("Attendance Scanner", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    video.release()
                    cv2.destroyAllWindows()
                    return
                continue

            face_encoding = face_encodings[0]
            location = face_locations[0]
            matches = face_recognition.compare_faces([student['encoding']], face_encoding)

            if matches[0]:
                name = student['name']
                msg = f"✅ Welcome, {name}!"
                mark_attendance(student['id'], name, "Present")
                speak(f"Welcome, {name}")
                print(msg)

                top, right, bottom, left = location
                cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                start_pause = time.time()
                while time.time() - start_pause < 2:
                    cv2.imshow("Attendance Scanner", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        video.release()
                        cv2.destroyAllWindows()
                        return
                match_found = True
                break

            cv2.imshow("Attendance Scanner", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video.release()
                cv2.destroyAllWindows()
                return

        if not match_found:
            print(f"❌ No match found for {student['name']} in time.")
            speak(f"{student['name']} was marked absent.")
            mark_attendance(student['id'], student['name'], "Absent")

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
