import cv2
import numpy as np
import time
import pygame
import threading
import os

# === SOUND ALERT SETUP ===
pygame.mixer.init()

def play_alert(file):
    def _play():
        try:
            sound = pygame.mixer.Sound(file)
            sound.play()
        except Exception as e:
            print("Sound error:", e)
    threading.Thread(target=_play, daemon=True).start()

# === CASCADE LOADING ===
cv_base = os.path.dirname(cv2.__file__)
face_cascade = cv2.CascadeClassifier(os.path.join(cv_base, 'data', 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cv_base, 'data', 'haarcascade_eye.xml'))

# === SETTINGS ===
EYE_BLACK_RATIO_THRESHOLD = 0.25  # Adjust based on ambient light
LEVEL_1_TIME = 0.3                # Seconds to trigger level 1
LEVEL_2_TIME = 2.0                # Seconds to trigger level 2

eye_closed_start = None
last_played_level = -1

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

def is_eye_closed(eye_img, debug=False):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    black_pixels = np.sum(thresh == 0)
    total_pixels = thresh.size
    black_ratio = black_pixels / total_pixels

    if debug:
        cv2.imshow("Eye Debug", thresh)
        print("Black ratio:", black_ratio)

    return black_ratio > EYE_BLACK_RATIO_THRESHOLD, black_ratio

# === MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_closed = False
    eye_ratio = 0

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        closed_eyes = 0

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)
            closed, ratio = is_eye_closed(eye_img)
            eye_ratio = ratio
            if closed:
                closed_eyes += 1

        # ✅ FIXED LOGIC: Even one closed eye counts
        if closed_eyes >= 1:
            eyes_closed = True
        break  # process only one face

    current_time = time.time()

    if eyes_closed:
        if eye_closed_start is None:
            eye_closed_start = current_time
        duration = current_time - eye_closed_start

        if duration >= LEVEL_2_TIME:
            level = 2
            status = "Drowsiness Detected!"
            color = (0, 0, 255)
        elif duration >= LEVEL_1_TIME:
            level = 1
            status = "Stay Alert!"
            color = (0, 255, 255)
        else:
            level = 0
            status = "Eyes Closed (Waiting...)"
            color = (0, 255, 0)
    else:
        eye_closed_start = None
        duration = 0
        level = 0
        status = "Awake"
        color = (0, 255, 0)

    # === SOUND ALERT CONTROL ===
    if level == 2 and last_played_level != 2:
        play_alert("alert2.wav")
        last_played_level = 2
    elif level == 1 and last_played_level != 1:
        play_alert("alert1.wav")
        last_played_level = 1
    elif level == 0:
        last_played_level = 0

    # === DISPLAY TEXT ===
    cv2.putText(frame, f"Level {level}: {status}", (10, 30), font, 0.9, color, 2)
    cv2.putText(frame, f"Eye Ratio: {eye_ratio:.3f}", (10, 60), font, 0.7, color, 2)
    cv2.putText(frame, f"Closed Time: {duration:.2f}s", (10, 90), font, 0.7, color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
