import cv2
import numpy as np
import time
import pygame
import threading
import os

# === SOUND ALERT SETUP ===
pygame.mixer.init()

# Function to play alert sound in a separate thread (non-blocking)
def play_alert(file):
    def _play():
        try:
            sound = pygame.mixer.Sound(file)
            sound.play()
        except Exception as e:
            print("Sound error:", e)
    threading.Thread(target=_play, daemon=True).start()

# === CASCADE LOADING ===
# Load Haar cascade classifiers for face and eye detection
cv_base = os.path.dirname(cv2.__file__)
face_cascade = cv2.CascadeClassifier(os.path.join(cv_base, 'data', 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cv_base, 'data', 'haarcascade_eye.xml'))

# === SETTINGS ===
EYE_BLACK_RATIO_THRESHOLD = 0.25  # Threshold for how dark an eye must be to be considered closed
LEVEL_1_TIME = 0.3                # Seconds of closure before warning
LEVEL_2_TIME = 2.0                # Seconds of closure before critical alert

eye_closed_start = None          # Time when eyes were first detected closed
last_played_level = -1           # Track last played alert level to avoid repeating

cap = cv2.VideoCapture(0)        # Open webcam
font = cv2.FONT_HERSHEY_SIMPLEX  # Font for on-screen text

# Function to determine if an eye is closed based on darkness ratio
def is_eye_closed(eye_img, debug=False):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)        # Convert to grayscale
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)  # Threshold the image
    black_pixels = np.sum(thresh == 0)                      # Count black pixels
    total_pixels = thresh.size
    black_ratio = black_pixels / total_pixels               # Calculate black ratio

    if debug:
        cv2.imshow("Eye Debug", thresh)
        print("Black ratio:", black_ratio)

    return black_ratio > EYE_BLACK_RATIO_THRESHOLD, black_ratio

# === MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)   # Mirror the frame for natural interaction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    eyes_closed = False   # Assume eyes are open
    eye_ratio = 0         # Store latest eye darkness ratio

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]        # Region of interest in grayscale
        roi_color = frame[y:y+h, x:x+w]      # ROI in color for display

        eyes = eye_cascade.detectMultiScale(roi_gray)  # Detect eyes in face ROI
        closed_eyes = 0

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]  # Crop eye image
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)  # Draw rectangle
            closed, ratio = is_eye_closed(eye_img)   # Check if this eye is closed
            eye_ratio = ratio   # Save ratio for display
            if closed:
                closed_eyes += 1

        #  FIXED LOGIC: Even one closed eye counts as "eyes closed"
        if closed_eyes >= 1:
            eyes_closed = True
        break  # Only process the first detected face

    current_time = time.time()

    # === EYE STATUS LOGIC ===
    if eyes_closed:
        if eye_closed_start is None:
            eye_closed_start = current_time  # Mark the start time of eye closure
        duration = current_time - eye_closed_start

        if duration >= LEVEL_2_TIME:
            level = 2
            status = "Drowsiness Detected!"  # Critical alert
            color = (0, 0, 255)
        elif duration >= LEVEL_1_TIME:
            level = 1
            status = "Stay Alert!"           # Mild warning
            color = (0, 255, 255)
        else:
            level = 0
            status = "Eyes Closed (Waiting...)"  # Not long enough to alert
            color = (0, 255, 0)
    else:
        eye_closed_start = None  # Reset if eyes are open again
        duration = 0
        level = 0
        status = "Awake"
        color = (0, 255, 0)

    # === SOUND ALERT CONTROL ===
    if level == 2 and last_played_level != 2:
        play_alert("alert2.wav")  # Play critical alert
        last_played_level = 2
    elif level == 1 and last_played_level != 1:
        play_alert("alert1.wav")  # Play mild alert
        last_played_level = 1
    elif level == 0:
        last_played_level = 0     # Reset to no alert

    # === DISPLAY INFO ON SCREEN ===
    cv2.putText(frame, f"Level {level}: {status}", (10, 30), font, 0.9, color, 2)
    cv2.putText(frame, f"Eye Ratio: {eye_ratio:.3f}", (10, 60), font, 0.7, color, 2)
    cv2.putText(frame, f"Closed Time: {duration:.2f}s", (10, 90), font, 0.7, color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
