import cv2
import numpy as np
import time
import pygame
import threading
import os
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# === SOUND ALERT SETUP ===
pygame.mixer.init()

# Store sound objects
alert_sounds = {
    1: pygame.mixer.Sound("alert1.wav"),
    2: pygame.mixer.Sound("alert2.wav")
}

def play_alert(level):
    stop_alert()
    if level in alert_sounds:
        alert_sounds[level].play(-1)

def stop_alert():
    pygame.mixer.stop()

# === CASCADE LOADING ===
cv_base = os.path.dirname(cv2.__file__)
face_cascade = cv2.CascadeClassifier(os.path.join(cv_base, 'data', 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cv_base, 'data', 'haarcascade_eye.xml'))

# === SETTINGS ===
EYE_BLACK_RATIO_THRESHOLD = 0.15
LEVEL_1_TIME = 0.3
LEVEL_2_TIME = 2.0
blink_threshold = 0.25

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

eye_closed_start = None
last_played_level = -1
blink_count = 0
eye_prev_closed = False
alert_playing = False

eye_status_history = deque(maxlen=100)  # 1 = closed, 0 = open

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
    eye_ratios = []
    eye_ratio = 0.0

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        closed_eyes = 0
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)
            closed, ratio = is_eye_closed(eye_img)
            eye_ratios.append(ratio)
            if closed:
                closed_eyes += 1

        if len(eyes) >= 2 and closed_eyes >= 2:
            eyes_closed = True
        break

    if eye_ratios:
        eye_ratio = sum(eye_ratios) / len(eye_ratios)

    current_time = time.time()

    if eyes_closed:
        eye_status_history.append(1)
        if eye_closed_start is None:
            eye_closed_start = current_time
        duration = current_time - eye_closed_start
        eye_prev_closed = True
    else:
        eye_status_history.append(0)
        if eye_closed_start is not None:
            blink_duration = current_time - eye_closed_start
            if blink_duration < blink_threshold:
                blink_count += 1
        eye_closed_start = None
        duration = 0
        eye_prev_closed = False

    if eyes_closed and duration >= LEVEL_2_TIME:
        level = 2
        status = "Drowsiness Detected!"
        color = (0, 0, 255)
    elif eyes_closed and duration >= LEVEL_1_TIME:
        level = 1
        status = "Stay Alert!"
        color = (0, 255, 255)
    elif eyes_closed:
        level = 0
        status = "Eyes Closed (Waiting...)"
        color = (0, 255, 0)
    else:
        level = 0
        status = "Awake"
        color = (0, 255, 0)

    if level in [1, 2]:
        if last_played_level != level:
            play_alert(level)
            alert_playing = True
    elif level == 0:
        stop_alert()
        alert_playing = False

    last_played_level = level

    # === DISPLAY TEXT ===
    cv2.putText(frame, f"Level {level}: {status}", (10, 30), font, 0.9, color, 2)
    cv2.putText(frame, f"Eye Ratio: {eye_ratio:.3f}", (10, 60), font, 0.7, color, 2)
    cv2.putText(frame, f"Closed Time: {duration:.2f}s", (10, 90), font, 0.7, color, 2)
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 120), font, 0.7, (255, 255, 0), 2)

    # === EMBED GRAPH ===
    fig, ax = plt.subplots(figsize=(5, 1))
    ax.fill_between(range(len(eye_status_history)), list(eye_status_history), color='red', alpha=0.6, step="mid", label='Closed')
    ax.fill_between(range(len(eye_status_history)), 0, np.where(np.array(eye_status_history)==0, 1, 0),
                    color='lime', alpha=0.6, step="mid", label='Open')
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    ax.set_title("Eye Status Timeline", fontsize=9, color='gray', pad=2)
    fig.tight_layout(pad=0)
    canvas = FigureCanvas(fig)
    canvas.draw()
    graph_img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    graph_img = cv2.resize(graph_img, (frame.shape[1], 100))
    combined = np.vstack((frame, graph_img))

    cv2.imshow("Driver Drowsiness Detection", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
