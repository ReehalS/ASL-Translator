import os
import time
import warnings
import threading

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

# Suppress PyAV warnings.
av.logging.set_level(av.logging.ERROR)
warnings.filterwarnings("ignore", message="Using NORM_RECT without IMAGE_DIMENSIONS")

st.title("ASL Hand Landmark & Letter Prediction Test")
st.write("This demo shows live hand landmark detection and letter prediction using MediaPipe.")

# Sidebar settings pane.
st.sidebar.header("Settings")
hand_detect_delay = st.sidebar.number_input("Hand detection delay (s)", value=0.5, min_value=0.1, step=0.1)
min_hold_time = st.sidebar.number_input("Minimum hold time (s)", value=0.75, min_value=0.1, step=0.05)
inter_letter_gap = st.sidebar.number_input("Inter-letter gap (s)", value=2.0, min_value=0.1, step=0.1)

# New: Model selection
models_folder = "Models"
available_models = [f for f in os.listdir(models_folder) if f.endswith(".pkl")]
default_model = "multi_layer_perceptron_best.pkl" if "multi_layer_perceptron_best.pkl" in available_models else available_models[0]
selected_model = st.sidebar.selectbox("Choose Model", available_models, index=available_models.index(default_model))

# Load the chosen model.
model_path = os.path.join(models_folder, selected_model)
mlp_model = joblib.load(model_path)

# New: Prediction mode selection.
prediction_mode = st.sidebar.selectbox("Prediction Mode", ["Aggregate (Dual)", "Flipped Only"], index=0)

# Note: To improve video quality, try increasing resolution.
# For example: {"video": {"width": 1280, "height": 720}, "audio": False}

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

# A thread-safe container to hold our shared values.
lock = threading.Lock()
letter_container = {
    "predicted_letter": "None",
    "letter_string": "",              # The built string of letters.
    "hand_first_detected_time": None, # Time when a hand is first detected.
    "letter_hold_start_time": None,   # Time when the candidate letter began to be held.
    "current_letter_candidate": None, # The current letter being held.
    "last_letter_added_time": 0,      # Time when a letter was last appended.
    "prev_time": time.time()          # For FPS calculation.
}

def get_landmarks(img):
    """Given an image, process with MediaPipe and extract landmarks if exactly one hand is detected."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    landmarks_data = None
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]
        data = []
        for landmark in hand_landmarks.landmark:
            data.append(landmark.x)
            data.append(landmark.y)
        if len(data) == 42:
            landmarks_data = np.array(data).reshape(1, -1)
    return landmarks_data, results

def callback(frame):
    current_time = time.time()
    # Get the original image from the frame.
    original_img = frame.to_ndarray(format="bgr24")
    # Create a flipped image for mirror view.
    flipped_img = cv2.flip(original_img, 1)
    
    # Process images depending on prediction mode.
    if prediction_mode == "Aggregate (Dual)":
        landmarks_flipped, results_flipped = get_landmarks(flipped_img)
        landmarks_original, results_original = get_landmarks(original_img)
    else:  # "Flipped Only"
        landmarks_flipped, results_flipped = get_landmarks(flipped_img)
        landmarks_original, results_original = None, None  # ignore original
    
    # For drawing, we use the flipped image.
    img = flipped_img.copy()
    
    # Draw landmarks (prefer the flipped result; if not, use original)
    if results_flipped and results_flipped.multi_hand_landmarks:
        for hand_landmarks in results_flipped.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    elif results_original and results_original.multi_hand_landmarks:
        temp = original_img.copy()
        for hand_landmarks in results_original.multi_hand_landmarks:
            mp_drawing.draw_landmarks(temp, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        img = cv2.flip(temp, 1)
    
    with lock:
        if (results_flipped and results_flipped.multi_hand_landmarks) or (results_original and results_original.multi_hand_landmarks):
            # Set the time when the hand first appears.
            if letter_container["hand_first_detected_time"] is None:
                letter_container["hand_first_detected_time"] = current_time
            
            # Only start letter evaluation after the hand detection delay.
            if current_time - letter_container["hand_first_detected_time"] >= hand_detect_delay:
                combined_probs = None
                used_prediction = False
                # Compute prediction on flipped image if landmarks exist.
                if landmarks_flipped is not None:
                    probs_flipped = mlp_model.predict_proba(landmarks_flipped)[0]
                    combined_probs = probs_flipped
                    used_prediction = True
                # If using aggregate mode, compute prediction on original image.
                if prediction_mode == "Aggregate (Dual)" and landmarks_original is not None:
                    probs_original = mlp_model.predict_proba(landmarks_original)[0]
                    if combined_probs is not None:
                        combined_probs += probs_original
                    else:
                        combined_probs = probs_original
                    used_prediction = True
                if used_prediction and combined_probs is not None:
                    classes = mlp_model.classes_
                    # Determine the letter with the highest overall probability.
                    idx = np.argmax(combined_probs)
                    predicted_letter = classes[idx]
                else:
                    predicted_letter = "None"
                    
                letter_container["predicted_letter"] = predicted_letter

                if predicted_letter != "None":
                    # If the candidate changes, reset the hold timer.
                    if letter_container["current_letter_candidate"] != predicted_letter:
                        letter_container["current_letter_candidate"] = predicted_letter
                        letter_container["letter_hold_start_time"] = current_time
                    else:
                        # If held long enough (>=min_hold_time) and sufficient gap (>=inter_letter_gap)
                        if (letter_container["letter_hold_start_time"] is not None and
                            current_time - letter_container["letter_hold_start_time"] >= min_hold_time and
                            current_time - letter_container["last_letter_added_time"] >= inter_letter_gap):
                            
                            if predicted_letter.lower() == "space":
                                letter_container["letter_string"] += " "
                            elif predicted_letter.lower() == "del":
                                # Remove the last character if the string is not empty.
                                letter_container["letter_string"] = letter_container["letter_string"][:-1]
                            else:
                                letter_container["letter_string"] += predicted_letter
                            
                            letter_container["last_letter_added_time"] = current_time
                else:
                    # No valid prediction.
                    letter_container["current_letter_candidate"] = None
                    letter_container["letter_hold_start_time"] = None
        else:
            # Reset detection timers when no hand is present.
            letter_container["hand_first_detected_time"] = None
            letter_container["letter_hold_start_time"] = None
            letter_container["current_letter_candidate"] = None
            letter_container["predicted_letter"] = "None"
    
    # FPS calculation.
    with lock:
        dt = current_time - letter_container.get("prev_time", current_time)
        fps = 1.0 / dt if dt > 0 else 0.0
        letter_container["prev_time"] = current_time

    # Overlay the prediction, built string, and FPS on the video feed.
    cv2.putText(img, f'Prediction: {letter_container["predicted_letter"]}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f'String: {letter_container["letter_string"]}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # Position the FPS text at the top right.
    img_width = img.shape[1]
    cv2.putText(img, f'FPS: {fps:.2f}', (img_width - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Show time left before the next letter can be added.
    time_since_last = current_time - letter_container["last_letter_added_time"]
    if time_since_last < inter_letter_gap:
        time_left = inter_letter_gap - time_since_last
        cv2.putText(img, f"Time left: {time_left:.2f}s", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Return the annotated frame.
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the WebRTC stream using the callback.
ctx = webrtc_streamer(
    key="asl_letter_prediction",
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False}
)

# Display the current predicted letter and the built string on the main interface.
string_placeholder = st.empty()
letter_placeholder = st.empty()

while ctx.state.playing:
    with lock:
        current_letter = letter_container["predicted_letter"]
        current_string = letter_container["letter_string"]
    letter_placeholder.markdown(f"Current Predicted Letter **{current_letter}**")
    string_placeholder.markdown(f"## Current String: {current_string}")
    time.sleep(0.1)
