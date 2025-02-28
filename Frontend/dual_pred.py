import os
import time
import warnings

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

warnings.simplefilter(action="ignore", category=UserWarning)

# --------------------
# Sidebar: Parameter and Model Selection
# --------------------
st.sidebar.header("Settings")

# Numeric inputs for various delays (in seconds)
hand_detection_delay = st.sidebar.number_input(
    "Time after hand appears to start detection", min_value=0.0, value=0.75, step=0.1
)
letter_hold_time = st.sidebar.number_input(
    "Time letter must be held for confirmation", min_value=0.0, value=0.75, step=0.1
)
inter_letter_delay = st.sidebar.number_input(
    "Delay between letters", min_value=0.0, value=2.0, step=0.1
)

# Prediction mode selection
prediction_mode = st.sidebar.selectbox(
    "Prediction Mode", options=["Aggregate", "Flipped Only"], index=0
)

# Dropdown for model selection from the Models folder
models_dir = "Models"
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
default_model = "multi_layer_perceptron_best_params.pkl"
if default_model not in model_files:
    default_model = model_files[0] if model_files else None

selected_model = st.sidebar.selectbox(
    "Select Model", options=model_files, index=model_files.index(default_model) if default_model else 0
)
model_path = os.path.join(models_dir, selected_model)
mlp_model = joblib.load(model_path)

# --------------------
# Set up MediaPipe and other initializations
# --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(image):
    """
    Extracts 21 hand landmarks from the given image if exactly one hand is visible.
    Returns a NumPy array (1x42) for model input, or None otherwise.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) == 1:
        landmarks = results.multi_hand_landmarks[0]
        data = []
        for landmark in landmarks.landmark:
            data.append(landmark.x)
            data.append(landmark.y)
        if len(data) == 42:
            return np.array(data).reshape(1, -1)
    return None

# --------------------
# Initialize Streamlit session state variables
# --------------------
if 'result_string' not in st.session_state:
    st.session_state.result_string = ""
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = 0  # Time when last letter was confirmed

# Buttons to clear or backspace the result string
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear String"):
        st.session_state.result_string = ""
with col2:
    if st.button("Backspace"):
        st.session_state.result_string = st.session_state.result_string[:-1]

st.title("ASL Hand Sign Recognition")
st.write("Hold a sign to confirm the letter. Adjust the timings in the sidebar.")

# --------------------
# Global timing variables for the callback
# --------------------
global_prev_frame_time = time.time()
current_letter = None
letter_start_time = None

def callback(frame):
    global global_prev_frame_time, current_letter, letter_start_time
    print("hello")
    # Convert frame to a numpy array
    img = frame.to_ndarray(format="bgr24")
    current_time = time.time()
    frame_interval = current_time - global_prev_frame_time
    global_prev_frame_time = current_time
    fps = 1.0 / frame_interval if frame_interval > 0 else 0

    # Flip for mirror view
    img = cv2.flip(img, 1)

    # Process with MediaPipe
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    if num_hands > 1:
        cv2.putText(img, "Please show only one hand", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        current_letter = None
        letter_start_time = None
    else:
        landmarks = extract_hand_landmarks(img)
        if landmarks is not None:
            # Use the same prediction logic for both modes in this example.
            if prediction_mode == "Aggregate":
                probs = mlp_model.predict_proba(landmarks)[0]
            else:
                probs = mlp_model.predict_proba(landmarks)[0]
            classes = mlp_model.classes_
            ranking = sorted(list(zip(classes, probs)), key=lambda x: x[1], reverse=True)
            top4 = ranking[:4]
            predicted_label = top4[0][0]
            print(predicted_label)
            # Draw prediction results
            cv2.putText(img, f'Prediction: {predicted_label}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            for i, (cls, prob) in enumerate(top4, start=1):
                text = f'Rank {i}: {cls} ({prob*100:.1f}%)'
                cv2.putText(img, text, (10, 50 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f'Combined Confidence: {top4[0][1]*100:.1f}%', (10, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Letter-hold logic with inter-letter delay
            time_since_last = current_time - st.session_state.last_prediction_time
            if time_since_last < inter_letter_delay:
                remaining = inter_letter_delay - time_since_last
                cv2.putText(img, f'Time until next letter: {remaining:.1f}s', (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                if current_letter is None or predicted_label != current_letter:
                    current_letter = predicted_label
                    letter_start_time = current_time
                else:
                    if current_time - letter_start_time >= letter_hold_time:
                        if predicted_label == "space":
                            st.session_state.result_string += " "
                        elif predicted_label == "del":
                            st.session_state.result_string = st.session_state.result_string[:-1]
                        else:
                            st.session_state.result_string += predicted_label
                        st.session_state.last_prediction_time = current_time
                        current_letter = None
                        letter_start_time = None

    # Overlay FPS and the resulting text
    height, width, _ = img.shape
    cv2.putText(img, f'FPS: {fps:.1f}', (width - 170, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f'Text: {st.session_state.result_string}', (10, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the WebRTC stream with the callback
# webrtc_streamer(key="asl_app", video_frame_callback=callback)
webrtc_streamer(
    key="asl_app",
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)