import os
import streamlit as st
import cv2
import joblib
import mediapipe as mp
import numpy as np
import time
import warnings

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

# New input: Prediction Mode selection (retained for consistency)
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
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']

mp_hands = mp.solutions.hands
# Allow up to 2 hands so we can check if exactly one is present.
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
# Streamlit session state initialization
# --------------------
if 'result_string' not in st.session_state:
    st.session_state.result_string = ""
if 'current_letter' not in st.session_state:
    st.session_state.current_letter = None
if 'letter_start_time' not in st.session_state:
    st.session_state.letter_start_time = None
if 'hand_appeared_time' not in st.session_state:
    st.session_state.hand_appeared_time = None
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = 0  # Time when last letter was confirmed
if 'prev_frame_time' not in st.session_state:
    st.session_state.prev_frame_time = time.time()  # For FPS calculation

st.title("ASL Hand Sign Recognition")
st.write("Hold a sign to confirm the letter. Adjust the timings in the sidebar.")

# Create an image placeholder for the video feed
image_placeholder = st.empty()

# Create a placeholder for the result string (below the image)
result_placeholder = st.empty()

# Create a container for control buttons
button_container = st.container()
with button_container:
    clear_button = st.button("Clear String")
    backspace_button = st.button("Backspace")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS using session state timing
    current_time = time.time()
    frame_interval = current_time - st.session_state.prev_frame_time
    st.session_state.prev_frame_time = current_time
    fps = 1.0 / frame_interval if frame_interval > 0 else 0

    # Flip image for natural hand positioning (mirror view)
    frame = cv2.flip(frame, 1)

    # Check number of hands detected (using the flipped frame)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    num_hands = 0
    if results.multi_hand_landmarks is not None:
        num_hands = len(results.multi_hand_landmarks)

    if num_hands > 1:
        cv2.putText(frame, "Please show only one hand", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        st.session_state.current_letter = None
        st.session_state.letter_start_time = None
        st.session_state.hand_appeared_time = None
    else:
        # Extract landmarks only if exactly one hand is detected
        landmarks = extract_hand_landmarks(frame)
        if landmarks is not None:
            # Based on prediction mode, get probabilities
            if prediction_mode == "Aggregate":
                # In this snippet, we'll only use the flipped image
                # for prediction even in "Aggregate" mode.
                probs = mlp_model.predict_proba(landmarks)[0]
            else:
                probs = mlp_model.predict_proba(landmarks)[0]
            classes = mlp_model.classes_
            ranking = sorted(list(zip(classes, probs)), key=lambda x: x[1], reverse=True)
            top4 = ranking[:4]
            predicted_label = top4[0][0]

            # Draw the top prediction on the frame
            cv2.putText(frame, f'Prediction: {predicted_label}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Draw the top 4 predictions on the frame
            for i, (cls, prob) in enumerate(top4, start=1):
                text = f'Rank {i}: {cls} ({prob*100:.1f}%)'
                cv2.putText(frame, text, (10, 50 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Combined Confidence: {top4[0][1]*100:.1f}%', (10, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Letter-hold logic
            time_since_last = current_time - st.session_state.last_prediction_time
            if time_since_last < inter_letter_delay:
                remaining = inter_letter_delay - time_since_last
                cv2.putText(frame, f'Time until next letter: {remaining:.1f}s', (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                if st.session_state.current_letter is None or predicted_label != st.session_state.current_letter:
                    st.session_state.current_letter = predicted_label
                    st.session_state.letter_start_time = current_time
                else:
                    if current_time - st.session_state.letter_start_time >= letter_hold_time:
                        if predicted_label == "space":
                            st.session_state.result_string += " "
                        elif predicted_label == "del":
                            st.session_state.result_string = st.session_state.result_string[:-1]
                        else:
                            st.session_state.result_string += predicted_label
                        st.session_state.last_prediction_time = current_time
                        st.session_state.current_letter = None
                        st.session_state.letter_start_time = None

    # Overlay FPS in the top right corner
    height, width, _ = frame.shape
    cv2.putText(frame, f'FPS: {fps:.1f}', (width - 170, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Update the Streamlit image placeholder
    image_placeholder.image(frame, channels="BGR")

    # Update the result string display below the image
    result_placeholder.markdown(f"### Resulting Text: {st.session_state.result_string}")

    # Check for key events locally (for testing with cv2.waitKey)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Also support backspace using Delete key (127) if running locally.
    if key == 127:
        st.session_state.result_string = st.session_state.result_string[:-1]

# cap.release()
# cv2.destroyAllWindows()
# hands.close()
