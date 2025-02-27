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

# Dropdown for model selection from the Models folder
models_dir = "Models"
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
default_model = "multi_layer_perceptron_best.pkl"
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
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(image):
    """
    Extracts 21 hand landmarks from the given image.
    Returns a NumPy array (1x42) for model input if landmarks are detected, otherwise None.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        data = []
        for landmark in landmarks.landmark:
            data.append(landmark.x)
            data.append(landmark.y)
        # Ensure we have 42 values (21 landmarks x 2 coordinates)
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
    st.session_state.prev_frame_time = time.time()  # For framerate calculation

st.title("ASL Hand Sign Recognition")
st.write("Hold a sign to confirm the letter. Adjust the timings in the sidebar.")

# Create placeholders for the image, recognized text, and ranked predictions
image_placeholder = st.empty()
text_placeholder = st.empty()
ranked_placeholder = st.empty()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Compute frame interval for framerate display
    current_time = time.time()
    frame_interval = current_time - st.session_state.prev_frame_time
    st.session_state.prev_frame_time = current_time
    fps = 1.0 / frame_interval if frame_interval > 0 else 0

    # Flip the frame horizontally (mirror view)
    frame = cv2.flip(frame, 1)
    landmarks = extract_hand_landmarks(frame)
    
    # Check if we're within the inter-letter waiting period.
    if landmarks is not None and (current_time - st.session_state.last_prediction_time) < inter_letter_delay:
        cv2.putText(frame, "Time between characters", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    elif landmarks is not None:
        # Get probability estimates from the model
        probs = mlp_model.predict_proba(landmarks)[0]
        classes = mlp_model.classes_
        ranked = list(zip(classes, probs))
        ranked.sort(key=lambda x: x[1], reverse=True)
        top5 = ranked[:5]
        ranked_text = "### Top Predictions:\n"
        for i, (cls, prob) in enumerate(top5, start=1):
            ranked_text += f"Rank {i}: {cls} ({prob*100:.1f}%)\n"
        ranked_placeholder.markdown(ranked_text)

        # Hand is detected and we're not in the waiting period.
        if st.session_state.hand_appeared_time is None:
            st.session_state.hand_appeared_time = current_time

        if current_time - st.session_state.hand_appeared_time >= hand_detection_delay:
            prediction = mlp_model.predict(landmarks)
            predicted_letter = prediction[0]
            # If this is a new prediction, reset the letter timer.
            if st.session_state.current_letter != predicted_letter:
                st.session_state.current_letter = predicted_letter
                st.session_state.letter_start_time = current_time
            else:
                if current_time - st.session_state.letter_start_time >= letter_hold_time:
                    if predicted_letter == "space":
                        st.session_state.result_string += " "
                    elif predicted_letter == "del":
                        st.session_state.result_string = st.session_state.result_string[:-1]
                    else:
                        st.session_state.result_string += predicted_letter

                    st.session_state.last_prediction_time = current_time
                    st.session_state.current_letter = None
                    st.session_state.letter_start_time = None
                    st.session_state.hand_appeared_time = None

            cv2.putText(frame, f'Prediction: {predicted_letter}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        # No hand detected: reset timers and prediction
        st.session_state.current_letter = None
        st.session_state.letter_start_time = None
        st.session_state.hand_appeared_time = None
        ranked_placeholder.markdown("### No hand detected.")

    # Overlay the framerate on the frame
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    image_placeholder.image(frame, channels="BGR")
    text_placeholder.markdown(f"### Resulting Text: {st.session_state.result_string}")

    # Exit loop if 'q' is pressed (works when running locally)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
# cv2.destroyAllWindows()
# hands.close()
