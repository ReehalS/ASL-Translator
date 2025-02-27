import streamlit as st
import cv2
import joblib
import mediapipe as mp
import numpy as np
import time
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)

# --------------------
# Load model and set up MediaPipe
# --------------------
model_path = "Models/mlp_classifier_best_params.pkl"
mlp_model = joblib.load(model_path)

# Class labels as per your dataset
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(image):
    """Extracts 21 hand landmarks from the given image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        data = []
        for landmark in landmarks.landmark:
            data.append(landmark.x)
            data.append(landmark.y)
        return np.array(data).reshape(1, -1)  # Reshape for model input
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

st.title("ASL Hand Sign Recognition")
st.write("Hold a sign for 1 second to confirm the letter. A 2 second delay is enforced between letters.")
image_placeholder = st.empty()
text_placeholder = st.empty()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    landmarks = extract_hand_landmarks(frame)
    current_time = time.time()

    # Check if we're within the 2-second waiting period after a confirmed prediction.
    if landmarks is not None and (current_time - st.session_state.last_prediction_time) < 2:
        cv2.putText(frame, "Time between characters", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    elif landmarks is not None:
        # Hand is detected and we're not in the waiting period.
        if st.session_state.hand_appeared_time is None:
            st.session_state.hand_appeared_time = current_time

        # Wait 0.75 seconds after a hand is detected before starting letter prediction.
        if current_time - st.session_state.hand_appeared_time >= 0.75:
            prediction = mlp_model.predict(landmarks)
            predicted_letter = prediction[0]

            # If this is a new prediction, reset the letter timer.
            if st.session_state.current_letter != predicted_letter:
                st.session_state.current_letter = predicted_letter
                st.session_state.letter_start_time = current_time
            else:
                # Check if the letter has been held for at least 0.75 seconds.
                if current_time - st.session_state.letter_start_time >= 0.75:
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
        # If no hand is detected, reset all relevant timers and prediction values.
        st.session_state.current_letter = None
        st.session_state.letter_start_time = None
        st.session_state.hand_appeared_time = None

    image_placeholder.image(frame, channels="BGR")
    text_placeholder.markdown(f"### Resulting Text: {st.session_state.result_string}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
