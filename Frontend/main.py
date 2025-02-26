import streamlit as st
import cv2
import joblib
import mediapipe as mp
import numpy as np
import time
import warnings
import pandas as pd  # Import pandas for feature name handling

# Suppress warnings
warnings.simplefilter(action="ignore", category=UserWarning)

# Load the trained model
model_path = "Models/mlp_classifier_best_params.pkl"
mlp_model = joblib.load(model_path)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']  # Adjust based on your dataset labels

# Streamlit UI setup
st.title("ASL Alphabet Translator")
st.write("Show an ASL hand sign to the camera, and it will build a string!")

# Webcam input and predicted text placeholders
video = st.empty()
text_display = st.empty()

# Initialize state variables
predicted_string = ""
last_prediction = None
last_detected_time = None
waiting_for_empty = False

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
        
        # Ensure the extracted data is exactly 42 values (21 landmarks × 2 coordinates)
        if len(data) == 42:
            return np.array(data).reshape(1, -1)  # Reshape for model input

    return None

# OpenCV video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for natural hand positioning
    frame = cv2.flip(frame, 1)

    # Extract hand landmarks
    landmarks = extract_hand_landmarks(frame)
    current_time = time.time()

    if landmarks is not None:
        # Convert NumPy array to DataFrame to match training format
        landmarks_df = pd.DataFrame(landmarks, columns=[f"x{i//2}" if i % 2 == 0 else f"y{i//2}" for i in range(42)])

        # Make prediction
        prediction = mlp_model.predict(landmarks_df)[0]

        # Check if we should accept this letter
        if prediction != last_prediction:
            last_detected_time = current_time
            waiting_for_empty = False  # Reset waiting state

        elif last_prediction == prediction and (current_time - last_detected_time) >= 1.0 and not waiting_for_empty:
            if prediction == "del":
                predicted_string = predicted_string[:-1]  # Remove last letter
            elif prediction == "space":
                predicted_string += " "  # Add space
            else:
                predicted_string += prediction  # Add letter

            waiting_for_empty = True  # Now wait for an empty frame
            last_detected_time = current_time  # Reset timer

        last_prediction = prediction

        # Draw prediction text
        cv2.putText(frame, f'Prediction: {prediction}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # **Update Streamlit UI dynamically**
    text_display.subheader(f"Predicted Text: {predicted_string}")

    # Convert frame to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video.image(frame_rgb, channels="RGB")

    # Stop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
