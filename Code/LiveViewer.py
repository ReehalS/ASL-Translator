import cv2
import joblib
import mediapipe as mp
import numpy as np
import time

# Load the trained model
model_path = "Models/multi_layer_perceptron_best.pkl"
mlp_model = joblib.load(model_path)

# Initialize MediaPipe Hands (allow up to 2 hands to check for one-hand condition)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']

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

# Open webcam
cap = cv2.VideoCapture(0)
prev_time = time.time()

# Parameters
inter_letter_delay = 2.0        # 2-second delay between letters
LETTER_HOLD_THRESHOLD = 0.75    # seconds required to confirm a letter

# Variables for accumulating predicted letters and timing
result_string = ""
current_letter = None
letter_start_time = None
last_letter_time = 0  # Time when the last letter was appended

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for natural hand positioning
    frame = cv2.flip(frame, 1)

    # Calculate FPS
    current_time = time.time()
    frame_interval = current_time - prev_time
    prev_time = current_time
    fps = 1.0 / frame_interval if frame_interval > 0 else 0

    # Check number of hands detected (using the flipped frame)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    num_hands = 0
    if results.multi_hand_landmarks is not None:
        num_hands = len(results.multi_hand_landmarks)
    
    if num_hands > 1:
        cv2.putText(frame, "Please show only one hand", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        current_letter = None
        letter_start_time = None
    else:
        # Extract landmarks only if exactly one hand is detected
        landmarks = extract_hand_landmarks(frame)

        if landmarks is not None:
            # Get prediction probabilities and compute top 4 predictions
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

            # Letter-hold logic: if the predicted letter remains the same for LETTER_HOLD_THRESHOLD seconds
            # and if the inter-letter delay has passed, then append it to the result string.
            time_since_last = current_time - last_letter_time
            if time_since_last < inter_letter_delay:
                remaining = inter_letter_delay - time_since_last
                cv2.putText(frame, f'Time until next letter: {remaining:.1f}s', (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                if current_letter is None or predicted_label != current_letter:
                    current_letter = predicted_label
                    letter_start_time = current_time
                else:
                    if current_time - letter_start_time >= LETTER_HOLD_THRESHOLD:
                        if predicted_label == "space":
                            result_string += " "
                        elif predicted_label == "del":
                            result_string = result_string[:-1]
                        else:
                            result_string += predicted_label
                        last_letter_time = current_time
                        current_letter = None
                        letter_start_time = None

    # Overlay FPS in the top right corner
    height, width, _ = frame.shape
    cv2.putText(frame, f'FPS: {fps:.1f}', (width - 170, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Overlay the created result string on the frame (with thicker text)
    cv2.putText(frame, f'Result: {result_string}', (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (240, 16, 255), 4, cv2.LINE_AA)

    # Display the video feed
    cv2.imshow("ASL Hand Sign Prediction", frame)
    
    # Check key presses: if 'c' is pressed, clear the result string; if 'q' is pressed, exit.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        result_string = ""
    if key == 127:  # Backspace key code
        result_string = result_string[:-1]

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
