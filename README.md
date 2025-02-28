# ASL-Translator

[ASL Alphabet translator](https://asl-translator-ccc.streamlit.app/)

This project is an American Sign Language (ASL) translator that uses computer vision to detect hand landmarks in real time and then predicts the corresponding ASL letter. The application is built using [Streamlit](https://streamlit.io/), [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc), and [MediaPipe](https://mediapipe.dev/). It supports multiple prediction modes and models, provides live FPS feedback, and includes dynamic settings for fine-tuning the recognition parameters.

## Features

- **Real-Time Hand Landmark Detection:**  
  Leverages MediaPipe Hands for robust hand detection and landmark extraction.

- **ASL Letter Prediction:**  
  Uses a pre-trained machine learning model (e.g., a multi-layer perceptron) to predict the ASL letter from the hand landmarks.  
  - Supports both **"Flipped Only"** (mirror view) and **"Aggregate (Dual)"** modes (combining predictions from both the flipped and original images).

- **Dynamic Model Selection:**  
  Easily switch between different models stored in the `Models/` folder via the sidebar.

- **Customizable Prediction Settings:**  
  Configure hand detection delay, minimum hold time, and inter-letter gap via the sidebar.

- **Enhanced Video Quality:**  
  Optionally increase video resolution (e.g., 1280×720) through media stream constraints.

- **FPS and Timing Feedback:**  
  The interface overlays the current FPS and displays the time remaining before the next letter can be added to the prediction string.

- **Thread-Safe Processing:**  
  Uses locks to safely share data between the video callback thread and the main Streamlit interface.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ReehalS/ASL-Translator.git
   cd ASL-Translator
   ```

2. **Install Dependencies:**

   Make sure you have Python 3.9 installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

   The required packages include:
   - streamlit
   - streamlit-webrtc
   - mediapipe
   - opencv-python
   - joblib
   - numpy

3. **Models:**

   Place your pre-trained model files (in `.pkl` format) into the `Models/` folder. The project defaults to `multi_layer_perceptron_best.pkl` if available.

## Usage

Run the Streamlit app locally using:

```bash
streamlit run Frontend/dual_pred_local.py
```

Once the app starts, you can adjust various settings from the sidebar:

- **Hand Detection Delay:** Time to wait after the hand appears before starting predictions.
- **Minimum Hold Time:** Duration the predicted letter must remain unchanged to qualify.
- **Inter-Letter Gap:** Minimum time between adding successive letters.
- **Model Selection:** Choose a different ASL prediction model from the `Models/` folder.
- **Prediction Mode:** Switch between "Aggregate (Dual)" (combining predictions from both flipped and original images) and "Flipped Only".

The live video feed will display:
- The current predicted letter.
- The built string of letters.
- An FPS counter at the top right.
- The remaining time before the next letter is added.

## How It Works

1. **Video Streaming:**  
   The app uses `streamlit-webrtc` to capture live video. The video is processed frame by frame.

2. **Hand Landmark Extraction:**  
   Each frame is analyzed with MediaPipe to extract hand landmarks (if exactly one hand is detected).

3. **ASL Prediction:**  
   The extracted landmarks are fed into a pre-trained model to predict the corresponding ASL letter.  
   In "Aggregate (Dual)" mode, predictions are made on both the flipped and original images, and their probabilities are aggregated.

4. **Letter Accumulation:**  
   A letter is added to the output string only after it has been held for the specified minimum hold time following an inter-letter gap.

5. **Feedback & Overlays:**  
   Real-time overlays display the current prediction, the accumulated string, FPS, and time remaining before the next letter addition.


The dataset used to train this model is at [https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)