# Sign_lang_detector

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green?logo=opencv)](https://opencv.org)

I built a small Streamlit app that recognizes American Sign Language (ASL) letters (a–z) and digits (0–9) from a webcam or images. It uses MediaPipe hand landmarks and a simple Random Forest classifier under the hood.

## Features
- **Real-time detection** from your webcam
- **Image upload** inference
- **ASL classes**: digits 0–9 and letters a–z
- **Confidence score** for each prediction
- Simple, fast, and runs locally

## Quick Start

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If `requirements.txt` is empty, install manually:

```bash
pip install streamlit opencv-python mediapipe numpy scikit-learn pillow matplotlib
```

## Project Structure
```
Sign_lang/
├─ app.py                # Streamlit UI
├─ train_model.py        # Model training script
├─ utils.py              # Inference utilities (landmarks + prediction)
├─ model/
│  └─ asl_model.pkl      # Saved model (created after training)
├─ asl_dataset/          # Your dataset (see below)
├─ requirements.txt
└─ README.md
```

## How It Works
- **Hand landmarks**: I use `mediapipe.solutions.hands` to detect a single hand and extract 21 keypoints (x, y, z) → 63 numeric features per frame.
- **Classifier**: A `RandomForestClassifier` (sklearn) is trained on these features for all classes found in your dataset folders.
- **Inference**: For each frame/image, extract landmarks → predict class → show confidence and (optionally) draw landmarks.

Key code paths:
- Landmark extraction: `ASLDetector.extract_landmarks()` in `utils.py`
- Prediction: `ASLDetector.predict()` in `utils.py`
- Drawing landmarks: `ASLDetector.draw_landmarks()` in `utils.py`
- Training pipeline: `ASLModelTrainer` in `train_model.py`
- UI flows: `show_realtime_detection()`, `show_image_upload()` in `app.py`

## Requirements
- Python 3.9+ (I tested on Windows)
- Recommended packages (install via pip):
  - streamlit
  - opencv-python
  - mediapipe
  - numpy
  - scikit-learn
  - pillow
  - matplotlib

Install all at once with the commands in Quick Start above.

## Dataset Setup
Prepare `asl_dataset/` with one subfolder per class. Folder names become labels.

Example layout:
```
asl_dataset/
├─ 0/
├─ 1/
├─ ...
├─ 9/
├─ a/
├─ b/
├─ ...
└─ z/
```
Each folder should contain clear hand images (jpg/png/jpeg). The trainer will skip images without detectable hands.

## Train the Model
From the project root (`Sign_lang/`):
```bash
python train_model.py
```
What it does:
- Scans `asl_dataset/` for class folders
- Extracts hand landmarks with MediaPipe
- Trains a Random Forest classifier
- Saves `model/asl_model.pkl` with model, classes, and accuracy

If training succeeds, you’ll see accuracy and sample predictions in the console.

## Why I built this
- I wanted a minimal, local ASL recognizer that doesn’t need heavy deep learning or cloud services.
- MediaPipe hands are fast and reliable, and a classic ML model is enough for a first version.

## Notes from my setup
- Webcam index `0` worked on my machine. If your camera doesn’t open, try `1` or `2` in `cv2.VideoCapture(0)`.
- Mediapipe install can take a minute depending on your environment.
- Training time depends on how many images you have; small datasets finish quickly.

## Run the App
After training:
```bash
streamlit run app.py
```
In the app:
- Choose **Real-time Detection** to use your webcam
- Choose **Image Upload** to analyze a single image
- See **Model Info** for classes and quick training from the UI

## Usage Notes
- Ensure only one camera-using app runs at a time.
- Use consistent lighting and show a single hand for best results.
- If you change the dataset, retrain to update the model/classes.

## Troubleshooting
- "Model not found": Train first using `python train_model.py`.
- Camera not accessible: Close other apps using the webcam. Try a different camera index in `cv2.VideoCapture(0)` if needed.
- No hand detected: Ensure the hand is clearly visible, centered, and well-lit.
- Low accuracy: Add more images per class; keep backgrounds consistent; ensure images show the correct sign.

## Extending
- Swap `RandomForestClassifier` with another classifier that supports `predict_proba`.
- Add data augmentation (flips, lighting changes) when building the dataset.
- Support multi-hand detection by expanding feature extraction.

## License
MIT License.

## Acknowledgements
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) for fast, robust hand landmark detection
- [Streamlit](https://streamlit.io/) for the lightweight web UI
