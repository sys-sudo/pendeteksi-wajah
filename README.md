# Real-Time Human Detection and Facial Feature Analysis

This Python script uses OpenCV and deep learning models for real-time human detection, facial feature recognition, and demographic prediction (age and gender). It utilizes the webcam to perform face detection, eye tracking, and predicts the age and gender of detected faces.

## Features

- **Package Installation & Upgrade**: Automatically installs or upgrades required packages (`opencv-python`, `numpy`).
- **Model Loading**: Loads pre-trained deep learning models for age and gender prediction, and face detection using the OpenCV DNN module and Haar cascades.
- **Camera Detection**: Automatically detects available cameras and allows switching between them.
- **Real-Time Face & Feature Detection**: Detects faces, eyes, smiles, noses, and other features in real-time video feed.
- **Age & Gender Prediction**: Predicts the age and gender of detected faces using pre-trained models.
- **Real-Time Display**: Displays the webcam feed with detected faces, features, and predictions in real time.
- **Manual and Auto Tracking**: Toggle between auto-tracking of faces, raw mode, and processed mode for enhanced feature detection.

## Dependencies

- Python 3.x
- `opencv-python`: For video processing, object detection (face, eyes, etc.), and deep learning-based predictions.
- `numpy`: For handling numerical operations and array manipulations.

## Installation

### 1. Install Required Packages

Ensure you have `opencv-python` and `numpy` installed:

```bash
pip install opencv-python numpy
