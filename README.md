# Deepfake Detector

A web based deepfake detection tool that classifies real faces from AI generated ones, built with EfficientNet-B0.

## Screenshot

*(Add a screenshot of the app in action here)*

## Results

This is the baseline phase of the project: a smaller model trained locally on a subset of the dataset to validate the full pipeline end to end. Full scale training on the complete 140k image dataset is planned as the next phase.

Baseline performance:

- Validation AUC: ~0.68
- Validation Accuracy: ~62%

## How It Works

1. Faces are detected and cropped from an uploaded image or video using MTCNN.
2. The cropped face is passed through an EfficientNet-B0 classifier trained to distinguish real faces from deepfakes.
3. A Flask backend serves the model and returns a real or fake verdict through a simple web interface.

## Tech Stack

- Python, TensorFlow/Keras
- EfficientNet-B0
- MTCNN for face detection
- Flask backend
- HTML/CSS/JS frontend

## Dataset

Trained on the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset from Kaggle, used as a practical substitute for FaceForensics++.

## Project Status

Final year computer science capstone project. Baseline phase complete on a local subset. Next phase is training on the full dataset at scale.
