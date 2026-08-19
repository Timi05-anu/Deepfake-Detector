from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2
import numpy as np
import tensorflow as tf
import uvicorn
import shutil
import uuid
import os
import time
from PIL import Image
from preprocessor import preprocess_image, detect_and_crop_face, normalize
import config

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov'}
VIDEO_FRAMES_TO_SAMPLE = 20

app = FastAPI(title="Deepfake Detection API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend folder
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global model variable
model = None

def load_model():
    """Load trained model at startup."""
    global model
    model_path = os.path.join(config.MODEL_DIR, 'best_model_v2.h5')

    if not os.path.exists(model_path):
        print("WARNING: No trained model found. Train the model first.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file format
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: jpg, jpeg, png"
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    # Save uploaded file temporarily
    temp_dir  = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    start_time = time.time()

    try:
        # Preprocess the image
        face = preprocess_image(temp_path)

        if face is None:
            raise HTTPException(
                status_code=422,
                detail="No face detected in the uploaded image."
            )

        # Run inference
        face_tensor = np.expand_dims(face, axis=0)
        prediction  = model.predict(face_tensor, verbose=0)[0][0]

        # Apply threshold
        label      = "Fake" if prediction >= 0.5 else "Real"
        confidence = float(prediction) if prediction >= 0.5 else float(1 - prediction)
        confidence = round(confidence * 100, 2)

        processing_time = round(time.time() - start_time, 3)

        return {
            "label":           label,
            "confidence":      confidence,
            "processing_time": processing_time,
            "filename":        file.filename
        }

    finally:
        # Always delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    # Validate file format
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Allowed: mp4, avi, mov"
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    # Save uploaded file temporarily
    temp_dir  = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    start_time = time.time()

    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=422,
                detail="Could not open video file. It may be corrupted or use an unsupported codec."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise HTTPException(
                status_code=422,
                detail="Video has no readable frames."
            )

        # Evenly spaced frame indices across the whole video
        n_samples = min(VIDEO_FRAMES_TO_SAMPLE, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)

        scores = []
        frames_read = 0
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            frames_read += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            face = detect_and_crop_face(pil_frame)
            if face is None:
                # frame skipped -- no face detected, doesn't count toward the average
                continue

            face_norm   = normalize(face)
            face_tensor = np.expand_dims(face_norm, axis=0)
            pred        = model.predict(face_tensor, verbose=0)[0][0]
            scores.append(float(pred))

        cap.release()

        if not scores:
            raise HTTPException(
                status_code=422,
                detail="No face detected in any sampled frame of the video."
            )

        avg_score  = sum(scores) / len(scores)
        label      = "Fake" if avg_score >= 0.5 else "Real"
        confidence = avg_score if avg_score >= 0.5 else 1 - avg_score
        confidence = round(confidence * 100, 2)

        processing_time = round(time.time() - start_time, 3)

        return {
            "label":           label,
            "confidence":      confidence,
            "processing_time": processing_time,
            "filename":        file.filename,
            "frames_analyzed": frames_read,
            "frames_with_face": len(scores)
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)