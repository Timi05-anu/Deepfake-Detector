import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import config

# Initialize MTCNN once so it is not reloaded for every image
mtcnn = MTCNN(keep_all=False, device='cpu')

def load_image(image_path):
    """Load an image from disk and convert to RGB."""
    img = Image.open(image_path).convert('RGB')
    return img

def detect_and_crop_face(pil_image):
    """
    Detect face using MTCNN, expand the bounding box by 30%,
    crop and resize to 224x224.
    Returns a numpy array or None if no face is detected.
    """
    img_array = np.array(pil_image)
    height, width = img_array.shape[:2]

    # Detect face bounding box
    boxes, _ = mtcnn.detect(pil_image)

    if boxes is None:
        return None

    # Take the first detected face
    x1, y1, x2, y2 = boxes[0]

    # Expand bounding box by 30% on each side
    margin_x = int((x2 - x1) * config.FACE_MARGIN)
    margin_y = int((y2 - y1) * config.FACE_MARGIN)

    x1 = max(0, int(x1) - margin_x)
    y1 = max(0, int(y1) - margin_y)
    x2 = min(width,  int(x2) + margin_x)
    y2 = min(height, int(y2) + margin_y)

    # Crop and resize to 224x224
    face_crop = img_array[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, config.IMAGE_SIZE)

    return face_resized

def normalize(face_array):
    """
    Normalize pixel values to [0,1] then apply
    ImageNet mean and std standardization.
    """
    face = face_array.astype(np.float32) / 255.0

    mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(config.IMAGENET_STD,  dtype=np.float32)

    face = (face - mean) / std
    return face

def preprocess_image(image_path):
    """
    Full pipeline for a single image.
    Returns normalized face array or None if no face found.
    """
    pil_image = load_image(image_path)
    face      = detect_and_crop_face(pil_image)

    if face is None:
        return None

    normalized = normalize(face)
    return normalized