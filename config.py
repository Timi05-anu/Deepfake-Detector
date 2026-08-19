import os

# Base paths
BASE_DIR = r"C:\Users\Timi\Desktop\400L Project\Code"
DATA_DIR = os.path.join(BASE_DIR, "data", "real_vs_fake", "real-vs-fake")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# Model output
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Training settings
EPOCHS     = 20
LR         = 1e-4

# MTCNN settings
FACE_MARGIN = 0.3  # expand bounding box by 30% on each side

# ImageNet normalization values (required for EfficientNet-B0)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]