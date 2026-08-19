import albumentations as A
import numpy as np

def get_training_augmentation():
    """
    Augmentation applied only during training.
    Simulates real-world conditions including social media compression.
    """
    return A.Compose([
        # Horizontal flip
        A.HorizontalFlip(p=0.5),

        # Rotation within +/- 15 degrees
        A.Rotate(limit=15, p=0.5),

        # Brightness and contrast adjustment within +/- 20%
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # Low-level Gaussian noise
        A.GaussNoise(p=0.3),

        # JPEG compression simulation (quality 30-90)
        # This is the most important augmentation for this project
        # It trains the model to detect fakes even after platform compression
       A.ImageCompression(
            compression_type='jpeg',
            quality_range=(30, 90),
            p=0.5
        ),
    ])

def get_validation_augmentation():
    """
    No augmentation for validation and test sets.
    We want to evaluate on clean data only.
    """
    return A.Compose([])

def apply_augmentation(face_array, augmentation):
    """
    Apply augmentation to a face array.
    Input: float32 normalized array
    Output: float32 normalized array
    
    Albumentations expects uint8 input so we convert,
    augment, then convert back.
    """
    # Convert from normalized float back to uint8 for albumentations
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Denormalize
    face_uint8 = ((face_array * std + mean) * 255.0)
    face_uint8 = np.clip(face_uint8, 0, 255).astype(np.uint8)

    # Apply augmentation
    augmented = augmentation(image=face_uint8)['image']

    # Renormalize
    face_float = augmented.astype(np.float32) / 255.0
    face_float = (face_float - mean) / std

    return face_float
