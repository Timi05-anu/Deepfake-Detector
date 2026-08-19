import os
import numpy as np
import tensorflow as tf
from preprocessor import preprocess_image
from augmentation import get_training_augmentation, get_validation_augmentation, apply_augmentation
import config

def get_image_paths_and_labels(split_dir):
    """
    Walks the real and fake folders and returns
    a list of image paths and corresponding labels.
    Label: 0 = real, 1 = fake
    """
    image_paths = []
    labels = []

    for label_name, label_value in [('real', 0), ('fake', 1)]:
        folder = os.path.join(split_dir, label_name)
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(folder, filename))
                labels.append(label_value)

    return image_paths, labels

def load_and_preprocess(image_path, label, augmentation):
    """
    Load a single image, run it through the preprocessing
    pipeline, and apply augmentation.
    Returns the face array and label, or None if no face found.
    """
    face = preprocess_image(image_path)

    if face is None:
        return None, label

    face = apply_augmentation(face, augmentation)
    return face, label

def create_dataset(split_dir, augmentation, batch_size=config.BATCH_SIZE):
    """
    Creates a tf.data.Dataset from a split directory.
    Skips images where no face is detected.
    """
    image_paths, labels = get_image_paths_and_labels(split_dir)
    print(f"Found {len(image_paths)} images in {split_dir}")

    faces = []
    valid_labels = []
    skipped = 0

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        face, lbl = load_and_preprocess(path, label, augmentation)

        if face is None:
            skipped += 1
            continue

        faces.append(face)
        valid_labels.append(lbl)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(image_paths)} images, skipped {skipped}")

    print(f"Done. Total usable: {len(faces)}, Skipped: {skipped}")

    faces = np.array(faces, dtype=np.float32)
    valid_labels = np.array(valid_labels, dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((faces, valid_labels))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset