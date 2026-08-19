import os
import numpy as np
import tensorflow as tf
from model import build_model, compile_model
from preprocessor import preprocess_image
from augmentation import get_training_augmentation, get_validation_augmentation, apply_augmentation
from dataloader import get_image_paths_and_labels
import config

def create_small_dataset(split_dir, augmentation, max_per_class=250):
    """Load only max_per_class images per class for smoke testing."""
    faces = []
    labels = []
    skipped = 0

    for label_name, label_value in [('real', 0), ('fake', 1)]:
        folder = os.path.join(split_dir, label_name)
        files = os.listdir(folder)[:max_per_class]

        for filename in files:
            path = os.path.join(folder, filename)
            face = preprocess_image(path)

            if face is None:
                skipped += 1
                continue

            face = apply_augmentation(face, augmentation)
            faces.append(face)
            labels.append(label_value)

    print(f"Loaded {len(faces)} images, skipped {skipped}")
    return np.array(faces, dtype=np.float32), np.array(labels, dtype=np.float32)

print("Loading 500 images for smoke test...")
X_train, y_train = create_small_dataset(
    config.TRAIN_DIR,
    get_training_augmentation(),
    max_per_class=250
)

X_val, y_val = create_small_dataset(
    config.VALID_DIR,
    get_validation_augmentation(),
    max_per_class=50
)

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

print("Building model...")
model, base_model = build_model()
model = compile_model(model)

print("Running 2 epochs smoke test...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=2,
    batch_size=32
)

print("\nSmoke test complete.")
print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final val accuracy:   {history.history['val_accuracy'][-1]:.4f}")