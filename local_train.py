import os
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import albumentations as A
import time

print("Setting up...")

# Paths
DATA_DIR  = r"C:\Users\Timi\Desktop\400L Project\Code\data\real_vs_fake\real-vs-fake"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR  = os.path.join(DATA_DIR, "test")
MODEL_DIR = r"C:\Users\Timi\Desktop\400L Project\Code\model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Settings
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 16
EPOCHS        = 10
LR            = 1e-4
FACE_MARGIN   = 0.3
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Samples per class
TRAIN_PER_CLASS = 500
VALID_PER_CLASS = 100
TEST_PER_CLASS  = 100

# MTCNN
mtcnn = MTCNN(keep_all=False, device='cpu')

def preprocess_image(image_path):
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img_array = np.array(pil_img)
        boxes, _ = mtcnn.detect(pil_img)
        if boxes is None:
            return None
        x1, y1, x2, y2 = boxes[0]
        h, w = img_array.shape[:2]
        mx = int((x2 - x1) * FACE_MARGIN)
        my = int((y2 - y1) * FACE_MARGIN)
        x1 = max(0, int(x1) - mx)
        y1 = max(0, int(y1) - my)
        x2 = min(w, int(x2) + mx)
        y2 = min(h, int(y2) + my)
        face = img_array[y1:y2, x1:x2]
        face = cv2.resize(face, IMAGE_SIZE)
        face = face.astype(np.float32) / 255.0
        face = (face - IMAGENET_MEAN) / IMAGENET_STD
        return face
    except:
        return None

aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(p=0.3),
    A.ImageCompression(compression_type='jpeg', quality_range=(30, 90), p=0.5),
])

def apply_aug(face):
    face_uint8 = ((face * IMAGENET_STD + IMAGENET_MEAN) * 255.0)
    face_uint8 = np.clip(face_uint8, 0, 255).astype(np.uint8)
    augmented  = aug(image=face_uint8)['image']
    face_float = augmented.astype(np.float32) / 255.0
    face_float = (face_float - IMAGENET_MEAN) / IMAGENET_STD
    return face_float

def load_split(split_dir, per_class, augment=False):
    faces, labels = [], []
    for label_name, label_val in [('real', 0), ('fake', 1)]:
        folder = os.path.join(split_dir, label_name)
        files  = sorted(os.listdir(folder))[:per_class]
        print(f"  Loading {len(files)} {label_name} images...")
        for i, fname in enumerate(files):
            path = os.path.join(folder, fname)
            face = preprocess_image(path)
            if face is None:
                continue
            if augment:
                face = apply_aug(face)
            faces.append(face)
            labels.append(label_val)
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(files)} done")
    return np.array(faces, dtype=np.float32), np.array(labels, dtype=np.float32)

print("\nLoading training data (500 real + 500 fake)...")
start = time.time()
X_train, y_train = load_split(TRAIN_DIR, TRAIN_PER_CLASS, augment=True)
print(f"Train loaded: {X_train.shape} in {(time.time()-start)/60:.1f} mins")

print("\nLoading validation data...")
X_val, y_val = load_split(VALID_DIR, VALID_PER_CLASS, augment=False)
print(f"Val loaded: {X_val.shape}")

print("\nLoading test data...")
X_test, y_test = load_split(TEST_DIR, TEST_PER_CLASS, augment=False)
print(f"Test loaded: {X_test.shape}")

print("\nBuilding model...")
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(224, 224, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.3)(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
model.summary()

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_model.h5'),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    CSVLogger(os.path.join(MODEL_DIR, 'training_log.csv'))
]

print("\nStarting Phase 1 training (frozen base)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

print("\nPhase 2 fine-tuning (unfreezing top 20 layers)...")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR/10),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

print("\nEvaluating on test set...")
results = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {results[1]:.4f}")
print(f"Test AUC:      {results[2]:.4f}")

final_path = os.path.join(MODEL_DIR, 'final_model.h5')
model.save(final_path)
print(f"\nFinal model saved to: {final_path}")
print("Training complete.")