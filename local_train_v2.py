"""
local_train_v2.py — improved retrain of the deepfake detector.

Diagnosis of the v1 run (model/training_log.csv):
  val_auc peaked at epoch 0 (0.609) and declined every epoch after
  (0.605, 0.588, 0.586) while train auc kept climbing (0.591->0.619).
  Train-up / val-down from epoch 0 is classic overfitting, not an
  unstable/too-high LR (that usually shows up as erratic, non-monotonic
  loss). With only 1,000 images feeding a ~330k-param head (Dense-256 +
  BatchNorm + 2x Dropout), the head memorizes the tiny train set almost
  immediately. EarlyStopping(patience=3) then just stopped the run --
  there was no ReduceLROnPlateau, so the LR never got a chance to adapt
  and help the model escape the stall.
  Dropout (0.3/0.3) is a moderate, fairly standard setting -- not
  obviously the main culprit here, so it's left unchanged. The two
  levers that actually address what the curves show are (1) far more
  training data, which this script provides (6x), and (2) an LR
  schedule that can adapt instead of just giving up.
  Fine-tuning only unfroze the last 20 (of 237) EfficientNetB0 layers,
  which is very shallow -- with 6x the data the model can support more
  trainable capacity, so this script unfreezes the last 60 layers
  (roughly the last two MBConv stages + the top conv block).

Memory design note:
  Storing 6,800 normalized float32 224x224x3 images would need ~4.1GB,
  which doesn't fit in the ~2.5GB currently available on this machine.
  Face crops are instead cached as uint8 (4x smaller, ~1.0GB for the
  full target set) and normalized to ImageNet float32 stats on-the-fly
  per batch via a tf.data pipeline. This changes nothing about the
  model's input contract at inference time (preprocessor.py still
  normalizes before calling model.predict) -- it's purely a training-
  time memory optimization.
"""

import csv
import gc
import os
import sys
import time
import traceback

import numpy as np
import psutil
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
)
import albumentations as A

from preprocessor import load_image, detect_and_crop_face
import config

np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------- paths ---
DATA_DIR  = config.DATA_DIR
TRAIN_DIR = config.TRAIN_DIR
VALID_DIR = config.VALID_DIR
TEST_DIR  = config.TEST_DIR
MODEL_DIR = config.MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)

CHECKPOINT_PATH  = os.path.join(MODEL_DIR, "v2_checkpoint.h5")
BEST_MODEL_PATH  = os.path.join(MODEL_DIR, "best_model_v2.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model_v2.h5")
LOG_PATH         = os.path.join(MODEL_DIR, "training_log_v2.csv")
CACHE_PATH       = os.path.join(MODEL_DIR, "v2_data_cache.npz")
# NOTE: LOG_PATH is *not* deleted here unconditionally anymore -- a resume
# needs to read it to figure out where the last run left off. Fresh-start
# cleanup happens inside main(), only when no checkpoint is found.

# --------------------------------------------------------------- config ---
IMAGE_SIZE  = config.IMAGE_SIZE          # (224, 224)
BATCH_SIZE  = 16                         # CPU-only, kept modest per constraints
FACE_MARGIN = config.FACE_MARGIN

IMAGENET_MEAN = tf.constant(config.IMAGENET_MEAN, dtype=tf.float32)
IMAGENET_STD  = tf.constant(config.IMAGENET_STD, dtype=tf.float32)

TRAIN_PER_CLASS_TARGET = 3000
VAL_PER_CLASS_TARGET   = 200
TEST_PER_CLASS_TARGET  = 200

UNFREEZE_LAYERS   = 60      # up from 20 in v1 -- see diagnosis above
PHASE1_LR         = 1e-4
PHASE2_LR         = 1e-5
PHASE1_EPOCHS_MAX = 15
PHASE2_EPOCHS_MAX = 10

# Phase 2 backprops through 60 unfrozen base layers instead of just the head,
# which needs substantially more activation memory. A second unexpected kill
# happened mid-batch during Phase 2's first epoch (RAM-correlated, per a VS
# Code freeze at the same time), so Phase 2 gets a smaller batch size than
# Phase 1's frozen-base training.
PHASE2_BATCH_SIZE = 8

RAM_SAFETY_FRACTION = 0.55  # fraction of *currently available* RAM we may use for image arrays
RAM_WARN_PCT        = 85.0
RAM_HARD_STOP_PCT   = 92.0
RAM_CHECK_EVERY     = 250   # images

RAM_TRAIN_WARN_PCT    = 90.0  # in-fit circuit breaker (separate from the data-loading one above)
RAM_TRAIN_CHECK_EVERY = 20    # batches
RAM_TRAIN_PAUSE_SEC   = 5

BYTES_PER_IMAGE_UINT8 = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3  # 1 byte/channel

# ------------------------------------------------------------ ram utils ---
def ram_snapshot(stage=""):
    m = psutil.virtual_memory()
    flag = "  !! WARNING: high memory usage !!" if m.percent > RAM_WARN_PCT else ""
    print(f"[RAM] {stage}: {m.percent:.1f}% used, {m.available/1e9:.2f} GB available{flag}")
    return m


class RAMMonitorCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        ram_snapshot(f"epoch {epoch} end")


class RAMCircuitBreakerCallback(Callback):
    """
    Data-loading has its own hard-stop (see load_split), but nothing was
    watching RAM *during* model.fit() itself -- and that's exactly where
    the second crash happened (mid-batch, Phase 2). This checks every
    RAM_TRAIN_CHECK_EVERY batches and, if usage crosses RAM_TRAIN_WARN_PCT,
    forces a gc pass and a short pause to give the OS a chance to reclaim
    memory before the next batch allocates more.
    
    """
    def on_train_batch_end(self, batch, logs=None):
        if batch % RAM_TRAIN_CHECK_EVERY != 0:
            return
        pct = psutil.virtual_memory().percent
        if pct > RAM_TRAIN_WARN_PCT:
            print(f"  !! [RAM CIRCUIT BREAKER] {pct:.1f}% at batch {batch} -- "
                  f"forcing gc + pausing {RAM_TRAIN_PAUSE_SEC}s !!", flush=True)
            gc.collect()
            time.sleep(RAM_TRAIN_PAUSE_SEC)
            pct_after = psutil.virtual_memory().percent
            print(f"  [RAM CIRCUIT BREAKER] after pause: {pct_after:.1f}%", flush=True)


def plan_dataset_sizes():
    """Decide real per-class counts from currently available RAM."""
    base_train, base_val, base_test = (
        TRAIN_PER_CLASS_TARGET, VAL_PER_CLASS_TARGET, TEST_PER_CLASS_TARGET
    )
    base_total = 2 * (base_train + base_val + base_test)

    m = ram_snapshot("startup, before loading anything")
    budget_bytes = m.available * RAM_SAFETY_FRACTION
    budget_images = int(budget_bytes // BYTES_PER_IMAGE_UINT8)

    print(f"[RAM] budget: {budget_bytes/1e9:.2f} GB -> ~{budget_images} images "
          f"(target is {base_total} images at uint8 face-crop size)")

    if budget_images >= base_total * 3:
        # generous headroom -- grow val/test a bit too, train stays capped at target
        scale = min(2.0, budget_images / base_total)
        plan = (base_train, int(base_val * scale), int(base_test * scale))
        print(f"[PLAN] ample RAM headroom -- keeping train at target, "
              f"scaling val/test x{scale:.2f}: {plan}")
    elif budget_images >= base_total:
        plan = (base_train, base_val, base_test)
        print(f"[PLAN] RAM budget covers the full target: {plan}")
    else:
        ratio = max(budget_images / base_total, 0.05)
        plan = (
            max(200, int(base_train * ratio)),
            max(50, int(base_val * ratio)),
            max(50, int(base_test * ratio)),
        )
        print(f"[PLAN] !! RAM budget is below target ({budget_images} < {base_total} images).")
        print(f"[PLAN] Scaling dataset down by ~{ratio:.2f} to avoid crashing: {plan}")

    return plan


def find_resume_state():
    """
    If a prior run left a checkpoint + log behind, figure out where it
    stopped so we can continue instead of retraining from scratch.
    Returns None for a fresh start.
    """
    if not (os.path.exists(CHECKPOINT_PATH) and os.path.exists(LOG_PATH)):
        return None

    with open(LOG_PATH, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    rows.sort(key=lambda r: int(r['epoch']))
    last_epoch = int(rows[-1]['epoch'])
    val_aucs = [float(r['val_auc']) for r in rows]
    best_val_auc = max(val_aucs)

    return {
        'initial_epoch': last_epoch + 1,
        'best_val_auc': best_val_auc,
        'last_lr': float(rows[-1]['lr']),
    }


# -------------------------------------------------------------- loading ---
AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),                                   # was 15 -- dialed back
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(p=0.3),
    A.ImageCompression(compression_type='jpeg', quality_range=(55, 90), p=0.5),  # was (30, 90)
])


def detect_and_crop(path):
    try:
        pil_img = load_image(path)
        return detect_and_crop_face(pil_img)  # uint8 array or None
    except Exception:
        return None


def load_split(split_dir, per_class, augment, split_name):
    print(f"\nLoading {split_name} ({per_class} per class target = {2*per_class} images)...")
    total_target = per_class * 2
    faces = np.zeros((total_target, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    labels = np.zeros((total_target,), dtype=np.float32)

    count = 0
    stopped_early = False
    start = time.time()

    for label_name, label_val in [('real', 0), ('fake', 1)]:
        if stopped_early:
            break
        folder = os.path.join(split_dir, label_name)
        files = sorted(os.listdir(folder))[:per_class]
        print(f"  {label_name}: {len(files)} candidates")

        for i, fname in enumerate(files):
            face = detect_and_crop(os.path.join(folder, fname))
            if face is None:
                continue
            if augment:
                face = AUG(image=face)['image']
            faces[count] = face
            labels[count] = label_val
            count += 1

            if count % RAM_CHECK_EVERY == 0:
                m = ram_snapshot(f"{split_name} loading, {count} images loaded")
                if m.percent > RAM_HARD_STOP_PCT:
                    print(f"  !! RAM at {m.percent:.1f}% > {RAM_HARD_STOP_PCT}% hard stop -- "
                          f"halting {split_name} loading early at {count} images.")
                    stopped_early = True
                    break
            if (i + 1) % 500 == 0:
                print(f"    {label_name}: {i+1}/{len(files)} scanned, {count} usable so far")

        if stopped_early:
            break

    faces = faces[:count]
    labels = labels[:count]
    elapsed = time.time() - start
    print(f"  {split_name} done: {count} usable images in {elapsed/60:.1f} min "
          f"({elapsed/max(count,1):.3f} s/image)")
    return faces, labels


def make_dataset(faces_uint8, labels, batch_size, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((faces_uint8, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=42, reshuffle_each_iteration=True)

    def _normalize(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img, label

    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# --------------------------------------------------------------- model ----
def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
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
    return model, base_model


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )


# ----------------------------------------------------------------- main ---
def main():
    script_start = time.time()

    resume = find_resume_state()
    if resume:
        print(f"[RESUME] Found prior run: last completed epoch {resume['initial_epoch']-1}, "
              f"best val_auc {resume['best_val_auc']:.5f}, last LR {resume['last_lr']:.2e}")
        print(f"[RESUME] Will continue from epoch {resume['initial_epoch']}.")
        print("[RESUME] Note: EarlyStopping/ReduceLROnPlateau patience+best tracking restart "
              "fresh from this point (Keras resets them at the start of every fit() call, "
              "the same way it already does at the Phase 1 -> Phase 2 boundary) -- they are "
              "not literally continuous across the process restart.")
    else:
        print("[RESUME] No prior checkpoint found -- starting fresh.")
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)

    if os.path.exists(CACHE_PATH):
        print(f"\n[CACHE] Found cached preprocessed dataset at {CACHE_PATH}, loading instead of "
              f"re-running face detection...")
        cache = np.load(CACHE_PATH)
        X_train, y_train = cache['X_train'], cache['y_train']
        X_val, y_val     = cache['X_val'], cache['y_val']
        X_test, y_test   = cache['X_test'], cache['y_test']
        print(f"  train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        ram_snapshot("after cache load")
    else:
        train_pc, val_pc, test_pc = plan_dataset_sizes()

        X_train, y_train = load_split(TRAIN_DIR, train_pc, augment=True,  split_name="train")
        ram_snapshot("after train load")
        X_val, y_val     = load_split(VALID_DIR, val_pc,   augment=False, split_name="validation")
        ram_snapshot("after validation load")
        X_test, y_test   = load_split(TEST_DIR,  test_pc,  augment=False, split_name="test")
        ram_snapshot("after test load")

        print(f"\n[CACHE] Saving preprocessed dataset to {CACHE_PATH} for fast resume next time...")
        np.savez(CACHE_PATH, X_train=X_train, y_train=y_train,
                  X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   BATCH_SIZE, shuffle=False)
    test_ds  = make_dataset(X_test,  y_test,  BATCH_SIZE, shuffle=False)

    if resume:
        print(f"\n[RESUME] Loading weights + optimizer state from {CHECKPOINT_PATH}...")
        model = tf.keras.models.load_model(CHECKPOINT_PATH)
        base_model = next(l for l in model.layers if isinstance(l, tf.keras.Model))
        initial_epoch = resume['initial_epoch']
    else:
        print("\nBuilding model...")
        model, base_model = build_model()
        compile_model(model, PHASE1_LR)
        initial_epoch = 0
    model.summary()

    best_ckpt_cb = ModelCheckpoint(filepath=BEST_MODEL_PATH, monitor='val_auc', mode='max',
                                     save_best_only=True, verbose=1)
    if resume:
        # ModelCheckpoint(save_best_only=True) does NOT reset .best in on_train_begin
        # (unlike EarlyStopping/ReduceLROnPlateau), so this patch actually sticks --
        # without it the first resumed epoch would look like an automatic "new best"
        # (fresh instance defaults to -inf) and could overwrite best_model_v2.h5 with
        # a worse checkpoint than the true historical best.
        best_ckpt_cb.best = resume['best_val_auc']

    callbacks = [
        best_ckpt_cb,
        ModelCheckpoint(filepath=CHECKPOINT_PATH, save_best_only=False,
                         save_freq='epoch', verbose=0),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5,
                           patience=2, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_auc', mode='max', patience=6,
                       restore_best_weights=True, verbose=1),
        CSVLogger(LOG_PATH, append=True),
        RAMMonitorCallback(),
        RAMCircuitBreakerCallback(),
    ]

    if initial_epoch < PHASE1_EPOCHS_MAX:
        print(f"\nPhase 1: {'resuming' if resume else 'frozen base, starting'} at epoch "
              f"{initial_epoch}, ceiling {PHASE1_EPOCHS_MAX} epochs...")
        phase1_start = time.time()
        history1 = model.fit(
            train_ds, validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=PHASE1_EPOCHS_MAX, callbacks=callbacks
        )
        phase1_elapsed = time.time() - phase1_start
        ran = len(history1.epoch)
        print(f"Phase 1 segment done in {phase1_elapsed/60:.1f} min "
              f"({ran} epochs this segment, {phase1_elapsed/max(ran,1)/60:.1f} min/epoch)")
        phase1_last_epoch = history1.epoch[-1] if history1.epoch else initial_epoch - 1
    else:
        print(f"\nPhase 1 already complete (resumed at epoch {initial_epoch} >= "
              f"ceiling {PHASE1_EPOCHS_MAX}). Skipping to Phase 2.")
        phase1_last_epoch = initial_epoch - 1

    print(f"\nPhase 2: fine-tuning, unfreezing last {UNFREEZE_LAYERS} of "
          f"{len(base_model.layers)} base layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    compile_model(model, PHASE2_LR)

    print(f"Phase 2 batch size: {PHASE2_BATCH_SIZE} (down from {BATCH_SIZE} in Phase 1 -- "
          f"backprop through {UNFREEZE_LAYERS} unfrozen base layers needs much more "
          f"activation memory per batch than the frozen-base head-only training in Phase 1)")
    train_ds_p2 = make_dataset(X_train, y_train, PHASE2_BATCH_SIZE, shuffle=True)
    val_ds_p2   = make_dataset(X_val,   y_val,   PHASE2_BATCH_SIZE, shuffle=False)

    phase2_initial_epoch = phase1_last_epoch + 1
    phase2_ceiling = phase2_initial_epoch + PHASE2_EPOCHS_MAX
    phase2_start = time.time()
    history2 = model.fit(
        train_ds_p2, validation_data=val_ds_p2,
        initial_epoch=phase2_initial_epoch,
        epochs=phase2_ceiling,
        callbacks=callbacks
    )
    phase2_elapsed = time.time() - phase2_start
    print(f"Phase 2 done in {phase2_elapsed/60:.1f} min "
          f"({len(history2.epoch)} epochs, {phase2_elapsed/max(len(history2.epoch),1)/60:.1f} min/epoch)")

    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names
    for name, val in zip(metric_names, results):
        print(f"  test {name}: {val:.4f}")

    model.save(FINAL_MODEL_PATH)
    print(f"\nFinal model saved to: {FINAL_MODEL_PATH}")
    print(f"Best model (by val_auc) saved to: {BEST_MODEL_PATH}")
    print(f"Rolling per-epoch checkpoint at: {CHECKPOINT_PATH}")
    print(f"Full training log at: {LOG_PATH}")

    total_elapsed = time.time() - script_start
    print(f"\nTotal wall-clock time: {total_elapsed/60:.1f} min")
    ram_snapshot("script end")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n!! Training crashed. RAM state at failure: !!")
        try:
            ram_snapshot("crash")
        except Exception:
            pass
        traceback.print_exc()
        sys.exit(1)
