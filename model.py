import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
import config

def build_model():
    """
    Builds EfficientNet-B0 for binary deepfake detection.
    
    Strategy:
    - Load EfficientNet-B0 pretrained on ImageNet
    - Freeze the base layers initially
    - Replace the top with a binary classification head
    - Fine-tune on FaceForensics++ / 140k dataset
    """

    # Load base model without the top classification layer
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base model layers for initial training
    # We will unfreeze later for fine-tuning
    base_model.trainable = False

    # Build classification head
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

def compile_model(model):
    """
    Compile the model with binary crossentropy loss
    and metrics relevant to deepfake detection.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def unfreeze_model(model, base_model):
    """
    Unfreeze the top layers of the base model for fine-tuning.
    We only unfreeze the last 20 layers.
    """
    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Recompile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR / 10),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model