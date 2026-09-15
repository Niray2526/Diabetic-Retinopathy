import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# =========================
# CONFIGURATION
# =========================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 5

DATASET_DIR = "dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "train_images")
CSV_FILE = os.path.join(DATASET_DIR, "train.csv")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(CSV_FILE)

print("Dataset size:", len(df))
print(df.head())

# Convert ID into actual image path
df["image_path"] = df["id_code"].apply(
    lambda x: os.path.join(IMAGE_DIR, f"{x}.png")
)

# Remove missing files
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

print("Images found:", len(df))


# =========================
# TRAIN / VALIDATION SPLIT
# =========================

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["diagnosis"],
    random_state=42
)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))


# =========================
# DATA GENERATORS
# =========================

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)

    # EfficientNet can work with floating point images
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


train_paths = train_df["image_path"].values
train_labels = train_df["diagnosis"].values

val_paths = val_df["image_path"].values
val_labels = val_df["diagnosis"].values


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)


# =========================
# DATA AUGMENTATION
# =========================

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])


def prepare_train(path, label):
    image, label = load_image(path, label)
    image = augmentation(image, training=True)

    return image, label


def prepare_validation(path, label):
    return load_image(path, label)


train_dataset = (
    train_dataset
    .map(prepare_train, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset
    .map(prepare_validation, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# =========================
# BUILD MODEL
# =========================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False


inputs = layers.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(inputs, outputs)


# =========================
# COMPILE
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# =========================
# CALLBACKS
# =========================

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)


# =========================
# TRAIN
# =========================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# =========================
# SAVE FINAL MODEL
# =========================

model.save(
    os.path.join(MODEL_DIR, "diabetic_retinopathy_model.keras")
)

print("Training completed.")
print("Model saved inside:", MODEL_DIR)