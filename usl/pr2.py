import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import cifar10
from PIL import Image
# Load dataset
(x_train, y_train), _ = cifar10.load_data()
# Keep cats (3) and dogs (5)
mask = np.isin(y_train, [3, 5]).flatten()
x_train = x_train[mask]
y_train = (y_train[mask] == 5).astype(int)
# Create tf.data pipeline (RAM efficient)
IMG_SIZE = 128   # smaller = less RAM
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
# Build model
base = keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False
model = keras.Sequential([
    base,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
# Train
model.fit(dataset, epochs=3)
# Prediction on custom image
img = Image.open("/content/cat.jpg").resize((IMG_SIZE, IMG_SIZE))
x = np.expand_dims(np.array(img) / 255.0, 0)
prob = model.predict(x)[0][0]
if prob > 0.5:
    print("Prediction: Dog | Confidence:", float(prob))
else:
    print("Prediction: Cat | Confidence:", float(1 - prob))
