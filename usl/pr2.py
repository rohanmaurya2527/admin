#With Tensorflow
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

#Without Tensorflow
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import warnings as war
war.filterwarnings('ignore')
# Load CIFAR-10 dataset
cifar = fetch_openml('CIFAR_10_small')
X = cifar.data
y = cifar.target.astype(int)

# Keep cats (3) and dogs (5)
mask = np.isin(y, [3, 5])
X = X[mask]
y = (y[mask] == 5).astype(int)

# Normalize pixel values
X = X / 255.0

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Prediction on custom image
img = Image.open("cat.jpg").resize((32,32))
x = np.array(img).reshape(1, -1) / 255.0

prob = model.predict_proba(x)[0][1]

if prob > 0.5:
    print("Prediction: Dog | Confidence:", prob)
else:
    print("Prediction: Cat | Confidence:", 1 - prob)
