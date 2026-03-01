import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

# -------- Load Image --------
img = load_img("/content/cat.jpg", target_size=(224, 224))
img = img_to_array(img)
img = img / 255.0

# -------- tf.image Augmentation --------
img_tf = tf.image.random_flip_left_right(img)
img_tf = tf.image.random_brightness(img_tf, 0.2)
img_tf = tf.image.random_contrast(img_tf, 0.8, 1.2)

noise = tf.random.normal(shape=tf.shape(img_tf), mean=0.0, stddev=0.05)
img_tf = tf.clip_by_value(img_tf + noise, 0.0, 1.0)

# Convert back to numpy and add batch dimension
img_tf = np.expand_dims(img_tf.numpy(), axis=0)

# -------- ImageDataGenerator Augmentation --------
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2
)

# -------- Generate & Show 5 Images --------
for i, batch in enumerate(datagen.flow(img_tf, batch_size=1)):
    plt.imshow(batch[0])
    plt.axis("off")
    plt.show()
    if i == 4:
        break
