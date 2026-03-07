#Simple Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
# Load image
img = img_to_array(load_img("/content/cat.jpg"))
img = np.expand_dims(img, axis=0)
# Create data generator
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)
# Generate and display 5 augmented images
for i, batch in enumerate(datagen.flow(img, batch_size=1)):
    plt.imshow(batch[0].astype("uint8"))
    plt.axis("off")
    plt.show()
    if i == 4:
        break

#Advanced Augmentation
import tensorflow as tf
import matplotlib.pyplot as plt
def advanced_augment(image):
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, 0.8, 1.5)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[180, 180, 3])  # crop
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image
img = tf.image.decode_jpeg(tf.io.read_file("/content/cat.jpg"))
img = tf.image.resize(img, [224, 224]) / 255.0
aug = advanced_augment(img)
plt.imshow(aug)
plt.axis("off")
plt.show()
