#Simple Augmentation(with Tensorflow)
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

#Simple Augmentation(without Tensorflow)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
# Load image
img = Image.open("/content/cat.jpg")
# Function for augmentation
def simple_augment(image):
    # Rotation
    angle = random.uniform(-20, 20)
    aug = image.rotate(angle)
    # Horizontal Flip
    if random.random() > 0.5:
        aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
    # Zoom
    zoom = random.uniform(0.8, 1.2)
    w, h = aug.size
    new_w, new_h = int(w*zoom), int(h*zoom)
    aug = aug.resize((new_w, new_h))
    aug = aug.crop((0, 0, w, h))
    return aug
# Generate and display 5 images
for i in range(5):
    aug = simple_augment(img)
    plt.imshow(aug)
    plt.axis("off")
    plt.show()

#Advanced Augmentation(with Tensorflow)
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

#Advanced Augmentation(without Tensorflow)
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import random
# Load image
img = Image.open("/content/cat.jpg").resize((224,224))
img = np.array(img)/255.0
def advanced_augment(image):
    # Brightness
    brightness = random.uniform(0.7,1.3)
    image = np.clip(image * brightness,0,1)
    # Contrast
    contrast = random.uniform(0.8,1.5)
    mean = np.mean(image)
    image = np.clip((image-mean)*contrast + mean,0,1)
    # Horizontal flip
    if random.random()>0.5:
        image = np.fliplr(image)
    # Random crop
    h,w,_ = image.shape
    start_x = random.randint(0, w-180)
    start_y = random.randint(0, h-180)
    image = image[start_y:start_y+180, start_x:start_x+180]
    # Gaussian noise
    noise = np.random.normal(0,0.05,image.shape)
    image = np.clip(image+noise,0,1)
    return image
aug = advanced_augment(img)
plt.imshow(aug)
plt.axis("off")
plt.show()
