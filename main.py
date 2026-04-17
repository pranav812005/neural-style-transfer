import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def load_image(path):
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    img = img[np.newaxis, ...]
    return img

# load images
content = load_image("content.jpg")
style = load_image("style.jpg")

# load model
model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# stylize
stylized = model(content, style)[0]

# show image
plt.imshow(stylized[0])
plt.axis('off')
plt.show()

# 🔥 FORCE SAVE (important fix)
output_path = os.path.abspath("output.png")
img = tf.keras.preprocessing.image.array_to_img(stylized[0])
img.save(output_path)

print("✅ FILE SAVED HERE:", output_path)