import tensorflow as tf
from PIL import Image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("model/brain_tumor_model.keras")

# Class names
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Enter MRI image path
image_path = input("Enter MRI image path: ")

# Load and resize image
image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

# Convert image to array
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Predict
prediction = model.predict(image_array)
predicted_class = class_names[np.argmax(prediction)]

print("Predicted result:", predicted_class)