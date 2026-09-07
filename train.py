import tensorflow as tf

train_path = "dataset/Training"
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(224, 224),
    batch_size=32
)
class_names = train_data.class_names
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)
print("Images prepared successfully!")
print("Classes:", class_names)
# Create CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                           input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")
])

print("CNN model created successfully!")
# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully!")

# Train the model
history = model.fit(
    train_data,
    epochs=10
)

model.save("model/brain_tumor_model.keras")
print("Model training completed!")