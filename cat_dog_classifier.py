# Название файла: cat_dog_classifier.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

train_dir = 'path/to/cats_and_dogs/train'
validation_dir = 'path/to/cats_and_dogs/validation'
test_dir = 'path/to/cats_and_dogs/test'

train_datagen = ImageDataGenerator(
    rescale=1. / 255
)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255
)

train_data_gen = train_datagen.flow_from_directory(
    directory=train_dir,
    batch_size=BATCH_SIZE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

validation_data_gen = validation_datagen.flow_from_directory(
    directory=validation_dir,
    batch_size=BATCH_SIZE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

test_data_gen = test_datagen.flow_from_directory(
    directory=test_dir,
    batch_size=1,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode=None,
    shuffle=False
)


# Функция для отображения изображений
def plot_images(images, probs=None):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        if probs is not None:
            plt.title(f'Prob: {probs[i]:.2f}')
    plt.show()


# Создание генератора изображений для обучения с расширением данных
train_image_generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data_gen = train_image_generator.flow_from_directory(
    directory=train_dir,
    batch_size=BATCH_SIZE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

# Построение модели сверточной нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=10,
    validation_data=validation_data_gen,
    validation_steps=validation_data_gen.samples // BATCH_SIZE
)


# Визуализация точности и потерь модели
def plot_accuracy_and_loss(history):
    plt.figure(figsize=(12, 4))

    # Точность
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    # Потери
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.show()


plot_accuracy_and_loss(history)


# Использование модели для предсказания новых изображений
def predict_images(model, test_data_gen):
    test_images, _ = next(test_data_gen)
    predictions = model.predict(test_images)
    probabilities = predictions.flatten()

    plot_images(test_images, probabilities)
    return probabilities


probabilities = predict_images(model, test_data_gen)

print('Challenge complete!')
