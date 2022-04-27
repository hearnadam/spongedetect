from os import listdir, path
import numpy as np
from skimage import io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


data_path = 'data/out/data.npz'
data_path = 'data/'


image_size = (256, 256)
batch_size = 16

training = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    labels="inferred",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

validation = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

model = keras.Sequential([
        layers.Conv2D(32, 3, input_shape=(256, 256, 3)),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(3, activation='softmax'),
        ])

model.compile(
    'adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

history = model.fit(training,
                validation_data=validation,
                steps_per_epoch=5,
                epochs=100,
                validation_steps=10,
                verbose=2)

