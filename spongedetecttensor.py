from os import listdir, path
from random import shuffle
import numpy as np
from skimage import io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

def main():
    data_path = 'data/'
    image_size = (256, 256)
    batch_size = 16

    training, validation = prep_data(data_path=data_path, image_size=image_size, batch=batch_size)
    build_model(training=training, validation=validation)

def prep_data(data_path, image_size, batch):
    training = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        labels="inferred",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch,
    )

    validation = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch,
    )
    return training, validation

def build_model(training, validation):
    model = keras.Sequential([
            layers.Conv2D(32, 3, input_shape=(256, 256, 3)),
            layers.MaxPooling2D(pool_size=2),
            # layers.Conv2D(16, 3, input_shape=(256, 256, 3)),
            # layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(4, activation='softmax'),
            ])

    model.compile(
        'adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    history = model.fit(training,
                    shuffle=True,
                    validation_data=validation,
                    steps_per_epoch=5,
                    epochs=5,
                    validation_steps=5,
                    verbose=2)

