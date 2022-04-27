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

    tuner = kt.Hyperband(build_model,
                            objective='val_accuracy',
                            max_epochs=10,
                            factor=3,
                            directory="model",
                            project_name="spongedetect")

    tuner.search(training, epochs=2, validation_data=validation)
    tuner.results_summary()

    # model = build_model()
    # history = model.fit(training,
    #             shuffle=True,
    #             validation_data=validation,
    #             steps_per_epoch=5,
    #             epochs=5,
    #             validation_steps=5,
    #             verbose=2)


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

def build_model(hyper_parameters):
    model = keras.Sequential()
    model.add(layers.Conv2D(32, 3, input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(16, 3, input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D(pool_size=2))

    if hyper_parameters.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='softmax'))


    learning_rate = hyper_parameters.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


if __name__ == '__main__':
    main()