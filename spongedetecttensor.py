from os import listdir, path
from random import sample, shuffle
import numpy as np
from skimage import io
import sklearn.metrics
from plotmatrix import plotMatrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from keras_visualizer import visualizer 

def main():
    data_path = 'data/'
    image_size = (256, 256)
    batch_size = 16

    training, validation, class_weight = prep_data(data_path=data_path, image_size=image_size, batch=batch_size)

    tuner = kt.Hyperband(build_model,
                            objective='val_accuracy',
                            max_epochs=10,
                            factor=3,
                            directory="model",
                            project_name="spongedetect3")



    tuner.search(training, epochs=2, validation_data=validation, class_weight=class_weight)
    # tuner.results_summary()

    # tuner.get_best_models()[0].summary()

    # let's run some predictions biatch

    to_predict = []
    label = []

    predict_folder = 'predict'
    for directory in listdir(predict_folder):
        sub_dir = path.join(predict_folder, directory)

        for spongeimage in listdir(sub_dir):
            # load image from file
            image = io.imread(path.join(sub_dir, spongeimage))
            # image = skimage.transform.resize(image, (256, 256), anti_aliasing=True)
            # normalize the image 
            # image = image / 256
            # print(spongeimage)
            image = np.array(image)
            print(image.shape)

            to_predict.append(image)
            label.append(directory)


    to_predict = np.array(to_predict)


    print(label)
    print(validation.class_names)
    # get all predicted values by model
    all_predictions = tuner.get_best_models()[0].predict(to_predict)

    # take largest (most confident) prediction for each image
    y_pred = list(map(lambda predictions: np.where(predictions == max(predictions))[0][0], all_predictions))
    y_actual = list(map(lambda a_label: validation.class_names.index(a_label), label))

    print(y_pred)
    print(y_actual)
    matrix = sklearn.metrics.confusion_matrix(y_actual, y_pred)
    print(matrix)
    plotMatrix(validation.class_names, matrix, 'confusion_matrix.png', 'Confusion Matrix')


    visualizer(tuner.get_best_models()[0], format='png', view=False, filename='test')
    # visualizer(tuner.get_best_models()[1], format='pdf', view=True)

    tuner.get_best_models()[0].save('saved/model/')


def prep_data(data_path, image_size, batch):
    # generate class weights based on how many files are in each folder.
    class_weight = {}
    total_images = 0
    for directory in listdir(data_path):
        images = len(listdir(path.join(data_path, directory)))
        class_weight.update({directory: images})
        total_images += images

    for key, value in class_weight.items():
        class_weight[key] = (total_images / len(class_weight)) / value

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

    # renames classes in class_weights to int values
    for i in range(len(validation.class_names)):
        value = class_weight.pop(validation.class_names[i])
        class_weight.update({i: value})

    return training, validation, class_weight

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