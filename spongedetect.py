from os import listdir, path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
import sklearn.svm
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
# from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sklearn.cluster
from collections import Counter

from pylab import show

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from skimage import io
from skimage import filters
from skimage import color

from plotmatrix import plotMatrix


data_path = 'data'


# the dataset
X = []
y = []
for label_class in listdir(data_path):
    folder_path = path.join(data_path, label_class)

    # get each file
    for spongeimage in listdir(folder_path):

        # rate, data = scipy.io.wavfile.read(path.join(folder_path, spongeimage))
        # print(f"loaded file '{wav_file}' ({rate} Hz, {data.shape[-1]} samples)")

        # load image from file
        image = io.imread(path.join(folder_path, spongeimage))

        # create 2d image with 3 channels, RGB
        image_2d = image.reshape(image.shape[0] * image.shape[1], 3)

        # Create grayscale image for edge detection
        grayscale_image = color.rgb2gray(image)

        clf = sklearn.cluster.KMeans(n_clusters = 5)
        colors = clf.fit_predict(image_2d)

        # calculating horizontal edges using prewitt kernel
        prewitt_horizontal_edges = filters.prewitt_h(grayscale_image)
        prewitt_horizontal_edges = prewitt_horizontal_edges.reshape([len(prewitt_horizontal_edges) * len(prewitt_horizontal_edges[0])])

        # # calculating vertical edges using prewitt kernel
        # prewitt_vertical_edges = filters.prewitt_v(grayscale_image)
        # prewitt_vertical_edges = prewitt_vertical_edges.reshape([len(prewitt_vertical_edges) * len(prewitt_vertical_edges[0])])

        # n_components = 30
        # pca = sklearn.decomposition.PCA(n_components=n_components, whiten=True).fit(image)
        # fastICA = sklearn.decomposition.FastICA(n_components=n_components, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None, max_iter=300, tol=0.001, random_state=1)

        # X_transformed = fastICA.fit_transform(grayscale_image)
        # io.imshow(fastICA.inverse_transform(X_transformed))
        # show()
        # X_transformed = X_transformed.reshape([len(X_transformed) * len(X_transformed[0])])

        # print(X_transformed.shape)
        # print(type(X_transformed))


        # todo extract feature, then put the feature in this
        X.append(prewitt_horizontal_edges + colors)
        y.append(label_class)

# convert list of numpy arrays to one large numpy array
# to enable fancy indexing below
X = np.array(X)

# manual label encoding because sklearn is bad
encoded_labels = {
    'spongebob': 0,
    'patrick': 1,
    'squidward': 2,
    # 'krab': 3,
}
y = np.fromiter(map(lambda label: encoded_labels[label], y), dtype=int)
class_labels = encoded_labels.keys()

# accumulate overall results
y_test_all = None
y_pred_all = None
cm_all = None

# do k-fold cross validation, holding out 20% (1/5th) as test
# taking care to stratify the train and test sets
# such that the class balance is preserved
skf = StratifiedKFold(n_splits=3, shuffle=True)
for fold_num, (train_index, test_index) in enumerate(skf.split(X, y)):
    # slice out train, test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_train)

    # Standardize features by removing the mean and scaling to unit variance.
    X_train = scaler.transform(X_train)  
    # apply same transformation to test data
    X_test = scaler.transform(X_test)

    # a new, untrained model
    # model = sklearn.svm.SVC(class_weight='balanced')
    # model = sklearn.neural_network.MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    model = keras.Sequential(
    [
        layers.Dense(16, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    model.fit(X_train, y_train)

    # compute training error by predicting each item in the training set
    # using the model that was just trained on the training set
    y_pred = model.predict(X_test)

    # compare y_train with y_pred to see how accurate things are
    print(f"fold {fold_num+1} accuracy: {sklearn.metrics.accuracy_score(y_test, y_pred):.3f}")
    print(f"fold {fold_num+1} balanced accuracy: {sklearn.metrics.balanced_accuracy_score(y_test, y_pred):.3f}")

    # confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    if cm_all is None:
        y_true_all = y_test
        y_pred_all = y_pred
        cm_all = np.zeros_like(cm)
    else:
        y_true_all = np.concatenate((y_true_all, y_test))
        y_pred_all = np.concatenate((y_pred_all, y_pred))
    cm_all += cm
    print(f"fold {fold_num+1} confusion matrix:\n{cm}\n")

print(f"overall accuracy: {sklearn.metrics.accuracy_score(y_true_all, y_pred_all):.3f}")
print(f"overall balanced accuracy: {sklearn.metrics.balanced_accuracy_score(y_true_all, y_pred_all):.3f}")
print(f"overall confusion matrix:\n{cm_all}")
print(class_labels)

plotMatrix(class_labels, cm_all, 'confusion_matrix', 'Confusion Matrix')
