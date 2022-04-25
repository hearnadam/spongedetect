from os import listdir, path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
import sklearn.svm
import sklearn.neighbors

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

        # edge detection
        grayscale_image = color.rgb2gray(image)

        # calculating horizontal edges using prewitt kernel
        prewitt_horizontal_edges = filters.prewitt_h(grayscale_image)
        prewitt_horizontal_edges = prewitt_horizontal_edges.reshape([len(prewitt_horizontal_edges) * len(prewitt_horizontal_edges[0])])

        # calculating vertical edges using prewitt kernel
        prewitt_vertical_edges = filters.prewitt_v(grayscale_image)
        prewitt_vertical_edges = prewitt_vertical_edges.reshape([len(prewitt_vertical_edges) * len(prewitt_vertical_edges[0])])


        # todo extract feature, then put the feature in this
        X.append(prewitt_horizontal_edges)
        y.append(label_class)

# convert list of numpy arrays to one large numpy array
# to enable fancy indexing below
X = np.array(X)

# manual label encoding because sklearn is bad
encoded_labels = {
    'spongebob': 0,
    'patrick': 1,
    'squidward': 2,
    'krab': 3,
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
skf = StratifiedKFold(n_splits=2, shuffle=True)
for fold_num, (train_index, test_index) in enumerate(skf.split(X, y)):
    # slice out train, test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # a new, untrained model
    model = sklearn.svm.SVC(class_weight='balanced')
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
