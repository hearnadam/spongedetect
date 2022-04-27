from os import listdir, path
import numpy as np
from skimage import io


data_path = 'data'


# the dataset
X = []
y = []
for label_class in listdir(data_path):
    folder_path = path.join(data_path, label_class)

    # get each file
    for spongeimage in listdir(folder_path):

        # load image from file
        image = io.imread(path.join(folder_path, spongeimage))

        # todo extract feature, then put the feature in this
        X.append(np.array(image))
        y.append(label_class)
        

y = np.array(y)

# normalization
X = np.array(X)
X = X / 255.0

np.savez("./data/out/data.npz", x=X, y=y)
