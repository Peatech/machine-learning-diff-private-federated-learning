import os
import numpy as np
import pickle

def read(dataset="training", path="."):
    if dataset == "training":
        fname_img = os.path.join(path, 'train_images.npy')
        fname_lbl = os.path.join(path, 'train_labels.npy')
    elif dataset == "testing":
        fname_img = os.path.join(path, 'test_images.npy')
        fname_lbl = os.path.join(path, 'test_labels.npy')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    print(fname_lbl)

    # Load the data from .npy files
    img = np.load(fname_img)
    lbl = np.load(fname_lbl)

    # Reshape and normalize
    img = np.reshape(img, [img.shape[0], img.shape[1] * img.shape[2]]) * 1.0 / 255.0

    return img, lbl


def get_data(d):
    # Load the data
    x_train, y_train = read('training', d + '/MNIST_original')
    x_test, y_test = read('testing', d + '/MNIST_original')

    # Create validation set
    x_vali = list(x_train[50000:].astype(float))
    y_vali = list(y_train[50000:].astype(float))

    # Create training set
    x_train = x_train[:50000].astype(float)
    y_train = y_train[:50000].astype(float)

    # Sort training set (to make federated learning non-i.i.d.)
    indices_train = np.argsort(y_train)
    sorted_x_train = list(x_train[indices_train])
    sorted_y_train = list(y_train[indices_train])

    # Create test set
    x_test = list(x_test.astype(float))
    y_test = list(y_test.astype(float))

    return sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test


class Data:
    def __init__(self, save_dir, n):
        raw_directory = save_dir + '/DATA'
        self.client_set = pickle.load(open(raw_directory + '/clients/' + str(n) + '_clients.pkl', 'rb'))
        self.sorted_x_train, self.sorted_y_train, self.x_vali, self.y_vali, self.x_test, self.y_test = get_data(save_dir)
