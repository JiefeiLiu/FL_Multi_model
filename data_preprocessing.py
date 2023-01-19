import numpy as np
import pandas as pd


# -----------------------------------
# Read CICDDoS2019 data
def read_2019_data(path):
    X_train = np.load(path + "X_samp_train_mult500k.npy")
    X_test = np.load(path + "X_samp_test_mult500k.npy")
    y_train = np.load(path + "y_samp_train_mult500k.npy")
    y_test = np.load(path + "y_samp_test_mult500k.npy")

    # print("X train shape: ", X_train.shape)
    # print("y train shape: ", y_train.shape)
    # print("X test shape: ", X_test.shape)
    # print("y test shape: ", y_test.shape)

    '''re-split the training and testing'''
    # X = np.concatenate((X_train, X_test), axis=0)
    # y = np.concatenate((y_train, y_test), axis=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True, stratify=y)

    # '''binary class classification encoding'''
    # y_train[y_train > 0] = 1
    # y_test[y_test > 0] = 1

    # Verify
    unique, counts = np.unique(y_train, return_counts=True)
    # print(dict(zip(unique, counts)))
    # print(y_train)
    # print(str(len(y_test) / (len(y_train) + len(y_test))))
    return (X_train, y_train), (X_test, y_test)


# -----------------------------------
# Read CICIDS2017 data
def read_2017_data(path):
    # multi-class classification
    X_train = np.load(path + "x_train_mul_samp.npy")
    X_test = np.load(path + "x_test.npy")
    y_train = np.load(path + "y_train_mul_samp.npy")
    y_test = np.load(path + "y_test_mul.npy")

    # print("X train shape: ", X_train.shape)
    # print("y train shape: ", y_train.shape)
    # print("X test shape: ", X_test.shape)
    # print("y test shape: ", y_test.shape)
    # print(len(np.unique(y_train)))
    # print(y_train)
    # print(str(len(y_test) / (len(y_train) + len(y_test))))
    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    pass