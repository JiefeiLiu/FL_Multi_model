import sys
sys.path.append("..")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



def cic2017_feature_selection(X, y, n_features=41):
    # Drop columns with value -1
    # X_dropped = np.delete(X, [18, 56, 57], 1)
    # print(X_dropped[X_dropped < 0])
    # print(np.where(X_dropped < 0))
    # sys.exit()
    # replace negative value to zeros
    X[X < 0] = 0
    # feature selection
    features = SelectKBest(score_func=chi2, k=n_features)
    fit = features.fit(X, y)
    X_selected = fit.transform(X)
    return X_selected, y


def cic_2017_client_normalize(training_data_list, X_test, y_test, X_val, y_val):
    # normalize training data
    normed_training_data_list = []
    for training_data in training_data_list:
        temp_X, temp_y = training_data
        X_norm_train = MinMaxScaler().fit_transform(temp_X)
        normed_training_data_list.append((X_norm_train, temp_y))
    # normalize testing data
    X_norm_test = MinMaxScaler().fit_transform(X_test)
    # normalize validation data
    X_norm_val = MinMaxScaler().fit_transform(X_val)
    return normed_training_data_list, (X_norm_test, y_test), (X_norm_val, y_val)


# -----------------------------------
# Read CICIDS2017 data
def read_2017_data_for_FL(path):
    # # multi-class classification
    # X_train = np.load(path + "x_tr_dos-sl-hk_ddos_bf_pr_f40.npy")
    # y_train = np.load(path + "y_tr_mul_dos-sl-hk_ddos_bf_pr_f40.npy")
    # X_test = np.load(path + "x_ts_dos-sl-hk_ddos_bf_pr_f40.npy")
    # y_test = np.load(path + "y_ts_mul_dos-sl-hk_ddos_bf_pr_f40.npy")

    X = np.load(path + "cic17_all_X_org.npy")
    y = np.load(path + "cic17_all_y_org.npy")

    # feature selection
    X_s, y = cic2017_feature_selection(X, y, 40)

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    unique, counts = np.unique(y, return_counts=True)
    print("Total data shape", dict(zip(unique, counts)))


    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.10, random_state=1, shuffle=True, stratify=y)
    # validation/noise data generator
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.23, random_state=1, shuffle=True,
                                                      stratify=y_train)

    unique, counts = np.unique(y_train, return_counts=True)
    print("Training shape", dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print("Testing shape", dict(zip(unique, counts)))
    unique, counts = np.unique(y_val, return_counts=True)
    print("Validation shape", dict(zip(unique, counts)))
    # print(str(len(y_test) / (len(y_train) + len(y_test))))
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


if __name__ == '__main__':
    path = '../2017_data/'
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = read_2017_data_for_FL(path)
    # print(X_val.head())
