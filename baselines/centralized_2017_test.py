import sys

from sqlalchemy.dialects.mssql.information_schema import columns

sys.path.append("..")
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
import os
import copy
import random
import torch
from torch import nn
import time
from torch.utils.data import DataLoader
import utils
import models
from data_utils import CustomDataset
import single_client_test

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Set cuda
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cuda_num = random.randint(0, (torch.cuda.device_count()-1))
    cuda_name = "cuda:" + str(cuda_num)
    DEVICE = torch.device(cuda_name)


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


def feature_process(data_df):
    selected_features = [' Flow Duration', 'Bwd Packet Length Max', ' Bwd Packet Length Mean', ' Bwd Packet Length Min', ' Bwd Packet Length Std',
                         ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean',
                         ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
                         ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', 'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length',
                         ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', ' SYN Flag Count', ' RST Flag Count',
                         ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' Average Packet Size', ' Avg Fwd Segment Size',
                         ' Avg Bwd Segment Size', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', 'Idle Mean', ' Idle Std',
                         ' Idle Max', ' Idle Min']
    train_x = data_df.loc[:, selected_features].to_numpy()
    train_y = data_df['label'].to_numpy()
    return train_x, train_y



# -----------------------------------
# Read CICIDS2017 data
def read_2017_data_for_FL(path):
    # # multi-class classification
    # X_train = np.load(path + "x_tr_dos-sl-hk_ddos_bf_pr_f40.npy")
    # y_train = np.load(path + "y_tr_mul_dos-sl-hk_ddos_bf_pr_f40.npy")
    # X_test = np.load(path + "x_ts_dos-sl-hk_ddos_bf_pr_f40.npy")
    # y_test = np.load(path + "y_ts_mul_dos-sl-hk_ddos_bf_pr_f40.npy")
    ##############################################
    X_s = np.load(path + "cic17_all_X_org.npy")
    y = np.load(path + "cic17_all_y_org.npy")

    # # feature selection
    # X_s, y = cic2017_feature_selection(X, y, 41)
    ##############################################

    print("X shape: ", X_s.shape)
    print("y shape: ", y.shape)
    unique, counts = np.unique(y, return_counts=True)
    print("Total data shape", dict(zip(unique, counts)))

    # sys.exit()
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
    (X_train, y_train), (X_test, y_test), (_, _) = read_2017_data_for_FL(path)
    # print(X_val.head())
    #################### Testing on FLNET dataset #######################
    # # Load FLNET testing data
    # FLNET_data_path = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/MILCOM/data/un_use_data/"
    # X_test = np.load(FLNET_data_path + "FLNET_X_test.npy")
    # y_test = np.load(FLNET_data_path + "FLNET_y_test.npy")
    # # Replace label to match 2017 data
    # y_test[y_test == 5] = 4
    # y_test[y_test == 6] = 5
    # # Verify
    # unique, counts = np.unique(y_test, return_counts=True)
    # print("FLNET Testing data distribution:", dict(zip(unique, counts)))
    # print("training data shape: ", X_train.shape)
    # print("test data shape: ", X_test.shape)
    ###########################################
    # sys.exit()
    # define parameters
    epochs = 50
    learning_rate = 0.01
    batch_size = 64
    MLP_first_hidden = 64
    MLP_second_hidden = 128
    num_classes = 8
    classification = "Multi"
    neural_network = "MLP_Mult"
    train_data = CustomDataset(X_train, y_train, neural_network)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # x_test_new, y_test_new = utils.testing_data_extraction(testing_data, y_train)
    test_data = CustomDataset(X_test, y_test, neural_network)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # -------------------Create folder for model----------------------
    curr_path = os.getcwd()
    utils.make_dir(curr_path, "../results")
    # -------------- Define model ----------------------
    model = models.MLP_Mult(input_shape=X_train.shape[1], first_hidden=MLP_first_hidden,
                            second_hidden=MLP_second_hidden, num_classes=num_classes).to(DEVICE).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()  # Muti class classification
    # -------------------Testing model----------------------
    test_time = time.time()
    model_weights = torch.load("../models/centralized_2017_model.pt")
    model.load_state_dict(copy.deepcopy(model_weights))
    loss, accuracy, f1, precision, recall = single_client_test.test(model, loss_fn, test_loader, curr_path,
                                                                    device=DEVICE)
    print("Accuracy, precision, recall, f1, loss, training_time: ", accuracy, precision, recall, f1, loss)
    print("---Testing time: %s minutes. ---" % ((time.time() - test_time) / 60))