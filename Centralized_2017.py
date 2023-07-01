import sys

import pandas as pd
import numpy as np
import pickle
import os
import copy
import random
import torch
from torch import nn
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import utils
import models
from data_utils import CustomDataset
import single_client_test
import data_preprocessing
import centralized_2017_test


# Set cuda
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cuda_num = random.randint(0, (torch.cuda.device_count()-1))
    cuda_name = "cuda:" + str(cuda_num)
    DEVICE = torch.device(cuda_name)


# # -----------------------------------
# # Read CICIDS2017 data
# def read_2017_data_for_FL(path):
#     # multi-class classification
#     X_train = np.load(path + "x_tr_dos-sl-hk_ddos_bf_pr_f40.npy")
#     y_train = np.load(path + "y_tr_mul_dos-sl-hk_ddos_bf_pr_f40.npy")
#     X_test = np.load(path + "x_ts_dos-sl-hk_ddos_bf_pr_f40.npy")
#     y_test = np.load(path + "y_ts_mul_dos-sl-hk_ddos_bf_pr_f40.npy")
#
#     print("X training shape: ", X_train.shape)
#     print("y training shape: ", y_train.shape)
#     print("X training shape: ", X_test.shape)
#     print("y training shape: ", y_test.shape)
#     unique, counts = np.unique(y_train, return_counts=True)
#     print("Training data shape", dict(zip(unique, counts)))
#     unique, counts = np.unique(y_test, return_counts=True)
#     print("Testing data shape", dict(zip(unique, counts)))
#
#     '''re-split the training and testing'''
#     X = np.concatenate((X_train, X_test), axis=0)
#     y = np.concatenate((y_train, y_test), axis=0)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=True, stratify=y)
#     # validation/noise data generator
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.23, random_state=1, shuffle=True,
#                                                       stratify=y_train)
#
#     unique, counts = np.unique(y_train, return_counts=True)
#     print("Training shape", dict(zip(unique, counts)))
#     unique, counts = np.unique(y_test, return_counts=True)
#     print("Testing shape", dict(zip(unique, counts)))
#     unique, counts = np.unique(y_val, return_counts=True)
#     print("Validation shape", dict(zip(unique, counts)))
#     # print(str(len(y_test) / (len(y_train) + len(y_test))))
#     return (X_train, y_train), (X_test, y_test), (X_val, y_val)
# #
#
# def normalization(X_train, y_train, X_test, y_test):
#     # normalize training data
#     X_norm_train = MinMaxScaler().fit_transform(X_train)
#     # normalize testing data
#     X_norm_test = MinMaxScaler().fit_transform(X_test)
#     return X_norm_train, y_train, X_norm_test, y_test


if __name__ == '__main__':
    # pickle_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/MLP_model/data/partition.pkl"
    # Load partitioned data
    # data_dir = '2017_data/20_training.pkl'
    # with open(data_dir, 'rb') as file:
    #     # Call load method to deserialze
    #     partition_data_list = pickle.load(file)
    # for index in range(len(partition_data_list)):
    #     (client_X_train, client_y_train) = partition_data_list[index]
    #     # Verify
    #     unique, counts = np.unique(client_y_train, return_counts=True)
    #     print("Client ", str(index), "training shape", dict(zip(unique, counts)))
    # sys.exit()
    print("Random pick", DEVICE, "are using for training and testing.")
    data_dir = '2017_data/'
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = centralized_2017_test.read_2017_data_for_FL(data_dir)
    # (X_train, y_train), (X_test, y_test), (X_val, y_val) = data_preprocessing.read_2017_data_for_FL((data_dir))
    # define parameters
    epochs = 200
    learning_rate = 0.01
    batch_size = 64
    MLP_first_hidden = 64
    MLP_second_hidden = 128
    num_classes = 8
    # num_classes = 7
    classification = "Multi"
    neural_network = "MLP_Mult"
    train_data = CustomDataset(X_train, y_train, neural_network)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # x_test_new, y_test_new = utils.testing_data_extraction(testing_data, y_train)
    test_data = CustomDataset(X_test, y_test, neural_network)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # -------------------Create folder for model----------------------
    curr_path = os.getcwd()
    utils.make_dir(curr_path, "results")
    # -------------- Define model ----------------------
    model = models.MLP_Mult(input_shape=X_train.shape[1], first_hidden=MLP_first_hidden,
                            second_hidden=MLP_second_hidden, num_classes=num_classes).to(DEVICE).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()  # Muti class classification
    # -------------------Training model----------------------
    train_time = time.time()
    model_weights, _ = utils.train(model, optimizer, loss_fn, train_loader, epochs, neural_network, 1, device=DEVICE)
    training_time = (time.time() - train_time) / 60
    print("---Training time: %s minutes. ---" % training_time)
    # -------------------Testing model----------------------
    test_time = time.time()
    model.load_state_dict(copy.deepcopy(model_weights))
    loss, accuracy, f1, precision, recall = single_client_test.test(model, loss_fn, test_loader, curr_path, device=DEVICE)
    print("Accuracy, precision, recall, f1, loss, training_time: ", accuracy, precision, recall, f1, loss, training_time)
    print("---Testing time: %s minutes. ---" % ((time.time() - test_time) / 60))