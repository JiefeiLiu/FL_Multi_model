import os
import sys
import time
import random
import torch
import copy
import numpy as np
import pickle
import logging
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader

import utils
import models
import sampling
import data_preprocessing
from data_utils import CustomDataset
from multi_threading import CustomThread
from aggregation_functions import FedAvg


# Set cuda
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cuda_num = random.randint(0, (torch.cuda.device_count()-1))
    cuda_name = "cuda:" + str(cuda_num)
    DEVICE = torch.device(cuda_name)


if __name__ == '__main__':
    print(DEVICE, " are using for training and testing.")
    # clients hyperparameter setting
    client_epochs = 10
    learning_rate = 0.01
    batch_size = 64
    # Server hyperparameter setting
    num_clients = 30
    rounds = 5
    fraction = 1.0
    # Setting parameters
    neural_network = "MLP_Mult"
    # --------------------Data Loading-----------------------
    data_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    pickle_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/MLP_model/data/partition_attacks_2_imbalance.pkl"
    num_classes = 11
    print("Loading data...")
    (x_train_un_bin, y_train_un_bin), (x_test, y_test_bin) = data_preprocessing.read_2019_data(data_dir)
    # Load partitioned data
    with open(pickle_dir, 'rb') as file:
        # Call load method to deserialze
        partition_data_list = pickle.load(file)
    # extract subset of training data
    testing_label = []
    for i in range(num_clients):
        (temp_test_x, temp_test_y) = partition_data_list[i]
        testing_label = np.concatenate([testing_label, temp_test_y])
    # extract corresponding testing data
    new_testing_x, new_testing_y = data_preprocessing.testing_data_extraction(data_dir, testing_label)
    test_data = CustomDataset(new_testing_x, new_testing_y, neural_network)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # --------------Build global model and Select loss function----------------------
    if neural_network == "MLP":
        glob_model = models.MLP(input_shape=x_train_un_bin.shape[1]).to(DEVICE)
        loss_fn = nn.BCELoss()  # Binary classification
    elif neural_network == "MLP_Mult":
        glob_model = models.MLP_Mult(input_shape=x_train_un_bin.shape[1], num_classes=num_classes).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()  # Muti class classification
    else:
        print("Wrong neural network type, exit.")
        sys.exit()
    optimizer = torch.optim.SGD(glob_model.parameters(), lr=learning_rate)
    glob_model.train()
    # --------------initialize weight----------------------
    w_glob = glob_model.state_dict()
    global_weight_record = []
    clients_weight_record = []
    # --------------------Server Training-----------------------
    # Record running time
    start_time = time.time()
    # for loop for FL around
    for iter in range(rounds):
        print("Rounds ", iter, "....")
        Round_time = time.time()
        w_clients, loss_clients, ac_clients = [], [], []
        temp_client_list = []
        for index in range(num_clients):
            # Get clients data
            (client_X_train, client_y_train) = partition_data_list[index]
            # process data
            train_data = CustomDataset(client_X_train, client_y_train, neural_network)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            # copy global model
            temp_local_model = copy.deepcopy(glob_model)
            # define local optimizer
            local_optimizer = torch.optim.SGD(temp_local_model.parameters(), lr=learning_rate)
            # create threads which represents clients
            client = CustomThread(target=utils.train, args=(
            temp_local_model, local_optimizer, loss_fn, train_loader, client_epochs, neural_network, DEVICE,))
            temp_client_list.append(client)
        # run clients simultaneously
        for client_index in temp_client_list:
            client_index.start()
        # wait clients finish
        for client_index in temp_client_list:
            local_weights = client_index.join()
            w_clients.append(copy.deepcopy(local_weights))
        # Global model weight updates
        global_weight_record.append(copy.deepcopy(w_glob))
        clients_weight_record.append(copy.deepcopy(w_clients))
        w_glob = FedAvg(w_clients)
        # Update global model
        glob_model.load_state_dict(w_glob)
        # --------------------Server Round Testing-----------------------
        round_loss, round_accuracy, f1, precision, recall = utils.test(glob_model, loss_fn, test_loader, neural_network, device=DEVICE)
        print('Round %d, Loss %f, Accuracy %f, Round Running time(min): %s' % (iter, round_loss, round_accuracy,
                     ((time.time() - Round_time) / 60)))
    # --------------------Save Records-----------------------
    # save records
    with open('global_weight_records_imbalance.pkl', 'wb') as file:
    # A new file will be created
        pickle.dump(global_weight_record, file)
    with open('client_weight_records_imbalance.pkl', 'wb') as file:
    # A new file will be created
        pickle.dump(clients_weight_record, file)
    # print("---Server running time: %s minutes. ---" % ((time.time() - start_time) / 60))
    # --------------------Server Testing-----------------------
    test_time = time.time()
    loss, accuracy, f1, precision, recall = utils.test(glob_model, loss_fn, test_loader, neural_network, device=DEVICE)
    server_running_time = ((time.time() - test_time) / 60)
    print("Global model, Loss %f, Accuracy %f, F1 %f, Total Running time(min): %s" % (loss, accuracy, f1, server_running_time))
    # print("---Server testing time: %s minutes. ---" % server_running_time)
    print("Finish.")
