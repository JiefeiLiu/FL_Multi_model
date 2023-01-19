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
    # --------------------Parameter Setting-----------------------
    # clients hyperparameter setting
    client_epochs = 10
    learning_rate = 0.01
    batch_size = 64
    # Server hyperparameter setting
    num_clients = 30
    rounds = 5
    fraction = 0.9
    # Setting parameters
    neural_network = "MLP_Mult"
    # --------------------Logging setting-----------------------
    curr_path = os.getcwd()
    utils.make_dir(curr_path, "log_file")
    log_name = "log_file/" + "FL" + "_NN_" + neural_network + "_clients_" + str(num_clients) + "_epochs_" + str(client_epochs) + "_rounds_" + str(rounds) + "_fraction_" + str(fraction) + "_date_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
    logging.basicConfig(filename=log_name, format='%(asctime)s - %(message)s', level=logging.INFO)
    # --------------------Data Loading-----------------------
    # data_dir = "/home/jliu/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    data_dir = "/home/jliu/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    pickle_dir = "/home/jliu/DoD_Misra_project/jiefei_liu/DOD/MLP_model/partition.pkl"
    num_classes = 11
    print("Loading data...")
    (x_train_un_bin, y_train_un_bin), (x_test, y_test_bin) = data_preprocessing.read_2019_data(data_dir)
    # partition_data_list = sampling.partition_bal_equ(x_train_un_bin, y_train_un_bin, num_clients)
    # Load partitioned data
    with open(pickle_dir, 'rb') as file:
        # Call load method to deserialze
        partition_data_list = pickle.load(file)
    test_data = CustomDataset(x_test, y_test_bin, neural_network)
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
    # --------------------initialize weight----------------------
    w_glob = glob_model.state_dict()
    # --------------------Server Training-----------------------
    # Record running time
    start_time = time.time()
    # for loop for FL around
    for iter in range(rounds):
        print("Rounds ", iter, "....")
        Round_time = time.time()
        w_clients, loss_clients, ac_clients = [], [], []
        # random select clients based on fraction
        num_clients_with_fraction = max(int(fraction * num_clients), 1)
        clients_index = np.random.choice(range(num_clients), num_clients_with_fraction, replace=False)
        temp_client_list = []
        # Create clients
        for index in clients_index:
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
            client = CustomThread(target=utils.train, args=(temp_local_model, local_optimizer, loss_fn, train_loader, client_epochs, neural_network, DEVICE,))
            temp_client_list.append(client)
        # run clients simultaneously
        for client_index in temp_client_list:
            client_index.start()
        # wait clients finish
        for client_index in temp_client_list:
            local_weights = client_index.join()
            w_clients.append(copy.deepcopy(local_weights))
        # Global model weight updates
        w_glob_last = copy.deepcopy(w_glob)
        w_glob = FedAvg(w_clients)
        # Update global model
        glob_model.load_state_dict(w_glob)
        # --------------------Server Round Testing-----------------------
        round_loss, round_accuracy = utils.test(glob_model, loss_fn, test_loader, neural_network, device=DEVICE)
        logging.info('Round %d, Loss %f, Accuracy %f, Round Running time(min): %s', iter, round_loss, round_accuracy, ((time.time() - Round_time) / 60))
    print("---Server running time: %s minutes. ---" % ((time.time() - start_time) / 60))
    # --------------------Server Testing-----------------------
    test_time = time.time()
    loss, accuracy = utils.test(glob_model, loss_fn, test_loader, neural_network, device=DEVICE)
    server_running_time = ((time.time() - test_time) / 60)
    logging.info('Global model, Loss %f, Accuracy %f, Total Running time(min): %s', loss, accuracy, server_running_time)
    print("---Server testing time: %s minutes. ---" % server_running_time)
    print("Finish.")
