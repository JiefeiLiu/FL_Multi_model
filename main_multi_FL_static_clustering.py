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

import aggregation_functions
import utils
import similarity_utils
import models
import sampling
import data_preprocessing
from data_utils import CustomDataset
from multi_threading import CustomThread
from aggregation_functions import FedAvg
from sklearn.cluster import KMeans


# Set cuda
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cuda_num = random.randint(0, (torch.cuda.device_count()-1))
    cuda_name = "cuda:" + str(cuda_num)
    DEVICE = torch.device(cuda_name)


# Drop elements from org_list based on target_list
def drop_elements(org_list, target_list):
    for k in target_list:
        org_list.remove(k)
    return org_list


if __name__ == '__main__':
    print(DEVICE, " are using for training and testing.")
    # --------------------Parameter Setting-----------------------
    # clients hyperparameter setting
    client_epochs = 10
    learning_rate = 0.01
    batch_size = 64
    # Server hyperparameter setting
    num_clients = 30
    rounds = 10
    fraction = 1.0
    # Setting parameters
    neural_network = "MLP_Mult"
    # a list to store global models, 0 index is init global model
    global_models = []
    # a dict to store temp {global models : [temp clients index]}
    global_model_to_clients_recording = {}
    # --------------------Logging setting-----------------------
    curr_path = os.getcwd()
    utils.make_dir(curr_path, "log_file")
    log_name = "log_file/" + "FL" + "_static_clustering_NN_" + neural_network + "_clients_" + str(num_clients) + "_epochs_" + str(client_epochs) + "_rounds_" + str(rounds) + "_fraction_" + str(fraction) + "_date_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
    logging.basicConfig(filename=log_name, format='%(asctime)s - %(message)s', level=logging.INFO)
    # --------------------Data Loading-----------------------
    data_dir = "/home/jliu/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    pickle_dir = "/home/jliu/DoD_Misra_project/jiefei_liu/DOD/MLP_model/data/partition_attacks_2.pkl"
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
    # --------------Build init global model and Select loss function----------------------
    if neural_network == "MLP":
        init_glob_model = models.MLP(input_shape=x_train_un_bin.shape[1]).to(DEVICE)
        loss_fn = nn.BCELoss()  # Binary classification
    elif neural_network == "MLP_Mult":
        init_glob_model = models.MLP_Mult(input_shape=x_train_un_bin.shape[1], num_classes=num_classes).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()  # Muti class classification
    else:
        print("Wrong neural network type, exit.")
        sys.exit()
    optimizer = torch.optim.SGD(init_glob_model.parameters(), lr=learning_rate)
    init_glob_model.train()
    global_models.append(init_glob_model)
    # --------------------initialize record----------------------
    global_weight_record = []
    clients_weight_record = []
    # --------------------Server Training-----------------------
    # Record running time
    server_training_time = []
    start_time = time.time()
    # for loop for FL around
    for iter in range(rounds):
        print("Rounds ", iter, "....")
        Round_time = time.time()
        w_clients = {}
        # models_w = []
        temp_client_list_index = []
        # train each model
        for index in range(num_clients):
            temp_client_list = []
            temp_w_clients = []
            # Get clients data
            (client_X_train, client_y_train) = partition_data_list[index]
            # process data
            train_data = CustomDataset(client_X_train, client_y_train, neural_network)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            # copy corresponding global model
            temp_global_model_index = utils.dict_search(global_model_to_clients_recording, index)
            temp_local_model = copy.deepcopy(global_models[temp_global_model_index])
            # define local optimizer
            local_optimizer = torch.optim.SGD(temp_local_model.parameters(), lr=learning_rate)
            # Set the different local epochs for first round
            if iter == 0:
                temp_local_epoch = client_epochs * 5
            else:
                temp_local_epoch = client_epochs
            # create threads which represents clients
            client = CustomThread(target=utils.train, args=(temp_local_model, local_optimizer, loss_fn, train_loader, temp_local_epoch, neural_network, DEVICE,))
            temp_client_list.append(client)
            temp_client_list_index.append(index)
        # run clients simultaneously
        for client_index in temp_client_list:
            client_index.start()
        # wait clients finish
        for client_index in temp_client_list:
            local_weights, client_index = client_index.join()
            w_clients[client_index] = copy.deepcopy(local_weights)
        # -----------------Find similar clients and aggregate to multiple global models for first round-----------------
        if iter == 0:
            # calculate the weight change of last layer for each client
            clients_last_layer = similarity_utils.weight_changes_of_last_layer(temp_client_list_index, w_clients,
                                                                               global_models,
                                                                               global_model_to_clients_recording,
                                                                               DEVICE)
            # Find the best K for clustering
            utils.find_best_k(clients_last_layer, iter)
            best_k = 5
            # Use Kmeans clustering the clients
            k_means = KMeans(n_clusters=best_k, random_state=0, algorithm="lloyd").fit(clients_last_layer)
            labels = k_means.labels_
            # record the similar clients
            global_model_to_clients_recording = utils.record_clients_clustering(global_model_to_clients_recording,
                                                                                temp_client_list_index, labels, best_k)
            print("Clients distribution: ", global_model_to_clients_recording)
            logging.info('Clients distribution: ', global_model_to_clients_recording)
        # -------------------- Aggregate to global models --------------------
        global_models = aggregation_functions.Multi_model_FedAvg(global_models, global_model_to_clients_recording, w_clients)
        print("Generated ", str(len(global_models) - 1), " Global models")
        # Record model weight updates
        global_weight_record.append(copy.deepcopy(global_models))
        clients_weight_record.append(copy.deepcopy(w_clients))
        # --------------------Server Round Testing-----------------------
        round_loss, round_accuracy, f1, precision, recall = utils.multi_model_test(global_models[1:], loss_fn, test_loader, neural_network, device=DEVICE)
        round_training_time = (time.time() - Round_time) / 60
        server_training_time.append(round_training_time)
        logging.info('Round %d, Loss %f, Accuracy %f, Round Running time(min): %s', iter, round_loss, round_accuracy, round_training_time)
    # --------------------Save Records-----------------------
    # save records
    with open('static_global_weight_records_imbalance.pkl', 'wb') as file:
    # A new file will be created
        pickle.dump(global_weight_record, file)
    with open('static_client_weight_records_imbalance.pkl', 'wb') as file:
    # A new file will be created
        pickle.dump(clients_weight_record, file)
    with open('static_global_to_clients.pkl', 'wb') as file:
    # A new file will be created
        pickle.dump(global_model_to_clients_recording, file)
    # --------------------Server running time-----------------------
    print("---Server running time: %s minutes. ---" % ((time.time() - start_time) / 60))
    logging.info('Total training time(min) %s', sum(server_training_time))
    # --------------------Server Testing-----------------------
    test_time = time.time()
    model_loss, model_accuracy, model_f1, model_precision, model_recall = utils.multi_model_test(
        global_models[1:], loss_fn, test_loader, neural_network, device=DEVICE)
    server_running_time = ((time.time() - test_time) / 60)
    print("Global model, Loss %f, Accuracy %f, F1 %f, Total Running time(min): %s" % (model_loss, model_accuracy, model_f1, server_running_time))
    logging.info('Global model, Loss %f, Accuracy %f, F1 %f, Precision %f, Recall %f, Total Running time(min): %s', model_loss, model_accuracy, model_f1, model_precision, model_recall, server_running_time)
    print("---Server testing time: %s minutes. ---" % server_running_time)
    print("Finish.")
