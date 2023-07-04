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
from sklearn.cluster import KMeans

sys.path.append("..")
import aggregation_functions
import utils
import baseline.baseline_utils as baseline_utils
import models
import sampling
import data_preprocessing
from data_utils import CustomDataset
from multi_threading import CustomThread


# Set cuda
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cuda_num = random.randint(0, (torch.cuda.device_count() - 1))
    cuda_name = "cuda:" + str(cuda_num)
    DEVICE = torch.device(cuda_name)


if __name__ == '__main__':
    try:
        DEVICE = sys.argv[1]
        print(DEVICE, " is using for training and testing.")
    except:
        print(DEVICE, " is using for training and testing.")

    # --------------------Parameter Setting-----------------------
    # clients hyperparameter setting
    client_epochs = 5
    learning_rate = 0.01
    batch_size = 64
    # Server hyperparameter setting
    num_clients = 20
    training_data_name = str(num_clients) + '_training.pkl'
    rounds = 20
    fraction = 1.0
    # data dir
    dataset = 2017
    if dataset == 2019:
        data_dir = "../2019_data/"
        num_classes = 11
        num_features = 41
    elif dataset == 2017:
        data_dir = "../2017_data/"
        num_classes = 8
        num_features = 40
    else:
        print("No data found, exit.")
        sys.exit()
    # Setting parameters
    neural_network = "MLP_Mult"
    # a list to store global models, 0 index is init global model
    global_models = []
    # a dict to store temp {global models : [temp clients index]}
    global_model_to_clients_recording = {}
    global_model_to_clients_recording_for_aggregation = {}
    global_model_to_clients_sim = {}
    # --------------------Logging setting-----------------------
    curr_path = os.getcwd()
    utils.make_dir(curr_path, "log_file")
    log_name = "log_file/" + "FL" + "_fedsem_dataset_" + str(
        dataset) + "_NN_" + neural_network + "_clients_" + str(
        num_clients) + "_epochs_" + str(client_epochs) + "_rounds_" + str(rounds) + "_fraction_" + str(
        fraction) + "_date_" + datetime.now().strftime(
        "%m_%d_%Y_%H_%M_%S") + ".log"
    logging.basicConfig(filename=log_name, format='%(asctime)s - %(message)s', level=logging.INFO)
    # --------------------Data Loading-----------------------
    print("Loading data...")
    partition_data_list, testing_data, validation_data = utils.load_data(data_dir, training_data=training_data_name)
    (x_test, y_test_bin) = testing_data
    (x_val, y_val) = validation_data
    test_data = CustomDataset(x_test, y_test_bin, neural_network)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # --------------------Logging writing init-----------------------
    logging.info('Parameter setting:')
    logging.info('Number of clients: %d', num_clients)
    logging.info('Number of rounds: %d', rounds)
    logging.info('Client epochs: %d', client_epochs)
    logging.info('Client learning rate : %f', learning_rate)
    logging.info('Client batch size : %d', batch_size)
    logging.info('Fraction : %f', fraction)
    logging.info('Classification method selected : %s', neural_network)
    logging.info('Total number of classes: %d', num_classes)
    logging.info('Loading data path : %s', data_dir)
    logging.info('Experiment results: ')
    # --------------Build init global model and Select loss function----------------------
    if neural_network == "MLP":
        init_glob_model = models.MLP(input_shape=num_features).to(DEVICE)
        loss_fn = nn.BCELoss()  # Binary classification
    elif neural_network == "MLP_Mult":
        init_glob_model = models.MLP_Mult(input_shape=num_features, num_classes=num_classes).to(DEVICE)
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
        temp_client_list = []
        temp_client_list_index = []
        # train each model
        for index in range(num_clients):
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
            # create threads which represents clients
            client = CustomThread(target=utils.train, args=(
                temp_local_model, local_optimizer, loss_fn, train_loader, client_epochs, neural_network, index,
                DEVICE,))
            temp_client_list.append(client)
            temp_client_list_index.append(index)
            # run clients simultaneously
        for client_thread_index in temp_client_list:
            client_thread_index.start()
            # wait clients finish
        for client_thread_index in temp_client_list:
            local_weights, client_index = client_thread_index.join()
            w_clients[client_index] = copy.deepcopy(local_weights)
        # -----------------preprocess weight values-----------------
        processed_weight = baseline_utils.weight_preprocess(w_clients, device=DEVICE)
        # _____________________ Kmeans Clustering ____________________
        K = 4
        k_means = KMeans(n_clusters=4, random_state=42, algorithm="elkan").fit(processed_weight)
        labels = k_means.labels_
        centers = k_means.cluster_centers_
        # print(labels)
        # print(centers)
        # sys.exit()
        # _____________________ matching cluster results to global model recording ____________________
        global_model_to_clients_recording = utils.record_clients_clustering(global_model_to_clients_recording,temp_client_list_index, labels, K)
        global_models = aggregation_functions.Multi_model_FedAvg(global_models, global_model_to_clients_recording,
                                                                 w_clients)
        print("Generated ", str(len(global_models) - 1), " Global models")
        # Record model weight updates
        global_weight_record.append(copy.deepcopy(global_models))
        clients_weight_record.append(copy.deepcopy(w_clients))
        # --------------------Server Round Testing-----------------------
        round_loss, round_accuracy, f1, precision, recall = baseline_utils.fedsem_test(global_models[1:], loss_fn,
                                                                                   test_loader, neural_network,
                                                                                   device=DEVICE)
        round_training_time = (time.time() - Round_time) / 60
        server_training_time.append(round_training_time)
        logging.info('Round %d, Loss %f, Accuracy %f, Round Running time(min): %s', iter, round_loss, round_accuracy,
                     round_training_time)
        print("Round %d, Loss %f, Accuracy %f, Round Running time(min): %s", iter, round_loss, round_accuracy,
         round_training_time)
    # --------------------Server running time-----------------------
    print("---Server running time: %s minutes. ---" % ((time.time() - start_time) / 60))
    logging.info('Total training time(min) %s', sum(server_training_time))
    # --------------------Server Testing-----------------------
    test_time = time.time()
    model_loss, model_accuracy, model_f1, model_precision, model_recall = baseline_utils.fedsem_test(
        global_models[1:], loss_fn, test_loader, neural_network, device=DEVICE)
    server_running_time = ((time.time() - test_time) / 60)
    print("Global model, Loss %f, Accuracy %f, F1 %f, Total Running time(min): %s" % (
        model_loss, model_accuracy, model_f1, server_running_time))
    logging.info('Global model, Loss %f, Accuracy %f, F1 %f, Precision %f, Recall %f, Total Running time(min): %s',
                 model_loss, model_accuracy, model_f1, model_precision, model_recall, server_running_time)
    print("---Server testing time: %s minutes. ---" % server_running_time)
    print("Finish.")










