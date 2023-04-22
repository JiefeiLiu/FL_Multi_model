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
from sklearn.cluster import SpectralClustering

# Set cuda
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cuda_num = random.randint(0, (torch.cuda.device_count() - 1))
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
    rounds = 5
    fraction = 1.0
    conf_filter = 0.7
    percentage_of_noise = 0.4
    # Setting parameters
    neural_network = "MLP_Mult"
    # a list to store global models, 0 index is init global model
    global_models = []
    # a dict to store temp {global models : [temp clients index]}
    over_lapping_clients_selection = False
    global_model_to_clients_recording = {}
    global_model_to_clients_recording_for_aggregation = {}
    global_model_to_clients_sim = {}
    # --------------------Logging setting-----------------------
    curr_path = os.getcwd()
    utils.make_dir(curr_path, "log_file")
    log_name = "log_file/" + "FL" + "_static_clustering_NN_" + neural_network + "_clients_" + str(
        num_clients) + "_epochs_" + str(client_epochs) + "_rounds_" + str(rounds) + "_fraction_" + str(
        fraction) + "_date_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
    logging.basicConfig(filename=log_name, format='%(asctime)s - %(message)s', level=logging.INFO)
    # --------------------Data Loading-----------------------
    # data_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    # pickle_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/MLP_model/data/partition_attacks_2.pkl"
    data_dir = "/home/jliu/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    pickle_dir = "/home/jliu/DoD_Misra_project/jiefei_liu/DOD/MLP_model/data/partition.pkl"
    num_classes = 12
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
        temp_client_list = []
        temp_client_list_index = []
        # train each model
        for index in range(num_clients):
            temp_w_clients = []
            # Get clients data
            (client_X_train, client_y_train) = partition_data_list[index]
            x_train_new, y_train_new = data_preprocessing.noise_generator(x_train_un_bin, y_train_un_bin, client_X_train,
                                                                          client_y_train, percentage_noise=percentage_of_noise)
            # process data
            train_data = CustomDataset(x_train_new, y_train_new, neural_network)
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
            client = CustomThread(target=utils.train, args=(
            temp_local_model, local_optimizer, loss_fn, train_loader, temp_local_epoch, neural_network, index, DEVICE,))
            temp_client_list.append(client)
            temp_client_list_index.append(index)
        # run clients simultaneously
        for client_thread_index in temp_client_list:
            client_thread_index.start()
        # wait clients finish
        for client_thread_index in temp_client_list:
            local_weights, client_index = client_thread_index.join()
            w_clients[client_index] = copy.deepcopy(local_weights)
        # -----------------Find similar clients and aggregate to multiple global models for first round-----------------
        if iter == 0:
            # calculate the weight change of last layer for each client
            clients_last_layer = similarity_utils.weight_changes_of_last_layer(temp_client_list_index, w_clients,
                                                                               global_models,
                                                                               global_model_to_clients_recording,
                                                                               DEVICE)
            # Calculate weight similarity matrix
            similarity_matrix = utils.cosine_similarity_matrix(clients_last_layer)
            # _____________________ Find the best K for clustering _____________________
            # utils.find_best_k(clients_last_layer, iter)
            # best_k = 23
            # _____________________ Kmeans Clustering ____________________
            # k_means = KMeans(n_clusters=best_k, random_state=0, algorithm="lloyd").fit(clients_last_layer)
            # labels = k_means.labels_
            # _____________________ Spectral Clustering ____________________
            # compute the similarity matrix of clients last layer
            # Spectral = SpectralClustering(n_clusters=best_k, affinity='precomputed').fit(similarity_matrix)
            # labels = Spectral.labels_
            # _____________________ matching cluster results to global model recording ____________________
            # global_model_to_clients_recording = utils.record_clients_clustering(global_model_to_clients_recording,
            #                                                                     temp_client_list_index, labels, best_k)
            # ___________________ Manually Select the model with high similarity without overlapping ___________________
            # global_model_to_clients_recording = {
            #     1: [0, 11],
            #     2: [1, 4],
            #     3: [2],
            #     4: [3, 26],
            #     5: [5],
            #     6: [6, 25],
            #     7: [7, 17],
            #     8: [8],
            #     9: [9],
            #     10: [10],
            #     11: [12],
            #     12: [13],
            #     13: [14],
            #     14: [15],
            #     15: [16, 28],
            #     16: [18],
            #     17: [19],
            #     18: [20],
            #     19: [21, 23],
            #     20: [22],
            #     21: [24],
            #     22: [27],
            #     23: [29],
            # }
            # _____________________ Manually Select the model with high similarity with overlapping ____________________
            # global_model_to_clients_recording_for_aggregation = {
            #     1: [0, 11],
            #     2: [1, 2, 7, 13, 17, 23, 29],
            #     3: [3, 4, 9, 15, 18, 19, 25, 28],
            #     4: [4, 9, 25, 3, 15, 18, 19, 25, 28],
            #     5: [5],
            #     6: [6, 24, 26],
            #     7: [7, 1, 2, 13, 17, 23, 29],
            #     8: [10, 22],
            #     9: [12],
            #     10: [13, 23, 1, 2, 7, 16, 17, 29],
            #     11: [14],
            #     12: [15, 18, 3, 4, 9, 17, 19, 25, 28],
            #     13: [16, 13],
            #     14: [17, 1, 2, 7, 13, 15, 18, 23, 29],
            #     15: [19, 3, 4, 9, 15, 18, 25, 28],
            #     16: [20],
            #     17: [21],
            #     18: [24, 6, 26],
            #     19: [26, 6, 24],
            #     20: [27, 16],
            #     21: [28, 3, 4, 9, 15, 18, 19, 25],
            #     22: [29, 1, 2, 7, 13, 17, 23],
            #     23: [8],
            # }
            # _____________________ Group the clients from script _____________________
            global_model_to_clients_recording = utils.group_clients_from_sim_matrix(similarity_matrix, temp_client_list_index)
            if over_lapping_clients_selection:
                global_model_to_clients_recording_for_aggregation, global_model_to_clients_sim = utils.overlapping_group_clients_from_sim_matrix(similarity_matrix, temp_client_list_index)
                print("Overlapping clients distribution", global_model_to_clients_recording_for_aggregation)
                print("Overlapping clients similarity: ", global_model_to_clients_sim)
            print("Clients distribution: ", global_model_to_clients_recording)
            logging.info('Clients distribution: %s', global_model_to_clients_recording)
            # logging.info('Clients similarity: %s', global_model_to_clients_sim)
        # --------------------Save Temp Records-----------------------
        # save records
        curr_path = os.getcwd()
        utils.make_dir(curr_path, "testing_weight_records")
        records_saving_path = curr_path + "/testing_weight_records/"
        with open(records_saving_path + 'static_init_global_weight_records_imbalance.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(init_glob_model, file)
        with open(records_saving_path + 'static_first_round_client_weight_records_imbalance.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(w_clients, file)
        with open(records_saving_path + 'static_first_round_global_to_clients.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(global_model_to_clients_recording, file)
        # sys.exit()
        # -------------------- Aggregate to global models --------------------
        if over_lapping_clients_selection:
            global_models = aggregation_functions.Multi_model_FedAvg_with_attention(global_models,
                                                                     global_model_to_clients_recording_for_aggregation,
                                                                     global_model_to_clients_sim, w_clients)
        else:
            global_models = aggregation_functions.Multi_model_FedAvg(global_models, global_model_to_clients_recording,
                                                                     w_clients)
        print("Generated ", str(len(global_models) - 1), " Global models")
        # Record model weight updates
        global_weight_record.append(copy.deepcopy(global_models))
        clients_weight_record.append(copy.deepcopy(w_clients))
        # --------------------Server Round Testing-----------------------
        round_loss, round_accuracy, f1, precision, recall = utils.multi_model_test(global_models[1:], loss_fn,
                                                                                   test_loader, neural_network,
                                                                                   device=DEVICE, conf_rule=conf_filter)
        round_training_time = (time.time() - Round_time) / 60
        server_training_time.append(round_training_time)
        logging.info('Round %d, Loss %f, Accuracy %f, Round Running time(min): %s', iter, round_loss, round_accuracy,
                     round_training_time)
    # --------------------Save All Records-----------------------
    # save records
    curr_path = os.getcwd()
    utils.make_dir(curr_path, "weight_records")
    records_saving_path = curr_path + "/weight_records/"
    with open(records_saving_path + 'static_global_weight_records_imbalance.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(global_weight_record, file)
    with open(records_saving_path + 'static_client_weight_records_imbalance.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(clients_weight_record, file)
    with open(records_saving_path + 'static_global_to_clients.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(global_model_to_clients_recording, file)
    # save result global models
    for i in range(len(global_models)):
        temp_saving_name = records_saving_path + "static_global_model_" + str(i) + ".pt"
        torch.save(global_models[i], temp_saving_name)
    # --------------------Server running time-----------------------
    print("---Server running time: %s minutes. ---" % ((time.time() - start_time) / 60))
    logging.info('Total training time(min) %s', sum(server_training_time))
    # --------------------Server Testing-----------------------
    test_time = time.time()
    model_loss, model_accuracy, model_f1, model_precision, model_recall = utils.multi_model_test(
        global_models[1:], loss_fn, test_loader, neural_network, device=DEVICE, conf_rule=conf_filter)
    server_running_time = ((time.time() - test_time) / 60)
    print("Global model, Loss %f, Accuracy %f, F1 %f, Total Running time(min): %s" % (
    model_loss, model_accuracy, model_f1, server_running_time))
    logging.info('Global model, Loss %f, Accuracy %f, F1 %f, Precision %f, Recall %f, Total Running time(min): %s',
                 model_loss, model_accuracy, model_f1, model_precision, model_recall, server_running_time)
    print("---Server testing time: %s minutes. ---" % server_running_time)
    print("Finish.")
