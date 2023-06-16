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

# control random selection
np.random.seed(42)

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
    try:
        DEVICE = sys.argv[1]
        print(DEVICE, " is using for training and testing.")
    except:
        print(DEVICE, " is using for training and testing.")
    # --------------------Parameter Setting-----------------------
    # clients hyperparameter setting
    client_epochs = 10
    learning_rate = 0.01
    batch_size = 64
    # Server hyperparameter setting
    num_clients = 25
    training_data_name = str(num_clients) + '_training.pkl'
    rounds = 20
    fraction = 1.0
    num_global_models = 5
    num_clients_per_model = 9
    # Setting parameters
    neural_network = "MLP_Mult"
    # data dir
    dataset = 2017
    if dataset == 2019:
        data_dir = "2019_data/"
        num_classes = 11
        num_features = 41
    elif dataset == 2017:
        data_dir = "2017_data/"
        num_classes = 7
        num_features = 40
    else:
        print("No data found, exit.")
        sys.exit()
    # --------------------Logging setting-----------------------
    curr_path = os.getcwd()
    utils.make_dir(curr_path, "log_file")
    log_name = "log_file/" + "FL" + "_dynamic_random_dataset_" + str(dataset) + "_NN_" + neural_network + "_clients_" + str(num_clients) + "_epochs_" + str(client_epochs) + "_rounds_" + str(rounds) + "_fraction_" + str(fraction) + "_date_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
    logging.basicConfig(filename=log_name, format='%(asctime)s - %(message)s', level=logging.INFO)
    # --------------------Data Loading-----------------------
    # data_dir = "/home/jliu/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    # data_dir = "/home/jliu/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    # pickle_dir = "/home/jliu/DoD_Misra_project/jiefei_liu/DOD/MLP_model/data/partition.pkl"
    print("Loading data...")
    # (x_train_un_bin, y_train_un_bin), (x_test, y_test_bin), (_, _) = data_preprocessing.read_2019_data(data_dir)
    # partition_data_list = sampling.partition_bal_equ(x_train_un_bin, y_train_un_bin, num_clients)
    # Load partitioned data
    # with open(pickle_dir, 'rb') as file:
    #     # Call load method to deserialze
    #     partition_data_list = pickle.load(file)
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
    logging.info('Client learning rate : %d', learning_rate)
    logging.info('Client batch size : %d', batch_size)
    logging.info('Fraction : %d', fraction)
    logging.info('Classification method selected : %s', neural_network)
    logging.info('Total number of classes: %d', num_classes)
    logging.info('Loading data path : %s', data_dir)
    logging.info('Experiment results: ')
    # --------------Build global models and Select loss function----------------------
    glob_models = []
    loss_functions = []
    optimizers = []
    for i in range(num_global_models):
        if neural_network == "MLP":
            glob_model = models.MLP(input_shape=num_features).to(DEVICE)
            optimizer = torch.optim.SGD(glob_model.parameters(), lr=learning_rate)
            loss_fn = nn.BCELoss()  # Binary classification
            glob_models.append(glob_model)
            loss_functions.append(loss_fn)
            optimizers.append(optimizer)

        elif neural_network == "MLP_Mult":
            glob_model = models.MLP_Mult(input_shape=num_features, num_classes=num_classes).to(DEVICE)
            optimizer = torch.optim.SGD(glob_model.parameters(), lr=learning_rate)
            loss_fn = nn.CrossEntropyLoss()  # Muti class classification
            glob_models.append(glob_model)
            loss_functions.append(loss_fn)
            optimizers.append(optimizer)
        else:
            print("Wrong neural network type, exit.")
            sys.exit()
    # --------------------initialize weight----------------------
    w_globals = []
    for i in range(num_global_models):
        glob_models[i].train()
        w_glob = glob_models[i].state_dict()
        w_globals.append(w_glob)
    # --------------------Random assign clients for each model, without overlapping-----------------------
    # model_clients = []
    # clients_list = list(range(0, num_clients))
    # num_clients_per_model = int(num_clients / num_global_models)
    # # print(clients_list)
    # for i in range(num_global_models):
    #     temp_clients_list = np.random.choice(clients_list, num_clients_per_model, replace=False)
    #     # print(temp_clients_list)
    #     model_clients.append(temp_clients_list)
    #     clients_list = drop_elements(clients_list, temp_clients_list)
    # --------------------Server Training-----------------------
    # Record running time
    server_training_time = []
    start_time = time.time()
    # for loop for FL around
    for iter in range(rounds):
        print("Rounds ", iter, "....")
        Round_time = time.time()
        models_w = []
        # train each model
        for model_index, single_model in enumerate(glob_models):
            temp_client_list = []
            temp_w_clients = []
            '''
            Dynamic select clients for each model per round 
            '''
            single_model_clients = np.random.choice(range(num_clients), num_clients_per_model, replace=False)
            for client_index in single_model_clients:
                # Get clients data
                (client_X_train, client_y_train) = partition_data_list[client_index]
                # process data
                train_data = CustomDataset(client_X_train, client_y_train, neural_network)
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                # copy corresponding global model
                temp_local_model = copy.deepcopy(glob_models[model_index])
                # define local optimizer
                local_optimizer = torch.optim.SGD(temp_local_model.parameters(), lr=learning_rate)
                # create threads which represents clients
                client = CustomThread(target=utils.train, args=(temp_local_model, local_optimizer, loss_functions[model_index], train_loader, client_epochs, neural_network, client_index, DEVICE,))
                temp_client_list.append(client)
            # run clients simultaneously
            for client_index in temp_client_list:
                client_index.start()
            # wait clients finish
            for client_index in temp_client_list:
                local_weights, client_index = client_index.join()
                temp_w_clients.append(copy.deepcopy(local_weights))
            # collect model weights
            models_w.append(temp_w_clients)

        # Global model weight updates
        w_globals_last = copy.deepcopy(w_globals)
        # FedAvg for each model
        for j in range(num_global_models):
            w_glob = FedAvg(models_w[j], DEVICE)
            # Update global models
            glob_models[j].load_state_dict(w_glob)
        # --------------------Server Round Testing-----------------------
        round_models_loss = []
        round_models_accuracy = []
        # Testing all models
        for j in range(num_global_models):
            temp_model_round_loss, temp_model_round_accuracy, _, _, _ = utils.test(glob_models[j], loss_functions[j], test_loader, neural_network, device=DEVICE)
            round_models_loss.append(temp_model_round_loss)
            round_models_accuracy.append(temp_model_round_accuracy)
        # find best model
        best_accuracy_index = round_models_accuracy.index(max(round_models_accuracy))
        round_training_time = (time.time() - Round_time) / 60
        server_training_time.append(round_training_time)
        logging.info('Round %d, Loss %f, Accuracy %f, Round Running time(min): %s', iter, round_models_loss[best_accuracy_index], round_models_accuracy[best_accuracy_index], round_training_time)
    print("---Server running time: %s minutes. ---" % ((time.time() - start_time) / 60))
    logging.info('Total training time(min) %s', sum(server_training_time))
    # --------------------Server Testing-----------------------
    test_time = time.time()
    # server_models_loss = []
    # server_models_accuracy = []
    # server_models_f1 = []
    # server_models_precision = []
    # server_models_recall = []
    # Testing all models
    # for j in range(num_global_models):
    #     temp_model_loss, temp_model_accuracy, temp_model_f1, temp_model_precision, temp_model_recall = utils.test(glob_models[j], loss_functions[j], test_loader, neural_network, device=DEVICE)
    #     server_models_loss.append(temp_model_loss)
    #     server_models_accuracy.append(temp_model_accuracy)
    #     server_models_f1.append(temp_model_f1)
    #     server_models_precision.append(temp_model_precision)
    #     server_models_recall.append(temp_model_recall)
    # # find best model
    # best_model_index = server_models_f1.index(max(server_models_f1))
    model_loss, model_accuracy, model_f1, model_precision, model_recall = utils.multi_model_test(
        glob_models, loss_functions[0], test_loader, neural_network, device=DEVICE)
    server_running_time = ((time.time() - test_time) / 60)
    print("Global model, Loss %f, Accuracy %f, F1 %f, Total Running time(min): %s" % (model_loss, model_accuracy, model_f1, server_running_time))
    logging.info('Global model, Loss %f, Accuracy %f, F1 %f, Precision %f, Recall %f, Total Running time(min): %s', model_loss, model_accuracy, model_f1, model_precision, model_recall, server_running_time)
    print("---Server testing time: %s minutes. ---" % server_running_time)
    print("Finish.")
