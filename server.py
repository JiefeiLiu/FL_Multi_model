import os
import sys
import time
import random
import flwr as fl
import numpy as np
from typing import Dict
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

import models
import client
import utils
from data_utils import CustomDataset

# Set cuda
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cuda_num = random.randint(0, (torch.cuda.device_count()-1))
    cuda_name = "cuda:" + str(cuda_num)
    DEVICE = torch.device(cuda_name)


# define parameters
# Select from 2017 or 2019
dataset = 2017
# Select from "Binary" or "Multi"
classification = "Multi"
neural_network = "MLP_Mult"
# number of classes
if dataset == 2019:
    n_classes = 11
    num_features = 41
elif dataset == 2017:
    n_classes = 7
    num_features = 40
# number of features
partition_num = 20

if dataset == 2019:
    data_dir = "2019_data/"
elif dataset == 2017:
    data_dir = "2017_data/"
elif dataset == "generated":
    data_dir = 'generated_data/'


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""
    # load data
    batch_size = 64
    training_data_name = str(partition_num) + '_training.pkl'
    if dataset == 2017:
        _, testing_data, _ = utils.load_data(data_dir, training_data_name)
        (x_test, y_test_bin) = testing_data
    elif dataset == 2019:
        _, testing_data, _ = utils.load_data(data_dir, training_data_name)
        (x_test, y_test_bin) = testing_data
    elif dataset == "generated":
        _, testing_data, _ = utils.load_data(data_dir, training_data_name)
        (x_test, y_test_bin) = testing_data
    else:
        print("Dataset does not found")
        sys.exit()
    test_data = CustomDataset(x_test, y_test_bin, neural_network)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    '''Select the binary or muti class classification'''
    if classification == "Binary":
        model = models.MLP(input_shape=num_features)
        loss_fn = nn.BCELoss()  # Binary classification
    elif classification == "Multi":
        model = models.MLP_Mult(input_shape=num_features, num_classes=n_classes)
        loss_fn = nn.CrossEntropyLoss()  # Muti class classification
    else:
        print("Classification method does not found.")
        sys.exit()
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        print(DEVICE, " is using for testing...")
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy, f1, precision, recall = utils.test(model, loss_fn, test_loader, neural_network, device=DEVICE)
        return loss, {"accuracy": accuracy}

    return evaluate


def get_parameters(model) -> List[np.ndarray]:
    model.train()
    # Return model parameters as a list of NumPy ndarrays
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def server_init():
    '''Init global model'''
    # load data
    batch_size = 64
    training_data_name = str(partition_num) + '_training.pkl'
    if dataset == 2017:
        _, testing_data, _ = utils.load_data(data_dir, training_data_name)
        (x_test, y_test_bin) = testing_data
    elif dataset == 2019:
        _, testing_data, _ = utils.load_data(data_dir, training_data_name)
        (x_test, y_test_bin) = testing_data
    elif dataset == "generated":
        _, testing_data, _ = utils.load_data(data_dir, training_data_name)
        (x_test, y_test_bin) = testing_data
    else:
        print("Dataset does not found")
        sys.exit()
    test_data = CustomDataset(x_test, y_test_bin, neural_network)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    '''Select the binary or muti class classification'''
    if classification == "Binary":
        model = models.MLP(input_shape=num_features)
    elif classification == "Multi":
        model = models.MLP_Mult(input_shape=num_features, num_classes=n_classes)
    else:
        print("Classification method does not found.")
        sys.exit()
    return model


if __name__ == "__main__":
    start_time = time.time()
    strategy = fl.server.strategy.FedAvgM(
        min_available_clients=partition_num,
        # min_fit_clients=30,
        # min_evaluate_clients=30,
        fraction_fit=0.9,
        fraction_evaluate=0.9,
        # server_learning_rate=0.1,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(server_init())),
        server_momentum=0.7,
        # eta=0.01,
        # beta_1=0.7,
        # tau=0.001,
        on_fit_config_fn=fit_round,
        evaluate_fn=get_evaluate_fn(),
    )
    fl.server.start_server(
        # server_address="128.123.63.250:3000",
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=20),
    )
    print("---Running time: %s minutes. ---" % ((time.time() - start_time) / 60))
