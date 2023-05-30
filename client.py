import random
import warnings
import sys
import os
import flwr as fl
import pickle
import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

import utils
import models
from data_utils import CustomDataset


USE_FEDBN: bool = False
# Set cuda
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cuda_num = random.randint(0, (torch.cuda.device_count()-1))
    cuda_name = "cuda:" + str(cuda_num)
    DEVICE = torch.device(cuda_name)


class MLPClient(fl.client.NumPyClient):
    def __init__(
            self,
            model: models,
            trainloader: DataLoader,
            testloader: DataLoader,
            optimizer,
            loss_fn,
            num_examples: Dict,
            epochs,
            neural_network,
            client_index,

    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.neural_network = neural_network
        self.client_index = client_index

    def get_parameters(self, config) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
            self, parameters: List[np.ndarray], config
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        utils.train(self.model, self.optimizer, self.loss_fn, self.trainloader, self.epochs, self.neural_network, self.client_index, device=DEVICE)
        print(f"Training finished for round {config['server_round']}")
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
            self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy, f1, precision, recall = utils.test(self.model, self.loss_fn, self.testloader, self.neural_network, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main() -> None:
    # Select from 2017, 2019 or generated
    dataset = 2019
    if dataset == 2019:
        data_dir = "2019_data/"
    elif dataset == 2017:
        data_dir = "2017_data/"
    elif dataset == "generated":
        data_dir = 'generated_data/'
    else:
        print("No data found.")
        sys.exit()

    print(DEVICE, " are using for training and testing.")
    # hyper-parameters
    client_epochs = 10
    learning_rate = 0.01
    batch_size = 64
    partition_num = 30
    num_classes = 11
    # Select from "Binary" or "Multi"
    classification = "Multi"
    neural_network = "MLP_Mult"
    # ------------------ Create saving model folder --------------
    # curr_path = os.getcwd()
    # utils.make_dir(curr_path, "models")
    # utils.make_dir(curr_path, "results")
    # -------------- load datasets ----------------------
    print("Loading data...")
    if dataset == 2017:
        partitioned_data, testing_data, _ = utils.load_data(data_dir)
        (x_test, y_test_bin) = testing_data
    elif dataset == 2019:
        partitioned_data, testing_data, _ = utils.load_data(data_dir)
        (x_test, y_test_bin) = testing_data
    elif dataset == "generated":
        partitioned_data, testing_data, _ = utils.load_data(data_dir)
        (x_test, y_test_bin) = testing_data
    else:
        print("Dataset does not found")
        sys.exit()
    # -------------- Load partition datasets -----------------
    try:
        partition_id = int(sys.argv[1])
        print("Client ", partition_id)
    except:
        partition_id = np.random.choice(partition_num)
        print("Random Client ", partition_id)
    (X_train, y_train) = partitioned_data[partition_id]
    # --------------Print Shape-----------------
    num_examples = {"trainset": len(y_train), "testset": len(y_test_bin)}
    print(num_examples)
    # -------------- Preprocess datasets -----------------
    train_data = CustomDataset(X_train, y_train, neural_network)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = CustomDataset(x_test, y_test_bin, neural_network)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # -------------- Define model ----------------------
    if classification == "Binary":
        model = models.MLP(input_shape=X_train.shape[1]).to(DEVICE).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.BCELoss()  # Binary classification
    elif classification == "Multi":
        model = models.MLP_Mult(input_shape=X_train.shape[1], num_classes=num_classes).to(DEVICE).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()  # Muti class classification
    else:
        print("Classification method does not found.")
        sys.exit()

    # -------------- Start client --------------
    client = MLPClient(model, train_loader, test_loader, optimizer, loss_fn, num_examples, client_epochs, neural_network, partition_id)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        # server_address="128.123.63.250:3000",
        client=client
    )


if __name__ == "__main__":
    main()