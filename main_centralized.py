import models
import data_preprocessing
from data_utils import CustomDataset
import os
import time
import sys
import copy
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from numpy.linalg import norm
from sklearn import metrics
import utils


if __name__ == '__main__':
    # data_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/LR_model/CICIDS2017/"
    data_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    # hyper-parameters
    epochs = 50
    learning_rate = 0.01
    batch_size = 64
    # Setting parameters
    neural_network = "MLP_Mult"
    # -------------------load datasets----------------------
    (x_train_un_bin, y_train_un_bin), (x_test, y_test_bin) = data_preprocessing.preprocess_data_with_random_drop_class(data_dir, 8)
    num_examples = {"trainset": len(y_train_un_bin), "testset": len(y_test_bin)}
    print(num_examples)

    # check to use GPU or not
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda:0" if use_cuda else "cpu")
    print(DEVICE, "are using for training and testing.")

    # -------------------Set model----------------------
    # Model, Optimizer, Loss func
    '''Select the different loss function for binary or muti class classification'''
    if neural_network == "MLP":
        model = models.MLP(input_shape=x_train_un_bin.shape[1]).to(DEVICE)
        loss_fn = nn.BCELoss()  # Binary classification
    elif neural_network == "MLP_Mult":
        model = models.MLP_Mult(input_shape=x_train_un_bin.shape[1], num_classes=11).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()  # Muti class classification
    else:
        print("Wrong neural network type, exit.")
        sys.exit()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # -------------------Training model----------------------
    train_time = time.time()
    train_data = CustomDataset(x_train_un_bin, y_train_un_bin, neural_network)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model_weights = utils.train(model, optimizer, loss_fn, train_loader, epochs, neural_network, device=DEVICE)
    print("---Training time: %s minutes. ---" % ((time.time() - train_time) / 60))
    # save model
    # saving_model_name = "models/model_client_" + str(single_client_index) + ".pth"
    # torch.save(copy.deepcopy(model_weights), saving_model_name)
    # -------------------Testing model----------------------
    test_time = time.time()
    test_data = CustomDataset(x_test, y_test_bin, neural_network)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.load_state_dict(copy.deepcopy(model_weights))
    loss, accuracy, f1, precision, recall = utils.test(model, loss_fn, test_loader, neural_network, device=DEVICE)
    print("---Testing time: %s minutes. ---" % ((time.time() - test_time) / 60))
