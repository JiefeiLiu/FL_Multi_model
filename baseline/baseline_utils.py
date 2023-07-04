import os
import time
import sys
import copy
import math
import numpy as np
import pandas as pd
import pickle
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from statistics import mean
from numpy.linalg import norm
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import models
import utils
import data_preprocessing
from data_utils import CustomDataset


def fedsem_test(models, loss_fn, test_loader, nn_type, device="cpu"):
    acc_recording = []
    F1_recording = []
    pre_recording = []
    rec_recording = []
    loss_recording = []
    for index, model in enumerate(models):
        model.to(device)
        # test the model
        model.eval()
        y_pred_list = []
        all_true_values = []
        all_pred_values = []
        all_conf_values = []
        loss = 0.0
        print("Test the model")
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                if nn_type == "MLP":
                    '''Attach Binary classification label'''
                    y_pred_tag = torch.round(output)
                    y_pred_tensor = y_pred_tag.flatten()
                    y_pred_np_int = y_pred_tensor.type(torch.float32)
                    pred_values = y_pred_np_int.tolist()
                    all_pred_values.extend(pred_values)
                    loss += loss_fn(output, target.reshape(-1, 1)).item()  # Binary classification
                elif nn_type == "MLP_Mult":
                    '''Attach Muti class classification label'''
                    if device == "cpu":
                        conf, predictions = torch.max(output, 1)
                    else:
                        conf, predictions = torch.max(output.cpu(), 1)
                    all_pred_values.extend(predictions)
                    all_conf_values.extend(conf)
                    loss += loss_fn(output, target).item()  # Muti class classification
                else:
                    print("Wrong neural network type, exit.")
                    sys.exit()

                '''Attach true label'''
                act_values = target.tolist()
                all_true_values.extend(act_values)

        # get performance for each model
        temp_accuracy, temp_f1, temp_precision, temp_recall = utils.get_performance(all_pred_values, all_true_values, nn_type)

        acc_recording.append(temp_accuracy)
        F1_recording.append(temp_f1)
        pre_recording.append(temp_precision)
        rec_recording.append(temp_recall)
        loss_recording.append(loss / len(test_loader))

    return mean(loss_recording), mean(acc_recording), mean(F1_recording), mean(pre_recording), mean(rec_recording)


def weight_preprocess(client_weights, device):
    res = []
    client_dict = dict(client_weights)
    client_value = list(client_dict.values())
    # make n-dim list to one-dim
    for client in client_value:
        temp_client = dict(client)
        temp_client_weight = list(temp_client.values())
        temp_res = []
        for weight in temp_client_weight:
            # temp_weight = [i for l in weight.numpy() for i in l]
            # print(temp_weight)
            # sys.exit()
            if device == "cpu":
                try:
                    temp_weight = [i for l in weight.numpy() for i in l]
                    temp_res.extend(temp_weight)
                except:
                    temp_res.extend(weight)
            else:
                try:
                    temp_weight = [i for l in weight.cpu().numpy() for i in l]
                    temp_res.extend(temp_weight)
                except:
                    temp_res.extend(weight)
        res.append(temp_res)
    return np.array(res)


def single_weight_process(weights, device='"cpu'):
    temp_res = []
    for weight in weights:
        # temp_weight = [i for l in weight.numpy() for i in l]
        # print(temp_weight)
        # sys.exit()
        if device == "cpu":
            try:
                temp_weight = [i for l in weight.numpy() for i in l]
                temp_res.extend(temp_weight)
            except:
                temp_res.extend(weight)
        else:
            try:
                temp_weight = [i for l in weight.cpu().numpy() for i in l]
                temp_res.extend(temp_weight)
            except:
                temp_res.extend(weight)
    return np.array(temp_res)


def euclidean_distance(A, B):
    return np.linalg.norm(A - B)
