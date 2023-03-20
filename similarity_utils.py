import io
import os
import time
import sys
import copy
import pickle
import pandas as pd
import numpy as np
import torch
from typing import List
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

import utils
import openpyxl


def weight_changes_cal(global_weight, client_weight):
    # Convert OrderedDict to dict
    global_dict = dict(global_weight)
    client_dict = dict(client_weight)
    global_keys = list(global_dict.keys())
    weight_changes_res = {}
    for key in global_keys:
        temp_g = global_dict[key].numpy()
        temp_c = client_dict[key].numpy()
        temp_changes = np.subtract(temp_c, temp_g)
        weight_changes_res[key] = temp_changes
    return weight_changes_res


def weight_changes_of_last_layer_cal(global_weight, client_weight, device):
    # Convert OrderedDict to dict
    global_dict = dict(global_weight)
    client_dict = dict(client_weight)
    global_keys = list(global_dict.keys())
    local_keys = list(client_dict.keys())
    if global_keys == local_keys:
        if device == "cpu":
            temp_last_layer_weight = np.subtract(client_dict[local_keys[-2]].numpy(), global_dict[global_keys[-2]].numpy())
            temp_last_layer_bias = np.subtract(client_dict[local_keys[-1]].numpy(), global_dict[global_keys[-1]].numpy())
        else:
            temp_last_layer_weight = np.subtract(client_dict[local_keys[-2]].cpu().numpy(),
                                                 global_dict[global_keys[-2]].cpu().numpy())
            temp_last_layer_bias = np.subtract(client_dict[local_keys[-1]].cpu().numpy(),
                                               global_dict[global_keys[-1]].cpu().numpy())
        res = np.concatenate((temp_last_layer_weight, temp_last_layer_bias), axis=None)
    else:
        print("global model and local model are inconsistent.")
        sys.exit()
    return res


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def weight_changes_utils(file_path, similarity_for_all_rounds=True):
    weight_changes = []
    if similarity_for_all_rounds:
        with open('global_weight_records_imbalance.pkl', 'rb') as file:
            # A new file will be created
            global_weight_record = pickle.load(file)
        with open('client_weight_records_imbalance.pkl', 'rb') as file:
            # A new file will be created
            client_weight_record = pickle.load(file)
        # global recoding for different [rounds]
        # print(global_weight_record[0].keys())

        # clients weight recoding for [rounds][client_index]
        # print((client_weight_record[0][0].keys()))

        # Calculate the weight change of each client
        # for loop go over round
        for i in range(len(global_weight_record)):
            temp_clients_weight_change = []
            # for loop go over clients
            for j in range(len(client_weight_record[i])):
                temp_single_client_weight_change = weight_changes_cal(global_weight_record[i], client_weight_record[i][j])
                temp_clients_weight_change.append(temp_single_client_weight_change)
            weight_changes.append(temp_clients_weight_change)
    else:
        global_model_record_path = file_path + "static_init_global_weight_records_imbalance.pkl"
        clients_model_record_path = file_path + "static_first_round_client_weight_records_imbalance.pkl"
        global_client_clustering_res = file_path + "static_first_round_global_to_clients.pkl"
        # read recording from pickle
        with open(global_model_record_path, 'rb') as file:
            # A new file will be created
            global_weight_record = CPU_Unpickler(file).load()
        with open(clients_model_record_path, 'rb') as file:
            # A new file will be created
            client_weight_record = CPU_Unpickler(file).load()
        with open(global_client_clustering_res, 'rb') as file:
            # A new file will be created
            global_model_client_clustering_res = pickle.load(file)
        print("Clustering results from testing: ", global_model_client_clustering_res)
        # print(global_weight_record)
        # print(client_weight_record[0])
        temp_clients_weight_change = []
        for i in range(len(client_weight_record)):
            temp_single_client_weight_change = weight_changes_cal(global_weight_record.state_dict(), client_weight_record[i])
            temp_clients_weight_change.append(temp_single_client_weight_change)
        weight_changes.append(temp_clients_weight_change)
    return weight_changes  # [Rounds][clients]


'''
helper function to calculate the weight change of last layer for each local model
input: trained client index, clients weight, global models, dict mapping between local and global model 
output: nd array (each row represent the last layer + bias of a client)
'''
def weight_changes_of_last_layer(clients_index, clients_weights, global_models, my_dict, device):
    res = []
    # print(type(clients_weights))
    # print(clients_weights.keys())
    # print(type(global_models))
    # sys.exit()
    for index, value in enumerate(clients_index):
        matched_global_model_index = utils.dict_search(my_dict, value, alart=False)
        # print(type(matched_global_model_index))
        try:
            matched_global_model = global_models[matched_global_model_index]
        except:
            print("global model index is not int: ", matched_global_model_index)
            sys.exit()
        client_weight = clients_weights.get(value)
        temp_weight_change_of_last_layer = weight_changes_of_last_layer_cal(matched_global_model.state_dict(), client_weight, device)
        res.append(temp_weight_change_of_last_layer.tolist())
    return np.array(res)


# calculate the similarity of weight changes for all clients
def pairwise_sim(weight_changes):
    sim_res = []
    # for loop rounds
    for i in range(len(weight_changes)):
        temp_round_sim = []
        # for loop go over clients and pairwise compare clients weight change
        for j in range(len(weight_changes[i])):
            temp_client_sim = []
            for k in range(len(weight_changes[i])):
                # calculate the cosine similarity of last layer
                temp_client_sim.append(utils.cosine_similarity_last_layer(weight_changes[i][j], weight_changes[i][k]))
            temp_round_sim.append(temp_client_sim)
        sim_res.append(temp_round_sim)
    return sim_res


def last_layer_extraction_for_clustering(weight_changes_of_clients):
    # print(len(weight_changes_of_clients[0]))
    data = []
    for client_weight in weight_changes_of_clients[0]:
        keys = list(client_weight.keys())
        data.append(np.concatenate((client_weight[keys[-2]], client_weight[keys[-1]]), axis=None))
    return np.array(data)


if __name__ == "__main__":
    # -------------------- Similarity Calculation between rounds --------------------
    file_path = "testing_weight_records/"
    weight_changes = weight_changes_utils(file_path, similarity_for_all_rounds=False)
    # -------------------- Cosine Similarity Calculation --------------------
    # calculate the similarity of last layer
    round_sim = pairwise_sim(weight_changes)
    # Save the results into excel
    # with pd.ExcelWriter("sim_log/" + "Static_testing_ex_imbalance_data_imbalance_last_layer_weight_change_similarity.xlsx") as writer:
    #     for i in range(len(round_sim)):
    #         df = pd.DataFrame(round_sim[i])
    #         sheet_name = 'Round ' + str(i)
    #         df.to_excel(writer, sheet_name=sheet_name)
    # sys.exit()
    # -------------------- Clustering clients --------------------
    # Extract last layer representation for each client
    clients_rep = last_layer_extraction_for_clustering(weight_changes)
    # Find best K
    utils.find_best_k(clients_rep, 0, max_K=30)
    best_k = 22
    # _____________________ Kmeans Clustering ____________________
    # Use Kmeans clustering the clients
    # k_means = KMeans(n_clusters=best_k, random_state=0, algorithm="lloyd").fit(clients_rep)
    # labels = k_means.labels_
    # _____________________ Spectral Clustering ____________________
    sim_matrix = np.array(round_sim[0])
    Spectral = SpectralClustering(n_clusters=best_k, affinity='precomputed').fit(sim_matrix)
    labels = Spectral.labels_
    # # _____________________ Agglomerative Clustering ____________________
    # Agg_clustering = AgglomerativeClustering(n_clusters=best_k).fit(clients_rep)
    # labels = Agg_clustering.labels_
    # record the similar clients
    global_model_to_clients_recording = {}
    temp_client_list_index = range(30)
    global_model_to_clients_recording = utils.record_clients_clustering(global_model_to_clients_recording,
                                                                        temp_client_list_index, labels, best_k)
    print("Clients distribution: ", global_model_to_clients_recording)

