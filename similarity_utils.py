import os
import time
import sys
import copy
import pickle
import pandas as pd
import numpy as np
import torch
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


def weight_changes_of_last_layer_cal(global_weight, client_weight):
    # Convert OrderedDict to dict
    global_dict = dict(global_weight)
    client_dict = dict(client_weight)
    global_keys = list(global_dict.keys())
    local_keys = list(client_dict.keys())
    if global_keys == local_keys:
        temp_last_layer_weight = np.subtract(client_dict[local_keys[-2]].numpy(), global_dict[global_keys[-2]].numpy())
        temp_last_layer_bias = np.subtract(client_dict[local_keys[-1]].numpy(), global_dict[global_keys[-1]].numpy())
        res = np.concatenate((temp_last_layer_weight, temp_last_layer_bias), axis=None)
    else:
        print("global model and local model are inconsistent.")
        sys.exit()
    return res



def weight_changes_utils():
    # read recording from pickle
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
    weight_changes = []
    # for loop go over round
    for i in range(len(global_weight_record)):
        temp_clients_weight_change = []
        # for loop go over clients
        for j in range(len(client_weight_record[i])):
            temp_single_client_weight_change = weight_changes_cal(global_weight_record[i], client_weight_record[i][j])
            temp_clients_weight_change.append(temp_single_client_weight_change)
        weight_changes.append(temp_clients_weight_change)
    return weight_changes  # [Rounds][clients]


'''
helper function to calculate the weight change of last layer for each local model
input: trained client index, clients weight, global models, dict mapping between local and global model 
output: nd array (each row represent the last layer + bias of a client)
'''
def weight_changes_of_last_layer(clients_index, clients_weights, global_models, my_dict):
    res = []
    for index, value in enumerate(clients_index):
        matched_global_model_index = utils.dict_search(my_dict, value, alart=False)
        # print(type(matched_global_model_index))
        try:
            matched_global_model = global_models[matched_global_model_index]
        except:
            print("global model index is not int: ", matched_global_model_index)
            sys.exit()
        client_weight = clients_weights.get(value)
        temp_weight_change_of_last_layer = weight_changes_of_last_layer_cal(matched_global_model.state_dict(), client_weight)
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


if __name__ == "__main__":
    weight_changes = weight_changes_utils()
    round_sim = pairwise_sim(weight_changes)
    # print(len(round_sim[0][0]))
    # Save the results into excel
    with pd.ExcelWriter("sim_log/" + "ex_imbalance_data_imbalance_last_layer_weight_change_similarity.xlsx") as writer:
        for i in range(len(round_sim)):
            df = pd.DataFrame(round_sim[i])
            sheet_name = 'Round ' + str(i)
            df.to_excel(writer, sheet_name=sheet_name)
