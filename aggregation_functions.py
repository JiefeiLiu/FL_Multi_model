import copy
import sys

import torch


def FedAvg(w, device):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def Multi_model_FedAvg(global_model_list, clustered_info, clients_weights, device='cpu'):
    del global_model_list[1:len(global_model_list)]
    init_global_model = copy.deepcopy(global_model_list[0])
    clustered_info_value = clustered_info.values()
    # for loop go through clusters
    for index, clients_list in enumerate(clustered_info_value):
        temp_clients_weights = []
        # for loop go through each client
        for client_index in clients_list:
            temp_clients_weights.append(clients_weights.get(client_index))
        temp_global_model_weight = FedAvg(temp_clients_weights, device)
        temp_init_global_model = copy.deepcopy(init_global_model)
        temp_init_global_model.load_state_dict(temp_global_model_weight)
        global_model_list.append(temp_init_global_model)
    return global_model_list


# FedAvg with attention
def FedAvgAtt(w, sim, device):
    w_avg = copy.deepcopy(w[0])
    for index, k in enumerate(w_avg.keys()):
        for i in range(1, len(w)):
            w_avg[k] += torch.mul(w[i][k], sim[i-1])
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def Multi_model_FedAvg_with_attention(global_model_list, clustered_info, sim_info, clients_weights, device='cpu'):
    del global_model_list[1:len(global_model_list)]
    init_global_model = copy.deepcopy(global_model_list[0])
    clustered_info_value = clustered_info.values()
    sim_info_value = list(sim_info.values())
    # for loop go through clusters
    for index, clients_list in enumerate(clustered_info_value):
        temp_clients_weights = []
        temp_sim_value = sim_info_value[index]
        # for loop go through each client
        for list_index, client_index in enumerate(clients_list):
            temp_clients_weights.append(clients_weights.get(client_index))
        temp_global_model_weight = FedAvgAtt(temp_clients_weights, temp_sim_value, device)
        temp_init_global_model = copy.deepcopy(init_global_model)
        temp_init_global_model.load_state_dict(temp_global_model_weight)
        global_model_list.append(temp_init_global_model)
    return global_model_list


if __name__ == '__main__':
    mydict = {1: [1, 2], 2: [2, 5], 3: [3, 5]}
    print(list(mydict.values()))