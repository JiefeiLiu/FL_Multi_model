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
from numpy.linalg import norm
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import models
import data_preprocessing
from data_utils import CustomDataset


# read dataset
def load_data(file_path, training_data='30_training.pkl'):
    # load training testing and validation
    with open(file_path + training_data, 'rb') as file:
        # Call load method to deserialze
        training_data_list = pickle.load(file)
    with open(file_path + 'testing.pkl', 'rb') as file:
        # Call load method to deserialze
        testing_data = pickle.load(file)
    with open(file_path + 'validation.pkl', 'rb') as file:
        # Call load method to deserialze
        validation_data = pickle.load(file)
    return training_data_list, testing_data, validation_data

'''
Training the model
    input parameter: 
        model: copy of the global model 
        optimizer: optimizer for specific client
        loss_fn: loss function 
        train_loader: processed data with Dataloader
        epochs: local client epochs
        device: run by device
    output: 
        the local model weight 
'''


def train(model, optimizer, loss_fn, train_loader, epochs, nn_type, client_index, device="cpu"):
    default_device = device
    print("Train the Client model")
    # if torch.cuda.is_available():
    #     cuda_num = random.randint(0, (torch.cuda.device_count() - 1))
    #     cuda_name = "cuda:" + str(cuda_num)
    #     device = torch.device(cuda_name)
    #     print(device, " is using for training.")
    model.train()
    model.to(device)
    losses = []
    for e in range(epochs):
        epoch_loss = 0
        # epoch_acc = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # calculate output
            output = model(data)
            '''select and calculate loss for binary or muti class classification'''
            if nn_type == "MLP":
                loss = loss_fn(output, target.reshape(-1, 1))  # Binary classification
            elif nn_type == "MLP_Mult":
                loss = loss_fn(output, target)  # Muti class classification
            else:
                print("Wrong neural network type, exit.")
                sys.exit()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        if e % 1 == 0:
            print(f'Epoch {(e + 1) + 0:02}: | Loss: {epoch_loss / len(train_loader):.5f}')
    # convert back to default device
    # model.to(default_device)
    return model.state_dict(), client_index


'''
Testing the model
    input parameter: 
        model: copy of the model 
        loss_fn: loss function 
        test_loader: processed data with Dataloader
        epochs: local client epochs
        device: run by device
    output: 
        the local model weight 
'''


def test(model, loss_fn, test_loader, nn_type, device="cpu"):
    # model.load_state_dict(torch.load(model_path))
    model.to(device)
    # test the model
    model.eval()
    y_pred_list = []
    loss = 0.0
    all_true_values = []
    all_pred_values = []
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
                    _, predictions = torch.max(output, 1)
                else:
                    _, predictions = torch.max(output.cpu(), 1)
                all_pred_values.extend(predictions)
                loss = loss_fn(output, target).item()  # Muti class classification
            else:
                print("Wrong neural network type, exit.")
                sys.exit()

            '''Attach true label'''
            act_values = target.tolist()
            all_true_values.extend(act_values)
    accuracy, f1, precision, recall = get_performance(all_pred_values, all_true_values, nn_type)
    losses = loss / len(test_loader)
    print("Loss = ", losses)
    return losses, accuracy, f1, precision, recall


'''
Testing the multi-model
    input parameter: 
        models: copy of the models
        loss_fn: loss function 
        test_loader: processed data with Dataloader
        epochs: local client epochs
        device: run by device
    output: 
        the local model weight 
'''


def multi_model_test(models, loss_fn, test_loader, nn_type, device="cpu", conf_rule=0.7):
    loss_recording = []
    ture_recording = []
    pred_recording = []
    conf_recording = []
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
                    loss = loss_fn(output, target).item()  # Muti class classification
                else:
                    print("Wrong neural network type, exit.")
                    sys.exit()

                '''Attach true label'''
                act_values = target.tolist()
                all_true_values.extend(act_values)
            # save predictions
            pred_recording.append(all_pred_values)
            conf_recording.append(all_conf_values)
            ture_recording.append(all_true_values)
        loss_recording.append(loss / len(test_loader))
    # combine model predictions by majority voting, ground truth, and loss
    voting = com_prediction_with_rule(pred_recording, conf_recording, confs_rule=conf_rule)  # combine prediction
    final_true = ture_recording[0]  # find one of the ground truth
    final_loss = sum(loss_recording) / len(loss_recording)  # Average the loss
    accuracy, f1, precision, recall = get_performance(voting, final_true, nn_type)
    return final_loss, accuracy, f1, precision, recall


# Combine model predictions
def com_prediction(preds):
    res = []
    for i in range(len(preds[0])):
        temp_list = []
        for j in range(len(preds)):
            temp_list.append(preds[j][i])
        res.append(most_frequent(temp_list))
    return res


# Combine model predictions if 11 is the most label then find the next most frequent element
def com_prediction_with_rule(preds, confs, confs_rule=0.7):
    res = []
    for i in range(len(preds[0])):
        try:
            temp_list = []
            for j in range(len(preds)):
                if confs[j][i].item() > confs_rule:
                    temp_list.append(preds[j][i].item())
            # Remove elements which has 11
            temp_list = list(filter((11).__ne__, temp_list))
            res.append(max(set(temp_list), key=temp_list.count))
        except:
            try:
                temp_list_exp = []
                for j in range(len(preds)):
                    temp_list_exp.append(preds[j][i].item())
                # Remove elements which has 11
                temp_list_exp = list(filter((11).__ne__, temp_list_exp))
                res.append(max(set(temp_list_exp), key=temp_list_exp.count))
            except:
                temp_list_exp = []
                for j in range(len(preds)):
                    temp_list_exp.append(preds[j][i].item())
                res.append(max(set(temp_list_exp), key=temp_list_exp.count))
    return res


# Program to find most frequent element in a list
def most_frequent(List):
    return max(set(List), key=List.count)


# Get model performance
def get_performance(y_pred, y_test, nn_type):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    if nn_type == "MLP":
        '''Binary classification score'''
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
    elif nn_type == "MLP_Mult":
        '''Multi class classification'''
        precision = metrics.precision_score(y_test, y_pred, average='macro')
        recall = metrics.recall_score(y_test, y_pred, average='macro')
        f1 = metrics.f1_score(y_test, y_pred, average='macro')
    else:
        print("Wrong neural network type, exit.")
        sys.exit()
    print("Classification Metrics: ")
    # print(metrics.confusion_matrix(y_test, y_pred))
    # print(metrics.classification_report(y_test, y_pred))
    print('Accuracy =', accuracy)
    print('Precision =', precision)
    print('Recall =', recall)
    print('F1 =', f1)
    return accuracy, f1, precision, recall


# -----------------------------------
def make_dir(path, dir_name):
    try:
        out_path = path + "/" + dir_name
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            return True
        else:
            print("Directory %s already exists" % dir_name)
            return False
    except OSError:
        print('Error: Creating directory. ' + out_path)
        sys.exit(0)


def cosine_similarity_element_wise(A, B):
    # Convert OrderedDict to dict
    A_dict = dict(A)
    B_dict = dict(B)
    keys_A = A_dict.keys()
    keys_B = B_dict.keys()
    x = []
    y = []
    # convert weight to be a linear array
    if keys_A == keys_B:
        for key in keys_A:
            temp_A = A_dict[key].numpy()
            # temp_A_vector = np.reshape(temp_A, temp_A.size)
            # print(len(temp_A_vector))
            # sys.exit()
            temp_B = B_dict[key].numpy()
            x = np.concatenate((x, temp_A), axis=None)
            y = np.concatenate((y, temp_B), axis=None)
            # num += np.multiply(temp_A, temp_B).sum()
            # den_x += np.square(temp_A).sum()
            # den_y += np.square(temp_B).sum()
        # print(len(x))
        # print(len(y))
    else:
        print("Two model are not consistent.")
        sys.exit()
    return np.dot(x, y)/(norm(x)*norm(y))


def cosine_similarity_last_layer(A, B):
    # Convert OrderedDict to dict
    A_dict = dict(A)
    B_dict = dict(B)
    keys_A = list(A_dict.keys())
    keys_B = list(B_dict.keys())
    x = []
    y = []
    # convert last layer and bias to be a linear array
    try:
        x = np.concatenate((x, A_dict[keys_A[-2]].numpy()), axis=None)
        # x = np.concatenate((x, A_dict[keys_A[-1]].numpy()), axis=None)
        y = np.concatenate((y, B_dict[keys_B[-2]].numpy()), axis=None)
        # y = np.concatenate((y, B_dict[keys_B[-1]].numpy()), axis=None)
    except:
        x = np.concatenate((x, A_dict[keys_A[-2]]), axis=None)
        y = np.concatenate((y, B_dict[keys_B[-2]]), axis=None)

    return np.dot(x, y) / (norm(x) * norm(y))


# compute the similarity matrix, return np array
def cosine_similarity_matrix(data):
    res_matrix = []
    for i in range(len(data)):
        temp_sim = []
        for j in range(len(data)):
            temp_sim.append(np.dot(data[i], data[j]) / (norm(data[i]) * norm(data[j])))
        res_matrix.append(temp_sim)
    return np.array(res_matrix)


def similarity_finder(folder_path):
    # define a list for store similarity
    res = []
    for i in range(30):
        temp_res = []
        model_A = torch.load(folder_path + "model_client_" + str(i) + ".pth")
        for j in range(30):
            model_B = torch.load(folder_path + "model_client_" + str(j) + ".pth")
            temp_res.append(cosine_similarity_last_layer(model_A, model_B))
        res.append(temp_res)
    df = pd.DataFrame(res)
    df.to_csv(folder_path + "ex_imbalance_last_layer_weight_similarity.csv")
    print(df)
    pass


# search dict values and return the key
def dict_search(my_dict, target, alart=True):
    key_list = list(my_dict.keys())
    val_list = list(my_dict.values())
    res = []
    init_index = 0
    # find the row position of target
    for index, temp_list in enumerate(val_list):
        if target in temp_list:
            res.append(index)
    # check if target appear more than once
    if len(res) == 1:
        return key_list[res[0]]
    elif len(res) == 0:
        if alart:
            print("Client " + str(target) + " identified as new client, use init global model.")
        return init_index
    else:
        print("Error: client " + str(target) + " used more than once.")
        sys.exit()


def find_best_k(data, round_num, max_K=10):
    K_value_range = [1, max_K]
    K = range(K_value_range[0], K_value_range[1])

    dist_points_from_cluster_center = []
    for no_of_clusters in K:
        k_model = KMeans(n_clusters=no_of_clusters, algorithm="lloyd")
        k_model.fit(data)
        dist_points_from_cluster_center.append(k_model.inertia_)

    plt.plot(K, dist_points_from_cluster_center, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title("The Elbow Method using Distortion of round " + str(round_num))
    plt.show()
    curr_path = os.getcwd()
    make_dir(curr_path, "K_plot")
    plot_saving_path = curr_path + "/K_plot/Elbow Method of round " + str(round_num) + ".pdf"
    plt.savefig(plot_saving_path)

    def calc_distance(x1, y1, a, b, c):
        d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
        return d

    a = dist_points_from_cluster_center[0] - dist_points_from_cluster_center[8]
    b = K[8] - K[0]
    c1 = K[0] * dist_points_from_cluster_center[8]
    c2 = K[8] + dist_points_from_cluster_center[0]
    c = c1 - c2

    distance_of_points_from_line = []
    for k in range(9):
        distance_of_points_from_line.append(calc_distance(K[k], dist_points_from_cluster_center[k], a, b, c))

    best_k = distance_of_points_from_line.index(max(distance_of_points_from_line)) + 1
    print("Found the best k", best_k)
    return best_k


def record_clients_clustering(model_clients_record, current_clients_index, current_labels, num_clusters):
    model_clients_record.clear()
    for i in range(num_clusters):
        temp_index = [j for j in range(len(current_labels)) if current_labels[j] == i]
        temp_clients_index = [current_clients_index[k] for k in temp_index]
        model_clients_record[i+1] = temp_clients_index
    return model_clients_record


# Find the optimized groups based on the similarity matrix (None overlapping selection)
def group_clients_from_sim_matrix(sim_matrix, clients_list):
    # define res dict
    res = {}
    model_index = 1
    used_clients = []
    # Get number of clients
    num_clients = len(clients_list)
    for client_index in range(num_clients):
        temp_client_sim = sim_matrix[client_index]
        # get element index which sim value is > 0.8
        similar_client_index = [index for index in range(len(temp_client_sim)) if temp_client_sim[index] > 0.8]
        # remove used client index
        similar_client_index = [i for i in similar_client_index if i not in used_clients]
        if similar_client_index:
            # Assign client index to global models
            res[model_index] = similar_client_index
            model_index += 1
        # recording used clients index
        used_clients = used_clients + similar_client_index
    return res


# Find the optimized groups based on the similarity matrix (overlapping selection)
def overlapping_group_clients_from_sim_matrix(sim_matrix, clients_list):
    # define res dict
    res = {}
    res_sim = {}
    model_index = 1
    used_clients = []
    # Get number of clients
    num_clients = len(clients_list)
    for client_index in range(num_clients):
        temp_client_sim = sim_matrix[client_index]
        # get element index which sim value is > 0.8
        similar_client_index = [index for index in range(len(temp_client_sim)) if temp_client_sim[index] > 0.8]
        # Get similarity value of similar clients
        similar_client_sim = list(map(temp_client_sim.__getitem__, similar_client_index))
        # get the overlapping element index which sim value is > 0.5
        overlapping_client_selection = [index for index in range(len(temp_client_sim)) if temp_client_sim[index] > 0.5]
        # remove existing similar client index from overlapping client selection
        overlapping_client_selection = [i for i in overlapping_client_selection if i not in similar_client_index]
        # Get similarity value for each overlapping selection
        overlapping_client_selected_sim = list(map(temp_client_sim.__getitem__, overlapping_client_selection))
        # remove used client index
        similar_client_index = [i for i in similar_client_index if i not in used_clients]
        if similar_client_index:
            # Assign client index to global models
            res[model_index] = similar_client_index + overlapping_client_selection
            res_sim[model_index] = similar_client_sim + overlapping_client_selected_sim
            model_index += 1
        # recording used clients index
        used_clients = used_clients + similar_client_index
    return res, res_sim



if __name__ == '__main__':
    # data_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/LR_model/CICIDS2017/"
    data_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    data_path = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/MLP_model/data/partition_attacks_2.pkl"
    # hyper-parameters
    epochs = 50
    learning_rate = 0.01
    batch_size = 64
    # create df for results recording
    res_list = []
    for i in range(30):
        single_client_index = i
        # Setting parameters
        neural_network = "MLP_Mult"
        # -------------------Create folder for model----------------------
        curr_path = os.getcwd()
        make_dir(curr_path, "models")
        # -------------------load datasets----------------------
        # (x_train_un_bin, y_train_un_bin), (x_test, y_test_bin) = data_preprocessing.read_2019_data(data_dir)
        # (x_train_un_bin, y_train_un_bin) = data_preprocessing.read_data_from_pickle(data_path, 17)
        (x_train_un_bin, y_train_un_bin) = data_preprocessing.regenerate_data(data_path, single_client_index)
        x_test, y_test_bin = data_preprocessing.testing_data_extraction(data_dir, y_train_un_bin)
        # (_, _), (x_test, y_test_bin) = data_preprocessing.read_2019_data(data_dir)
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
        model_weights, _ = train(model, optimizer, loss_fn, train_loader, epochs, neural_network, i, device=DEVICE)
        training_time = (time.time() - train_time) / 60
        print("---Training time: %s minutes. ---" % training_time)
        # save model
        saving_model_name = "models/model_client_" + str(single_client_index) + ".pt"
        torch.save(copy.deepcopy(model_weights), saving_model_name)
        # -------------------Testing model----------------------
        test_time = time.time()
        test_data = CustomDataset(x_test, y_test_bin, neural_network)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        model.load_state_dict(copy.deepcopy(model_weights))
        loss, accuracy, f1, precision, recall = test(model, loss_fn, test_loader, neural_network, device=DEVICE)
        res_list.append([accuracy, f1, precision, recall, loss, training_time])
        print("---Testing time: %s minutes. ---" % ((time.time() - test_time) / 60))
    res_df = pd.DataFrame(res_list, columns=['accuracy', 'f1', 'precision', 'recall', 'loss', 'training_time'])
    res_df.to_csv("models/clients_data_res.csv")
