import os
import sys
import time
import copy
import pandas as pd
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader

import utils
import models
import similarity_utils
import data_preprocessing
from data_utils import CustomDataset


# Set cuda
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cuda_num = random.randint(0, (torch.cuda.device_count() - 1))
    cuda_name = "cuda:" + str(cuda_num)
    DEVICE = torch.device(cuda_name)


# Program to find most frequent element in a list
def most_frequent(List):
    return max(set(List), key=List.count)


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


# extract one sample from each sample
def extract_un_testing_data(X_test, y_test):
    res_df = pd.DataFrame()
    # convert back to df
    df_data = pd.DataFrame(X_test)
    df_data['label'] = y_test
    # print(df_data.head)
    # print(df_data.shape)
    unique_labels = [*set(y_test)]
    for label in unique_labels:
        temp_df = df_data[df_data['label'] == label]
        sample_data = temp_df.iloc[[0]]
        res_df = pd.concat([res_df, sample_data])
    # print(res_df.head)
    # print(res_df.shape)
    # convert back to nd array
    new_X = res_df.iloc[:, :-1].to_numpy()
    new_y = res_df.iloc[:, -1].to_numpy()
    return new_X, new_y


def single_test_compare(models, loss_fn, test_loader, nn_type, device="cpu", conf_rule=0.7):
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
        print("Test the model ", str(index))
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
                        probs = torch.nn.functional.softmax(output, dim=1)
                        conf, predictions = torch.max(probs, 1)
                        # print(probs)
                        # print(predictions)
                    else:
                        probs = torch.nn.functional.softmax(output.cpu(), dim=1)
                        conf, predictions = torch.max(probs, 1)
                        # print(probs)
                        # print(predictions)
                    # if the prediction is the value we set for noise give up the current prediction
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
    # print predictions and label for each model
    pred_recording_np = np.array(pred_recording)
    pred_recording_np = pred_recording_np.transpose()
    conf_recording_np = np.array(conf_recording)
    conf_recording_np = conf_recording_np.transpose()
    # majority voting function
    voting = com_prediction_with_rule(pred_recording, conf_recording, confs_rule=conf_rule)
    # for i in range(len(ture_recording)):
    #     print(ture_recording[i])

    # Print predictions
    for i in range(len(ture_recording[0])):
        # print("Get predictions: ", len(pred_recording_np[i]))
        unique, counts = np.unique(pred_recording_np[i], return_counts=True)
        # print("Predict: ", list(zip(pred_recording_np[i], conf_recording_np[i])), " Ground truth:", ture_recording[0][i], "Count:", np.count_nonzero(pred_recording_np[i] == ture_recording[0][i]), " Voting results:", voting[i], "Count:", np.count_nonzero(pred_recording_np[i] == voting[i]), "  Predict counter:", dict(zip(unique, counts)))
        print("Predict: ", pred_recording_np[i], " Ground truth:", ture_recording[0][i], "Count:", np.count_nonzero(pred_recording_np[i] == ture_recording[0][i]), " Voting results:", voting[i], "Count:", np.count_nonzero(pred_recording_np[i] == voting[i]), "  Predict counter:", dict(zip(unique, counts)))

    final_true = ture_recording[0]  # find one of the ground truth
    final_loss = sum(loss_recording) / len(loss_recording)  # Average the loss
    accuracy, f1, precision, recall = utils.get_performance(voting, final_true, nn_type)
    return final_loss, accuracy, f1, precision, recall


# Save layer weight change into CSV
def weight_change_to_csv(weight_change, client_index):
    # -------------------Create folder for weight change----------------------
    curr_path = os.getcwd()
    utils.make_dir(curr_path, "weight_change_log")

    # Select weight parameter from dict
    # weights = {key: weight_change[key] for key in weight_change.keys() & {'fc1.weight', 'fc2.weight', 'fc3.weight', 'fc4.weight'}}
    # df_weight = pd.DataFrame.from_dict(weights)
    with pd.ExcelWriter(curr_path + "/weight_change_log/weight_change_of_client_" + str(client_index) + ".xlsx") as writer:
        keys = list(weight_change.keys())
        for key in keys:
            temp_weight = weight_change[key]
            temp_weight_df = pd.DataFrame(temp_weight)
            sheet_name = "Layer " + str(key)
            temp_weight_df.to_excel(writer, sheet_name=sheet_name)
    pass


if __name__ == '__main__':
    data_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    # data_dir = "/home/jliu/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    (x_train_un_bin, y_train_un_bin), (x_test, y_test_bin), (_, _) = data_preprocessing.read_2019_data(data_dir)
    # --------------------Read global models-----------------------
    model_path = "weight_records/"
    global_models = []
    for i in range(24):
        temp_global_model_name = model_path + "static_global_model_" + str(i) + ".pt"
        global_models.append(torch.load(temp_global_model_name, map_location=DEVICE))
    print("Number of global models read: ", len(global_models))
    # sys.exit()
    # --------------------Train client models-----------------------
    # client_models = []
    # data_path = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/MLP_model/data/partition_attacks_2.pkl"
    # epochs = 20
    # learning_rate = 0.01
    # batch_size = 64
    # for i in range(3):
    #     single_client_index = i
    #     # Setting parameters
    #     neural_network = "MLP_Mult"
    #     # -------------------Create folder for model----------------------
    #     curr_path = os.getcwd()
    #     utils.make_dir(curr_path, "models")
    #     # -------------------load datasets----------------------
    #     (x_train, y_train) = data_preprocessing.regenerate_data(data_path, single_client_index)
    #     # x_train_new, y_train_new = data_preprocessing.noise_generator(x_train_un_bin, y_train_un_bin, x_train, y_train)
    #     # check to use GPU or not
    #     use_cuda = torch.cuda.is_available()
    #     DEVICE = torch.device("cuda:0" if use_cuda else "cpu")
    #     print(DEVICE, "are using for training and testing.")
    #     # -------------------Set model----------------------
    #     # Model, Optimizer, Loss func
    #     '''Select the different loss function for binary or muti class classification'''
    #     if neural_network == "MLP":
    #         model = models.MLP(input_shape=x_train.shape[1]).to(DEVICE)
    #         loss_fn = nn.BCELoss()  # Binary classification
    #     elif neural_network == "MLP_Mult":
    #         model = models.MLP_Mult(input_shape=x_train.shape[1], num_classes=12).to(DEVICE)
    #         loss_fn = nn.CrossEntropyLoss()  # Muti class classification
    #     else:
    #         print("Wrong neural network type, exit.")
    #         sys.exit()
    #     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #     # -------------------Training model----------------------
    #     train_time = time.time()
    #     train_data = CustomDataset(x_train, y_train, neural_network)
    #     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #     client_model, _ = utils.train(model, optimizer, loss_fn, train_loader, epochs, neural_network, i, device=DEVICE)
    #     weight_changes = similarity_utils.weight_changes_cal(model.state_dict(), client_model.state_dict())
    #     weight_change_to_csv(weight_changes, single_client_index)
    #     # sys.exit()
    #     client_models.append(client_model)
    #     training_time = (time.time() - train_time) / 60
    #     print("---Training time: %s minutes. ---" % training_time)
    # --------------------Server Testing-----------------------
    test_time = time.time()
    batch_size = 64
    neural_network = "MLP_Mult"
    loss_fn = nn.CrossEntropyLoss()
    # data_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    x_test_new, y_test_new = extract_un_testing_data(x_test, y_test_bin)
    test_data = CustomDataset(x_test_new, y_test_new, neural_network)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Testing global models
    # single_test_compare(global_models[1:], loss_fn, test_loader, neural_network)
    model_loss, model_accuracy, model_f1, model_precision, model_recall = single_test_compare(global_models[1:],
                                                                                              loss_fn, test_loader,
                                                                                              neural_network, device=DEVICE, conf_rule=0.7)
    # single_test_compare(client_models, loss_fn, test_loader, neural_network)
    server_running_time = ((time.time() - test_time) / 60)
    print("Global model, Loss %f, Accuracy %f, F1 %f, Total Running time(min): %s" % (
        model_loss, model_accuracy, model_f1, server_running_time))
