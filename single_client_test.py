import copy
import random
import warnings
import sys
import time
import os
import pickle
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn import metrics

import utils
import models
from data_utils import CustomDataset


# Set cuda
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cuda_num = random.randint(0, (torch.cuda.device_count()-1))
    cuda_name = "cuda:" + str(cuda_num)
    DEVICE = torch.device(cuda_name)


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


# extract the corresponding testing data based on the class
def testing_data_extraction(testing_data, label):
    (x_test, y_test_bin) = testing_data
    un_label = np.unique(label)
    # convert to df
    df = pd.DataFrame(x_test)
    df["label"] = y_test_bin
    # extract testing based on label
    new_df = df[df['label'].isin(un_label)]
    # convert back to nd array
    new_testing_X = new_df.iloc[:, :-1].to_numpy()
    new_testing_y = new_df.iloc[:, -1].to_numpy()
    # Verify
    unique, counts = np.unique(new_testing_y, return_counts=True)
    print("New client testing data distribution:", dict(zip(unique, counts)))
    return new_testing_X, new_testing_y


# -----------------------------------
# Get model performance
def get_performance(y_pred, y_test, classification_method):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    if classification_method == "Binary":
        '''Binary classification score'''
        # precision = metrics.precision_score(y_test, y_pred)
        # recall = metrics.recall_score(y_test, y_pred)
        # f1 = metrics.f1_score(y_test, y_pred)
    elif classification_method == "Multi":
        '''Muti class classification'''
        precision_all_classes = metrics.precision_score(y_test, y_pred, average=None)
        recall_all_classes = metrics.recall_score(y_test, y_pred, average=None)
        f1_all_classes = metrics.f1_score(y_test, y_pred, average=None)
        print('Precision for all classes =', precision_all_classes)
        print('Recall for all classes =', recall_all_classes)
        print('F1 for all classes =', f1_all_classes)
        precision = metrics.precision_score(y_test, y_pred, average='macro')
        recall = metrics.recall_score(y_test, y_pred, average='macro')
        f1 = metrics.f1_score(y_test, y_pred, average='macro')
    else:
        print("Classification method does not found.")
        sys.exit()
    print("Classification Metrics: ")
    # print(metrics.confusion_matrix(y_test, y_pred))
    # print(metrics.classification_report(y_test, y_pred))
    print('Accuracy =', accuracy)
    print('Precision =', precision)
    print('Recall =', recall)
    print('F1 =', f1)
    return accuracy, f1, precision, recall


# ------------------------------------------------
def test(model, loss_fn, test_loader, curr_path, classification_method="Multi", device="cpu"):
    results_path = curr_path + "/results/cent-binary.csv"
    model_path = curr_path + "/models/cent-binary" + ".pth"

    # model.load_state_dict(torch.load(model_path))
    model.to(device)
    # test the model
    model.eval()
    y_pred_list = []
    loss = 0.0

    all_true_values = []
    all_pred_values = []
    print("Test the global model")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if classification_method == "Binary":
                '''Attach Binary classification label'''
                # y_pred_tag = torch.round(output)
                # y_pred_tensor = y_pred_tag.flatten()
                # y_pred_np_int = y_pred_tensor.type(torch.float32)
                # pred_values = y_pred_np_int.tolist()
                # all_pred_values.extend(pred_values)
                # loss += loss_fn(output, target.reshape(-1, 1)).item()      # Binary classification

            '''Attach true label'''
            act_values = target.tolist()
            all_true_values.extend(act_values)

            # write_predictions(results_path, act_values, pred_values)
            if classification_method == "Multi":
                '''Attach Muti class classification label'''
                if device == "cpu":
                    _, predictions = torch.max(output, 1)
                else:
                    _, predictions = torch.max(output.cpu(), 1)
                all_pred_values.extend(predictions)
                loss = loss_fn(output, target).item()                   # Muti class classification
    accuracy, f1, precision, recall  = get_performance(all_pred_values, all_true_values, classification_method)
    losses = loss/len(test_loader)
    return losses, accuracy, f1, precision, recall


if __name__ == "__main__":
    data_dir = '2017_data/'
    # hyper-parameters
    epochs = 50
    learning_rate = 0.01
    batch_size = 64
    MLP_first_hidden = 64
    MLP_second_hidden = 128
    num_classes = 7
    num_clients = 20
    training_data_name = str(num_clients) + '_training.pkl'
    classification = "Multi"
    neural_network = "MLP_Mult"
    # create df for results recording
    res_list = []
    # -------------------Create folder for model----------------------
    curr_path = os.getcwd()
    make_dir(curr_path, "results")
    # -------------- load datasets ----------------------
    print("Loading data...")
    partitioned_data, testing_data, _ = utils.load_data(data_dir, training_data=training_data_name)
    for i in range(num_clients):
        print("\nClient ", str(i+1), "training...")
        (X_train, y_train) = partitioned_data[i]
        # --------------Print Shape-----------------
        # Verify
        unique, counts = np.unique(y_train, return_counts=True)
        print("Client ", str(i+1), "data distribution:", dict(zip(unique, counts)))
        # -------------- Preprocess datasets -----------------
        train_data = CustomDataset(X_train, y_train, neural_network)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        x_test_new, y_test_new = testing_data_extraction(testing_data, y_train)
        test_data = CustomDataset(x_test_new, y_test_new, neural_network)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        # -------------- Define model ----------------------
        if classification == "Binary":
            model = models.MLP(input_shape=X_train.shape[1]).to(DEVICE).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            loss_fn = nn.BCELoss()  # Binary classification
        elif classification == "Multi":
            model = models.MLP_Mult(input_shape=X_train.shape[1], first_hidden=MLP_first_hidden, second_hidden=MLP_second_hidden, num_classes=num_classes).to(DEVICE).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            loss_fn = nn.CrossEntropyLoss()  # Muti class classification
        else:
            print("Classification method does not found.")
            sys.exit()
        # -------------------Training model----------------------
        train_time = time.time()
        model_weights, _ = utils.train(model, optimizer, loss_fn, train_loader, epochs, neural_network, i, device=DEVICE)
        training_time = (time.time() - train_time) / 60
        print("---Training time: %s minutes. ---" % training_time)
        # -------------------Testing model----------------------
        test_time = time.time()
        model.load_state_dict(copy.deepcopy(model_weights))
        loss, accuracy, f1, precision, recall = test(model, loss_fn, test_loader, curr_path, device=DEVICE)
        res_list.append([accuracy, precision, recall, f1, loss, training_time])
        print("---Testing time: %s minutes. ---" % ((time.time() - test_time) / 60))
    res_df = pd.DataFrame(res_list, columns=['Accuracy', 'Precision', 'Recall', 'F1', 'Loss', 'Training_time(min)'])
    res_df.to_csv("results/2017_clients_data_res.csv")
