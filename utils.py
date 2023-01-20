import os
import time
import sys
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn import metrics

import models
import data_preprocessing
from data_utils import CustomDataset

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


def train(model, optimizer, loss_fn, train_loader, epochs, nn_type, device="cpu"):
    print("Train the Client model")
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
        print(f'Epoch {(e + 1) + 0:02}: | Loss: {epoch_loss / len(train_loader):.5f}')
    return model.state_dict()


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
                loss += loss_fn(output, target.reshape(-1, 1)).item()      # Binary classification
            elif nn_type == "MLP_Mult":
                '''Attach Muti class classification label'''
                if device == "cpu":
                    _, predictions = torch.max(output, 1)
                else:
                    _, predictions = torch.max(output.cpu(), 1)
                all_pred_values.extend(predictions)
                loss = loss_fn(output, target).item()                   # Muti class classification
            else:
                print("Wrong neural network type, exit.")
                sys.exit()

            '''Attach true label'''
            act_values = target.tolist()
            all_true_values.extend(act_values)
    accuracy, f1, precision, recall = get_performance(all_pred_values, all_true_values, nn_type)
    losses = loss/len(test_loader)
    return losses, accuracy, f1, precision, recall


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


if __name__ == '__main__':
    data_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/LR_model/CICIDS2017/"
    # hyper-parameters
    epochs = 5
    learning_rate = 0.01
    batch_size = 64
    # Setting parameters
    neural_network = "MLP_Mult"

    # -------------------load datasets----------------------
    (x_train_un_bin, y_train_un_bin), (x_test, y_test_bin) = data_preprocessing.read_2017_data(data_dir)

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
    model_weights = train(model, optimizer, loss_fn, train_loader, epochs, neural_network, device=DEVICE)
    print("---Training time: %s minutes. ---" % ((time.time() - train_time) / 60))

    # -------------------Testing model----------------------
    test_time = time.time()
    test_data = CustomDataset(x_test, y_test_bin, neural_network)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.load_state_dict(copy.deepcopy(model_weights))
    loss, accuracy = test(model, loss_fn, test_loader, neural_network, device=DEVICE)
    print("---Testing time: %s minutes. ---" % ((time.time() - test_time) / 60))
