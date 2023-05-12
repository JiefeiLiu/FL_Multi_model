import random
import sys

import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import utils

# -----------------------------------
# Read CICDDoS2019 data
def read_2019_data(path):
    X_train = np.load(path + "X_samp_train_mult500k.npy")
    X_test = np.load(path + "X_samp_test_mult500k.npy")
    y_train = np.load(path + "y_samp_train_mult500k.npy")
    y_test = np.load(path + "y_samp_test_mult500k.npy")

    # print("X train shape: ", X_train.shape)
    # print("y train shape: ", y_train.shape)
    # print("X test shape: ", X_test.shape)
    # print("y test shape: ", y_test.shape)

    '''re-split the training and testing'''
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1, shuffle=True,
                                                      stratify=y_train)

    unique, counts = np.unique(y_train, return_counts=True)
    print("Training shape", dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print("Testing shape", dict(zip(unique, counts)))
    unique, counts = np.unique(y_val, return_counts=True)
    print("Validation shape", dict(zip(unique, counts)))
    # print(str(len(y_test) / (len(y_train) + len(y_test))))
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


# -----------------------------------
# Read CICIDS2017 data
def read_2017_data(path):
    # multi-class classification
    X_train = np.load(path + "x_train_mul_samp.npy")
    X_test = np.load(path + "x_test.npy")
    y_train = np.load(path + "y_train_mul_samp.npy")
    y_test = np.load(path + "y_test_mul.npy")

    # print("X train shape: ", X_train.shape)
    # print("y train shape: ", y_train.shape)
    # print("X test shape: ", X_test.shape)
    # print("y test shape: ", y_test.shape)

    '''re-split the training and testing'''
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1, shuffle=True, stratify=y_train)


    unique, counts = np.unique(y_train, return_counts=True)
    print("Training shape", dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print("Testing shape", dict(zip(unique, counts)))
    unique, counts = np.unique(y_val, return_counts=True)
    print("Validation shape", dict(zip(unique, counts)))
    # print(str(len(y_test) / (len(y_train) + len(y_test))))
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def read_data_from_pickle(pickle_dir, client_index):
    # Load partitioned data
    with open(pickle_dir, 'rb') as file:
        # Call load method to deserialze
        partition_data_list = pickle.load(file)
    (client_X_train, client_y_train) = partition_data_list[client_index]
    # Verify
    unique, counts = np.unique(client_y_train, return_counts=True)
    print("Client data distribution:", dict(zip(unique, counts)))
    # X_train, X_test, y_train, y_test = train_test_split(client_X_train, client_y_train, test_size=0.2, random_state=1, shuffle=True)

    return client_X_train, client_y_train


# -----------------------------------
# regenerate data the number of instance for classes equal to the smallest size of the class
def regenerate_data(pickle_dir, client_index):
    # Load partitioned data
    with open(pickle_dir, 'rb') as file:
        # Call load method to deserialze
        partition_data_list = pickle.load(file)
    (client_X_train, client_y_train) = partition_data_list[client_index]
    # Verify
    unique, counts = np.unique(client_y_train, return_counts=True)
    print("Client data distribution:", dict(zip(unique, counts)))
    # print(min(counts))

    # convert to df
    df = pd.DataFrame(client_X_train)
    df["label"] = client_y_train
    # print(df.head())

    # Sample data
    res_df = pd.DataFrame()
    for label in unique:
        temp_data = df[df['label'] == label]
        sample_data = temp_data.sample(n=min(counts), replace=False, )
        res_df = pd.concat([res_df, sample_data])

    # convert back to nd array
    new_X = res_df.iloc[:, :-1].to_numpy()
    new_y = res_df.iloc[:, -1].to_numpy()

    # Verify
    unique, counts = np.unique(new_y, return_counts=True)
    print("New client training data distribution:", dict(zip(unique, counts)))

    # X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.2, random_state=1, shuffle=True)

    return new_X, new_y


# extract the corresponding testing data based on the class
def testing_data_extraction(data_path, label):
    (x_train_un_bin, y_train_un_bin), (x_test, y_test_bin), (_, _) = read_2019_data(data_path)
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


# Randomly drop the class for centralized scenario
def preprocess_data_with_random_drop_class(data_path, missing_class):
    (x_train_un_bin, y_train_un_bin), (x_test, y_test_bin), (_, _) = read_2019_data(data_path)
    # Verify
    unique, counts = np.unique(y_train_un_bin, return_counts=True)
    print("Original label distribution", dict(zip(unique, counts)))
    # random drop label
    drop_class = random.sample(list(unique[1:]), missing_class)
    print("Dropped class: ", drop_class)
    new_unique = list(filter(lambda x: x not in drop_class, unique))
    # convert training and testing to df
    df_train = pd.DataFrame(x_train_un_bin)
    df_train["label"] = y_train_un_bin
    df_test = pd.DataFrame(x_test)
    df_test["label"] = y_test_bin
    # extract testing based on label
    new_df_train = df_train[df_train['label'].isin(new_unique)]
    new_df_test = df_test[df_test['label'].isin(new_unique)]
    # convert back to nd array
    new_train_X = new_df_train.iloc[:, :-1].to_numpy()
    new_train_y = new_df_train.iloc[:, -1].to_numpy()
    new_testing_X = new_df_test.iloc[:, :-1].to_numpy()
    new_testing_y = new_df_test.iloc[:, -1].to_numpy()
    # Verify
    unique, counts = np.unique(new_testing_y, return_counts=True)
    print("New client testing data distribution:", dict(zip(unique, counts)))
    return (new_train_X, new_train_y), (new_testing_X, new_testing_y)


# Generate samples differ with existing class for training data
def noise_generator(x_sample, y_sample, existing_x, existing_y, percentage_noise=-1):
    uni_existing_labels, existing_label_counts = np.unique(existing_y, return_counts=True)
    df_sample = pd.DataFrame(x_sample)
    df_sample['label'] = y_sample

    # define the size of noise
    if percentage_noise == -1:
        noise_size = min(existing_label_counts)
    else:
        total_attacks = sum(existing_label_counts[1:])
        noise_size = int(total_attacks*percentage_noise)

    # filter out the existing labels
    df_sample_filtered_out = df_sample[~df_sample['label'].isin(uni_existing_labels)]
    x_sample_np = df_sample_filtered_out.iloc[:, :-1].to_numpy()
    y_sample_np = df_sample_filtered_out.iloc[:, -1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(x_sample_np, y_sample_np, test_size=noise_size, stratify=y_sample_np, shuffle=True, random_state=1)
    new_x = np.concatenate((existing_x, X_test), axis=0)
    # Generate new label y
    generate_new_y = [11] * len(y_test)
    new_y = np.concatenate((existing_y, generate_new_y), axis=0)

    # Verify
    unique, counts = np.unique(new_y, return_counts=True)
    print("New client training data with noise distribution:", dict(zip(unique, counts)))

    return new_x, new_y


if __name__ == '__main__':
    data_path = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/MLP_model/data/partition_low_9_high_9.pkl"
    # regenerate_data(data_path, 17)
    # ------------------- Model Similarity ----------------------
    # A = torch.load("models/model_client_6.pth")
    # B = torch.load("models/model_client_25.pth")
    # A = {"A": torch.tensor([[1, 1, 2],
    #                         [2, 2, 2],
    #                         [2, 1, 3]])}
    # B = {"A": torch.tensor([[1, 1, 2],
    #                         [2, 2, 2]])}
    # print(utils.cosine_similarity_element_wise(A, B))
    # utils.similarity_finder("models/Ex_imbalance/")
    # print(utils.csm(A, B))
    # print(cosine_similarity(A, B))
    # ------------------- Training data verification ----------------------
    # pickle_dir = "/Users/jiefeiliu/Documents/DoD_Misra_project/jiefei_liu/DOD/MLP_model/data/partition_attacks_2.pkl"
    # print("Loading data...")
    # # Load partitioned data
    # with open(pickle_dir, 'rb') as file:
    #     # Call load method to deserialze
    #     partition_data_list = pickle.load(file)
    # label = []
    # for index in range(len(partition_data_list)):
    #     (client_X_train, client_y_train) = partition_data_list[index]
    #     label = np.concatenate((label, client_y_train), axis=None)
    # unique, counts = np.unique(label, return_counts=True)
    # print(dict(zip(unique, counts)))
    # ------------------- data re-split verification ----------------------
    # data_dir = "../DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    data_dir = "../DoD_Misra_project/jiefei_liu/DOD/LR_model/CICIDS2017/"
    (x_train_un_bin, y_train_un_bin), (x_test, y_test_bin), (_, _) = read_2017_data(data_dir)
