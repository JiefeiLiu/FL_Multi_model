import sys
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import random
import time
import matplotlib.pyplot as plt
import matplotlib


# -----------------------------------
# Read CICIDS2017 data
def read_2017_data(path):
    X = np.load(path + "cic17_all_X_org.npy")
    y = np.load(path + "cic17_all_y_org.npy")

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    unique, counts = np.unique(y, return_counts=True)
    print("Total data shape", dict(zip(unique, counts)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=True,
                                                        stratify=y)
    # validation/noise data generator
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=1, shuffle=True,
                                                      stratify=y_train)

    unique, counts = np.unique(y_train, return_counts=True)
    print("Training shape", dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print("Testing shape", dict(zip(unique, counts)))
    unique, counts = np.unique(y_val, return_counts=True)
    print("Validation shape", dict(zip(unique, counts)))
    # print(str(len(y_test) / (len(y_train) + len(y_test))))
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


# Read CICDDoS2019 data
def read_2019_data(path):
    X_train = np.load(path + "X_samp_train_mult500k.npy")
    X_test = np.load(path + "X_samp_test_mult500k.npy")
    y_train = np.load(path + "y_samp_train_mult500k.npy")
    y_test = np.load(path + "y_samp_test_mult500k.npy")

    print("X train shape: ", X_train.shape)
    print("y train shape: ", y_train.shape)
    print("X test shape: ", X_test.shape)
    print("y test shape: ", y_test.shape)

    '''re-split the training and testing'''
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.45, random_state=1, shuffle=True,
                                                      stratify=y_train)

    unique, counts = np.unique(y_train, return_counts=True)
    print("Training shape", dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print("Testing shape", dict(zip(unique, counts)))
    unique, counts = np.unique(y_val, return_counts=True)
    print("Validation shape", dict(zip(unique, counts)))

    # print(str(len(y_test) / (len(y_train) + len(y_test))))
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


# Partition the data with equal and balance subsets
def partition_bal_equ(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    # Calculate the partition size
    partition_size = int(len(y) / num_partitions)
    res = []
    y_count = []
    for i in range(num_partitions):
        # if it is in last partition direct append data into res
        if i == (num_partitions - 1):
            res.append((X, y))
            # Count y to check the data balance
            unique, counts = np.unique(y, return_counts=True)
            y_count.append(dict(zip(unique, counts)))
        else:
            X, X_temp, y, y_temp = train_test_split(X, y, test_size=partition_size, random_state=1, shuffle=True,
                                                    stratify=y)
            res.append((X_temp, y_temp))
            # Count y to check the data balance
            unique, counts = np.unique(y_temp, return_counts=True)
            y_count.append(dict(zip(unique, counts)))
    # print("Label count: ", y_count)
    return res


def partition_unbal_equ(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    # Calculate the partition size
    partition_size = int(len(y) / num_partitions)
    res = []
    y_count = []
    for i in range(num_partitions):
        # if it is in last partition direct append data into res
        if i == (num_partitions - 1):
            res.append((X, y))
            # Count y to check the data balance
            unique, counts = np.unique(y, return_counts=True)
            y_count.append(dict(zip(unique, counts)))
        else:
            X, X_temp, y, y_temp = train_test_split(X, y, test_size=partition_size, random_state=1, shuffle=True)
            res.append((X_temp, y_temp))
            # Count y to check the data balance
            unique, counts = np.unique(y_temp, return_counts=True)
            y_count.append(dict(zip(unique, counts)))
    # print("Label count: ", y_count)
    return res


def partition_unbal_unequ(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    total_data = X.shape[0]
    max_user_data = total_data // num_partitions
    min_user_data = int(max_user_data * 0.80)  # minimum number of samples per user
    random.seed(0)
    # Generate random partition size
    client_data_size = []
    for k in range(num_partitions):
        if k == (num_partitions - 1):
            client_data_size.append(max_user_data)
        else:
            data_size = random.randint(min_user_data, max_user_data)  # total number of data records a client can have
            client_data_size.append(data_size)
            total_data = total_data - data_size
            max_user_data = total_data // (num_partitions - k - 1)
            # print(max_user_data)
    # print(client_data_size)
    # print(sum(client_data_size))
    # print(len(y))

    # Partition data
    res = []
    y_count = []
    for i in range(num_partitions):
        # if it is in last partition direct append data into res
        if i == (num_partitions - 1):
            res.append((X, y))
            # Count y to check the data balance
            unique, counts = np.unique(y, return_counts=True)
            y_count.append(dict(zip(unique, counts)))
        else:
            X, X_temp, y, y_temp = train_test_split(X, y, test_size=client_data_size[i], random_state=1, shuffle=True)
            res.append((X_temp, y_temp))
            # Count y to check the data balance
            unique, counts = np.unique(y_temp, return_counts=True)
            y_count.append(dict(zip(unique, counts)))
    # print("Label count: ", y_count)
    return res


def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""
    if total < 1:
        total = int(total * 100)
        dividers = sorted(random.sample(range(1, total), n - 1))
        res = [a - b for a, b in zip(dividers + [total], [0] + dividers)]
        res = [i / 100 for i in res]
    else:
        dividers = sorted(random.sample(range(1, total), n - 1))
        res = [a - b for a, b in zip(dividers + [total], [0] + dividers)]
    return res


def stacked_bar_plot(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['brg_r'](np.linspace(0.1, 0.95, data.shape[1])) # RdYlGn_r or brg_r
    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 30}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    # ax.yaxis.set_visible(False)
    ax.set_yticks(list(range(1, 21, 2)))

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
        r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # ax.bar_label(rects, label_type='center', color=text_color)
    org_label = ['Normal', 'DDoS', 'DoS Goldeye', 'Dos Hulk', 'Dos SlowHTTP', 'Dos Slowloris', 'FTP Patator', 'Brute Force', 'SSH Patator']
    # org_label = ['Normal', 'DDoS SNMP', 'DDoS DNS', 'DDoS MSSQL', 'DDoS NetBIOS', 'DDoS UDP', 'DDoS SSDP', 'DDoS LDAP', 'DDoS NTP', 'UDP-Lag', 'SYN']
    ax.legend(org_label, ncol=2, bbox_to_anchor=(0, 1), loc='lower left', fontsize=25)

    return fig, ax


# re-generate the list to the length of number of class
def regenerate_list(data_list, label_index, number_class=11):
    label_index_int = list(map(int, label_index))
    res_list = []
    for i in range(number_class):
        if i in label_index_int:
            temp_index = label_index_int.index(i)
            res_list.append(data_list[temp_index])
        else:
            res_list.append(0)
    return res_list


# plot horizontal stacked bar chart
def plot_stacked_bar(partition_data, saving_path, saving_name, number_class=11):
    stat = []
    for i in range(len(partition_data)):
        (X_train, y_train) = partition_data[i]
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        # add zeros if the unique train is not have 11 length
        counts_train_new = regenerate_list(counts_train, unique_train, number_class=number_class)
        stat.append(counts_train_new)
    print("Stat Shape", len(stat), ",", len(stat[0]))
    data_dist = {}
    for i in range(len(stat)):
        data_dist[i + 1] = stat[i]
    # print(data_dist)
    class_names = list(range(1, number_class + 1))
    # print(data_dist)
    # print(class_names)
    stacked_bar_plot(data_dist, class_names)
    # plt.title("Clients' data distribution")
    plt.ylabel("Clients")
    plt.xlabel("Class distribution")
    plt.savefig(saving_path + saving_name, bbox_inches='tight')
    # plt.show()
    pass


# partition the imbalance data into extreme case and balance
def partition_ex_imbal_equ(X: np.ndarray, y: np.ndarray, num_partitions: int, low_bound_of_classes=1,
                           high_bound_of_classes=3, percentage_normal_traffic=60, seed=42):
    # fix random seed
    random.seed(seed)
    # define the return variable
    res = []
    # define the size of dataset for each client
    client_size = int(len(y) / num_partitions*0.8)
    # Convert np array to df
    df = pd.DataFrame(X)
    df['label'] = y
    # extract labels
    unique_train, counts_train = np.unique(y, return_counts=True)
    data_label = list(unique_train)
    # extract normal traffic flow
    normal_traffic = df[df['label'] == 0.0]
    # Extract the attack traffic flow
    attack_flow = df[df['label'] > 0.0]

    # partition the normal traffic to clients
    # for loop for number of partitions
    for i in range(num_partitions):
        # define the clients data
        clients_data_df = pd.DataFrame()
        # generate the random number around percentage of normal traffic
        client_normal_traffic_percentage = (random.randint(percentage_normal_traffic - 10, percentage_normal_traffic + 10)) / 100
        normal_partition_size = int(client_normal_traffic_percentage * client_size)
        client_normal_traffic = normal_traffic.sample(normal_partition_size)
        # Assign the normal traffic to clients data
        clients_data_df = pd.concat([clients_data_df, client_normal_traffic])
        # Remove the assigned normal traffic from original
        normal_traffic.drop(client_normal_traffic.index)
        # get remain dataset size
        attack_partition_size = client_size - normal_partition_size
        # random sample the number of classes for each client
        number_of_labels_for_client = random.randint(low_bound_of_classes, high_bound_of_classes)
        # Get the labels from attack data
        attack_label = [*set(attack_flow['label'].tolist())]
        # random sample classes for client
        random_label = random.sample(attack_label, number_of_labels_for_client)
        # random sample the size of classes for each client
        label_size_list = constrained_sum_sample_pos(len(random_label), attack_partition_size)
        # print(label_size_list)
        # if the sample size greater than the original size of current size, assign all current class and collect the remain
        remain_size = 0
        # for each client sample the corresponding classes of attack data
        for j in range(len(random_label)):
            # Get the target class data from attack traffic flow
            temp_class_flow = attack_flow[attack_flow['label'] == random_label[j]]
            # Sample the attack flow
            try:
                # if the sampling data less than original data
                temp_class_sample_flow = temp_class_flow.sample(label_size_list[j])
                # Assign attack sample to client data
                clients_data_df = pd.concat([clients_data_df, temp_class_sample_flow])
                # remove the assigned data from original attack flow
                attack_flow.drop(temp_class_sample_flow.index)
            except:
                # if the sample data greater than original data
                # assign all current data
                clients_data_df = pd.concat([clients_data_df, temp_class_flow])
                # remove the assigned data from original attack flow
                attack_flow.drop(temp_class_flow.index)
                remain_size = remain_size + (label_size_list[j] - temp_class_flow.shape[0])
        # if we current client does not reach the target size
        while remain_size > 0:
            # continue sample another class data
            # Get the labels from attack data
            attack_label = [*set(attack_flow['label'].tolist())]
            # random pick a class
            random_label_remain = random.sample(attack_label, 1)
            # Get the target class data from attack traffic flow
            temp_class_flow = attack_flow[attack_flow['label'] == random_label_remain[0]]
            try:
                # if the sampling data less than original data
                temp_class_sample_flow = temp_class_flow.sample(remain_size)
                # Assign attack sample to client data
                clients_data_df = pd.concat([clients_data_df, temp_class_sample_flow])
                # remove the assigned data from original attack flow
                attack_flow.drop(temp_class_sample_flow.index)
                break
            except:
                # if the sample data greater than original data
                # assign all current data
                clients_data_df = pd.concat([clients_data_df, temp_class_flow])
                # remove the assigned data from original attack flow
                attack_flow.drop(temp_class_flow.index)
                remain_size = remain_size - temp_class_flow.shape[0]

        # convert the client data df to np
        X_client = clients_data_df.iloc[:, :-1].to_numpy()
        y_client = clients_data_df.iloc[:, -1].to_numpy()
        res.append((X_client, y_client))
    return res


def random_client_selection(num_global_models, clients_list, low_boundary, high_boundary):
    model_clients = []
    for i in range(num_global_models):
        temp_clients_list = np.random.choice(clients_list, random.randint(low_boundary, high_boundary), replace=False)
        model_clients.append(temp_clients_list)
    return model_clients


# partition the imbalance data into extreme case and balance
def partition_ex_imbal_equ_testing(X: np.ndarray, y: np.ndarray, num_partitions: int, number_of_attack_classes=2):
    # define the return variable
    res = []
    # define the size of dataset for each client
    client_size = 100000 * (number_of_attack_classes + 1)
    # Convert np array to df
    df = pd.DataFrame(X)
    df['label'] = y
    # extract labels
    unique_train, counts_train = np.unique(y, return_counts=True)
    data_label = list(unique_train)
    # extract normal traffic flow
    normal_traffic = df[df['label'] == 0.0]
    # Extract the attack traffic flow
    attack_flow = df[df['label'] > 0.0]

    # partition the normal traffic to clients
    # for loop for number of partitions
    for i in range(num_partitions):
        # define the clients data
        clients_data_df = pd.DataFrame()
        # generate the random number around percentage of normal traffic
        client_normal_traffic_percentage = (random.randint(30, 35)) / 100
        normal_partition_size = int(client_size * client_normal_traffic_percentage)
        client_normal_traffic = normal_traffic.sample(normal_partition_size)
        # Assign the normal traffic to clients data
        clients_data_df = pd.concat([clients_data_df, client_normal_traffic])
        # Remove the assigned normal traffic from original
        normal_traffic.drop(client_normal_traffic.index)
        # Get the labels from attack data
        attack_label = [*set(attack_flow['label'].tolist())]
        # get remain dataset size
        attack_partition_size = client_size - normal_partition_size
        # random sample classes for client
        random_label = random.sample(attack_label, number_of_attack_classes)
        # random sample the size of classes for each client
        label_size_list = constrained_sum_sample_pos(len(random_label), attack_partition_size)
        # label_size_list = [100000] * number_of_attack_classes
        # if the sample size greater than the original size of current size, assign all current class and collect the remain
        remain_size = 0
        # for each client sample the corresponding classes of attack data
        for j in range(len(random_label)):
            # Get the target class data from attack traffic flow
            temp_class_flow = attack_flow[attack_flow['label'] == random_label[j]]
            # Sample the attack flow
            try:
                # if the sampling data less than original data
                temp_class_sample_flow = temp_class_flow.sample(label_size_list[j])
                # Assign attack sample to client data
                clients_data_df = pd.concat([clients_data_df, temp_class_sample_flow])
                # remove the assigned data from original attack flow
                attack_flow.drop(temp_class_sample_flow.index)
            except:
                continue
        # convert the client data df to np
        X_client = clients_data_df.iloc[:, :-1].to_numpy()
        y_client = clients_data_df.iloc[:, -1].to_numpy()
        res.append((X_client, y_client))
    return res


def verify_class_distribution(pickle_path):
    with open(pickle_path, 'rb') as file:
        # Call load method to deserialze
        partitioned_data = pickle.load(file)
    for client_index in range(len(partitioned_data)):
        (temp_X, temp_y) = partitioned_data[client_index]
        # Verify
        unique_train, counts_train = np.unique(temp_y, return_counts=True)
        print("Client " + str(client_index) + " train shape", dict(zip(unique_train, counts_train)))
    pass


if __name__ == "__main__":
    data_path = "2017_data/old_data/"
    # data_path = "../DoD_Misra_project/jiefei_liu/DOD/CICDDoS2019/"
    pickle_saving_path = "2017_data/"
    partition_num = 20
    num_attacks_range = [2, 5]
    start_time = time.time()
    # -------------------- Normal data partition ----------------------------
    # Split train set into partitions and randomly use one for training.
    # (X_train, y_train), testing, validation = read_2019_data(data_path)
    (X_train, y_train), testing, validation = read_2017_data(data_path)
    partitioned_data = partition_bal_equ(X_train, y_train, partition_num)
    print("The shape of partition data")
    print("The shape of X: ", len(X_train), len(X_train[0]))
    print("The shape of y: ", len(y_train))
    save_file_name = 'partition_equal_balance.pkl'
    # ------------------------------------------------------------
    # -------------------- load data  ----------------------------
    # ------------------------------------------------------------
    # (X_train, y_train), testing, validation = read_2019_data(data_path)
    # sys.exit()
    # -------------------- Extreme data partition testing ----------------------------
    # partitioned_data = partition_ex_imbal_equ_testing(X_train, y_train, partition_num, number_of_attack_classes=num_attacks_range[0])
    # save_file_name = pickle_saving_path + "partition_attacks_" + str(num_attacks_range[0]) + "_imbalance.pkl"
    # -------------------- Extreme data partition regular ----------------------------
    # partitioned_data = partition_ex_imbal_equ(X_train, y_train, partition_num,
    #                                           low_bound_of_classes=num_attacks_range[0],
    #                                           high_bound_of_classes=num_attacks_range[1], percentage_normal_traffic=60)
    # save_file_name = "2019_data/" + str(partition_num) + "_training.pkl"
    # -------------------- Save data partition ----------------------------
    # Open a file and use dump()
    with open(pickle_saving_path + save_file_name, 'wb') as file:
        # A new file will be created
        pickle.dump(partitioned_data, file)
    # saving testing
    with open(pickle_saving_path + "testing.pkl", 'wb') as file:
        # A new file will be created
        pickle.dump(testing, file)
    # saving validation
    with open(pickle_saving_path + "validation.pkl", 'wb') as file:
        # A new file will be created
        pickle.dump(validation, file)
    # ---------------------Verify data partition-----------------------------
    # Open the file in binary mode
    with open(pickle_saving_path + 'partition_equal_balance.pkl', 'rb') as file:
        # Call load method to deserialze
        partitioned_data = pickle.load(file)
    for i in range(partition_num):
        (X_train, y_train) = partitioned_data[i]
        print("Client", str(i), ":")
        print("X train shape: ", X_train.shape)
        print("y train shape: ", y_train.shape)
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        print("train label:", unique_train)
        print("train count:", counts_train)
        print()
    # ---------------------Plot data partition-----------------------------
    plot_name = "Partition_" + str(partition_num) + "_2017_balance_and_equal.pdf"
    plot_stacked_bar(partitioned_data, pickle_saving_path, plot_name)
    # ---------------------verify data partition-----------------------------
    # verify_class_distribution(save_file_name)


    # ----------------------Generate random sampling----------------------------
    # num_clients = 30
    # num_global_models = 5
    # clients_list = list(range(0, num_clients))
    # low = int(num_clients / num_global_models)
    # high = 10
    # model_client_list = random_client_selection(num_global_models, clients_list, low, high)
    # # Open a file and use dump()
    # with open('client_selection_list.pkl', 'wb') as file:
    #     # A new file will be created
    #     pickle.dump(model_client_list, file)
    # with open('client_selection_list.pkl', 'rb') as file:
    #     # Call load method to deserialze
    #     model_client_list = pickle.load(file)
    # print(model_client_list)
    # print(constrained_sum_sample_pos(10, 500))
    print("--- %s seconds ---" % (time.time() - start_time))
