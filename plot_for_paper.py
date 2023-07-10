import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import utils
import sampling


def curve_plot(comp_loss, comp_acc, plot_acc=True, plot_comp=True):
    losses_centralized = [[(0, 0.00015056385803222656), (1, 0.000155355224609375), (2, 0.00013385609436035155), (3, 0.00011074392700195312), (4, 9.893773651123047e-05), (5, 8.763913726806641e-05), (6, 8.786256408691407e-05), (7, 8.011539459228516e-05), (8, 7.885035705566407e-05), (9, 7.674244689941406e-05), (10, 7.724567413330079e-05), (11, 7.51748046875e-05), (12, 7.09059066772461e-05), (13, 7.123036956787109e-05), (14, 6.963818359375e-05), (15, 7.024172973632812e-05), (16, 7.231837463378906e-05), (17, 6.813994598388672e-05), (18, 6.794043731689453e-05), (19, 6.524508666992188e-05), (20, 6.92929458618164e-05)],
                          [(0, 0.00015290451049804687), (1, 0.0002037772979736328), (2, 0.0001188478012084961), (
                              3, 0.0001140562744140625), (4, 0.00010555089569091796), (5, 9.751801300048828e-05), (
                              6, 8.783932495117187e-05), (7, 8.818844604492188e-05), (8, 9.206346130371093e-05), (
                              9, 8.731367492675782e-05), (10, 8.400027465820313e-05), (11, 8.275271606445313e-05), (
                              12, 7.893376159667968e-05), (13, 8.0618896484375e-05), (14, 8.042945098876953e-05), (
                              15, 7.869074249267578e-05), (16, 7.631855010986328e-05), (17, 7.100086975097657e-05), (
                              18, 7.232896423339844e-05), (19, 6.764795684814453e-05), (20, 6.713542175292969e-05)],
                          [(0, 0.00015257269287109374), (1, 0.0002211698760986328), (2, 0.00012495555114746095), (
                              3, 0.00011262697601318359), (4, 0.00010537152099609375), (5, 0.00010087232971191407), (
                              6, 9.650676727294922e-05), (7, 9.52246551513672e-05), (8, 9.284107208251953e-05), (
                              9, 8.851715850830078e-05), (10, 8.575824737548829e-05), (11, 8.662493896484375e-05), (
                              12, 8.948761749267579e-05), (13, 9.083307647705078e-05), (14, 8.665264892578124e-05), (
                              15, 8.796790313720704e-05), (16, 8.502159118652344e-05), (17, 9.150987243652344e-05), (
                              18, 8.473251342773437e-05), (19, 8.461943054199219e-05), (20, 8.221849822998047e-05)],
                          [(0, 0.0001516129150390625), (1, 0.00020276853942871094), (2, 0.00012938319396972657), (
                              3, 0.00011794012451171875), (4, 0.00010540846252441406), (5, 9.6427001953125e-05), (
                              6, 9.122502136230469e-05), (7, 9.332180786132812e-05), (8, 8.42638931274414e-05), (
                              9, 8.38744888305664e-05), (10, 8.283631896972656e-05), (11, 7.921194458007812e-05), (
                              12, 7.698268127441407e-05), (13, 7.249375152587891e-05), (14, 7.430821228027343e-05), (
                              15, 6.92708969116211e-05), (16, 6.785153198242187e-05), (17, 6.63576889038086e-05), (
                              18, 6.453402709960937e-05), (19, 6.250899505615235e-05), (20, 6.115933990478516e-05)]
                          ]
    accuracy = [[(0, 0.070828), (1, 0.522127), (2, 0.503726), (3, 0.534692), (4, 0.538733), (5, 0.561427), (6, 0.584603), (7, 0.608663), (8, 0.610069), (9, 0.599564), (10, 0.58606), (11, 0.600094), (12, 0.598886), (13, 0.599948), (14, 0.604556), (15, 0.598373), (16, 0.598303), (17, 0.597808), (18, 0.60565), (19, 0.603469), (20, 0.59789)],
                [(0, 0.05), (1, 0.500702), (2, 0.504728), (3, 0.503249), (4, 0.505176), (5, 0.580199), (6, 0.590462),
                 (7, 0.561643), (8, 0.593632), (9, 0.610387), (10, 0.613743), (11, 0.592041), (12, 0.620285),
                 (13, 0.598496), (14, 0.596515), (15, 0.609503), (16, 0.609543), (17, 0.596464), (18, 0.606669),
                 (19, 0.613843), (20, 0.60812)],
                [(0, 0.068526), (1, 0.5), (2, 0.501524), (3, 0.503876), (4, 0.53552), (5, 0.549452), (6, 0.540052),
                 (7, 0.558721), (8, 0.55499), (9, 0.553072), (10, 0.596462), (11, 0.595354), (12, 0.574987),
                 (13, 0.59359), (14, 0.596722), (15, 0.587761), (16, 0.593619), (17, 0.597574), (18, 0.59285),
                 (19, 0.592524), (20, 0.591116)],
                [(0, 0.086007), (1, 0.500477), (2, 0.501292), (3, 0.503788), (4, 0.535366), (5, 0.553556),
                 (6, 0.560921), (7, 0.574817), (8, 0.613393), (9, 0.601793), (10, 0.60357), (11, 0.603254),
                 (12, 0.60951), (13, 0.597312), (14, 0.604376), (15, 0.598331), (16, 0.597575), (17, 0.590896),
                 (18, 0.594271), (19, 0.597735), (20, 0.598093)]
               ]

    loss_centralized_comparison = comp_loss
    accuracy_comparison = comp_acc
    loss_list = []
    acc_list = []
    loss_comparison_list = []
    acc_comparison_list = []

    for single_loss in losses_centralized:
        temp_loss = []
        for index, val in enumerate(single_loss):
            if index == 0:
                continue
            temp_loss.append(val[1])
        loss_list.append(temp_loss)

    for single_acc in accuracy:
        temp_acc = []
        for index, val in enumerate(single_acc):
            if index == 0:
                continue
            temp_acc.append(val[1])
        acc_list.append(temp_acc)

    for comp_single_loss in loss_centralized_comparison:
        comp_temp_loss = []
        for index, val in enumerate(comp_single_loss):
            comp_temp_loss.append(val)
        loss_comparison_list.append(comp_temp_loss)

    for comp_single_acc in accuracy_comparison:
        comp_temp_acc = []
        for index, val in enumerate(comp_single_acc):
            comp_temp_acc.append(val)
        acc_comparison_list.append(comp_temp_acc)

    x = range(len(loss_list[0]))
    # plot
    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 30}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    # using rc function
    plt.rc('font', **font)
    category_colors_1 = plt.colormaps['tab20'](np.linspace(0.05, 0.95, len(x)))
    for i, (loss, color) in enumerate(zip(loss_list, category_colors_1)):
        line1 = ax.plot(x, loss, c=color, ls="solid", marker='v', label='Loss', markersize=10, linewidth=5)
    if plot_comp:
        category_colors_3 = plt.colormaps['tab20'](np.linspace(0.05, 0.95, len(x)))
        for i, (loss, color) in enumerate(zip(loss_comparison_list, category_colors_3)):
            line2 = ax.plot(x, loss, c=color, ls="solid", marker='^', label='Loss', markersize=10, linewidth=5)

    if plot_acc:
        ax2 = ax.twinx()
        category_colors_2 = plt.colormaps['tab20b_r'](np.linspace(0.05, 0.95, len(x)))
        for i, (acc, color) in enumerate(zip(acc_list, category_colors_2)):
            line3 = ax2.plot(x, acc, c=color, ls="solid", marker='p', label='Accuracy', markersize=10, linewidth=5)
        if plot_comp:
            category_colors_4 = plt.colormaps['tab20b_r'](np.linspace(0.05, 0.95, len(x)))
            for i, (acc, color) in enumerate(zip(acc_comparison_list, category_colors_4)):
                line4 = ax2.plot(x, acc, c=color, ls="solid", marker='h', label='Accuracy', markersize=10, linewidth=5)
            ax2.set_ylabel('Accuracy', **font)
    ax.set_ylabel('Loss value', **font)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax.set_ylim([-0.0005, 0.013])
    # ax.set_yticks([0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012])
    ax.set_xlabel('Rounds', **font)

    ax.set_xticks(x, fontsize=33, weight='bold')
    # ax2.tick_params(axis='y', labelcolor='midnightblue')
    # ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.set_yticks(fontsize=33, weight='bold', fontname='Helvetica Neue')
    # ax2.set_yticks(fontsize=33, weight='bold', fontname='Helvetica Neue')
    if plot_acc:
        ax.legend(line1 + line2 + line3 + line4, ['Loss', 'Compare loss', 'Accuracy', 'Compare accuracy'], loc=7, fontsize=23, ncol=2, framealpha=0.3)
        if plot_comp:
            ax.legend(line1 + line3 , ['Loss', 'Accuracy'], loc=7, fontsize=23, ncol=2, framealpha=0.3)
    else:
        if plot_comp:
            ax.legend(line1 + line2, ['Loss', 'Compare loss'], loc=7, fontsize=23, ncol=2, framealpha=0.3)
        else:
            ax.legend(line1, ['Loss'], loc=7, fontsize=23, ncol=2, framealpha=0.3)

    # plt.savefig("plots/loss_plot.pdf", bbox_inches='tight')
    plt.show()


def read_log_file(path):
    file_regular_expression = "FL_dynamic_clustering"
    acc = []
    loss = []
    dir_list = os.listdir(path)
    # print(dir_list)
    for log in dir_list:
        if file_regular_expression in log:
            temp_file_loss = []
            temp_file_acc = []
            with open(path+log) as f:
                f = f.readlines()
            for line in f:
                if "Round" in line and "Loss" in line:
                    # print(line)
                    temp_loss = float(line.split("Loss ")[1].split(",")[0])
                    temp_acc = float(line.split("Accuracy ")[1].split(",")[0])
                    temp_file_loss.append(temp_loss)
                    temp_file_acc.append(temp_acc)
            acc.append(temp_file_acc)
            loss.append(temp_file_loss)
    return loss, acc


def noise_curve_plot():
    accuracy_static = [0.595, 0.764, 0.813, 0.816]
    F1_static = [0.186, 0.496, 0.586, 0.597]

    x = [0, 0.2, 0.4, 0.6]
    # plot
    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 33}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    # using rc function
    plt.rc('font', **font)

    line1 = ax.plot(x, accuracy_static, c='salmon', ls="solid", marker='v', label='Static Accuracy', markersize=15, linewidth=10)
    line2 = ax.plot(x, F1_static, c='royalblue', ls="solid", marker='^', label='Static F1',
                    markersize=15, linewidth=10)

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel('Accuracy / F1', **font)
    ax.set_xlabel('Noise size', **font)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.legend(line1 + line2, ['Static accuracy', 'Static F1'], loc=4, fontsize=33, ncol=1, framealpha=0.3)

    plt.savefig("plots/noise_plot_1.pdf", bbox_inches='tight')
    plt.show()

    # second plot
    accuracy_dynamic = [0.585, 0.771, 0.793, 0.812]
    F1_dynamic = [0.174, 0.521, 0.559, 0.593]

    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 33}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    # using rc function
    plt.rc('font', **font)

    line3 = ax.plot(x, accuracy_dynamic, c='forestgreen', ls="solid", marker='v', label='Dynamic Accuracy',
                    markersize=15, linewidth=10)
    line4 = ax.plot(x, F1_dynamic, c='tan', ls="solid", marker='^', label='Dynamic F1',
                    markersize=15, linewidth=10)

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    # ax.set_ylabel('Accuracy / F1', **font)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel('Noise size', **font)
    # ax.yaxis.set_visible(False)
    ax.legend(line3 + line4, ['Dynamic accuracy', 'Dynamic F1'], loc=4,
              fontsize=33, ncol=1, framealpha=0.3)

    plt.savefig("plots/noise_plot_2.pdf", bbox_inches='tight')
    plt.show()
    pass


def clients_curve_plot():
    accuracy_static = [0.765, 0.802, 0.813, 0.79, 0.805, 0.788, 0.813]
    F1_static = [0.494, 0.574, 0.586, 0.541, 0.576, 0.545, 0.585]

    x = [10, 15, 20, 25, 30, 50, 100]
    # plot
    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 33}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(15)
    # using rc function
    plt.rc('font', **font)

    line1 = ax.plot(x, accuracy_static, c='salmon', ls="solid", marker='v', label='Static Accuracy', markersize=15, linewidth=10)
    line2 = ax.plot(x, F1_static, c='royalblue', ls="solid", marker='^', label='Static F1',
                    markersize=15, linewidth=10)

    ax.set_yticks([0.4, 0.6, 0.8])
    ax.set_ylim([0.3, 0.9])
    ax.set_ylabel('Accuracy / F1', **font)
    # ax.set_xlabel('Number of clients', **font)
    ax.set_xticks([10, 15, 20, 25, 30, 50, 100])
    ax.legend(line1 + line2, ['Static accuracy', 'Static F1'], loc=4, fontsize=30, ncol=1, framealpha=0.3)

    plt.savefig("plots/clients_plot_1.pdf", bbox_inches='tight')
    plt.show()

    # second plot
    accuracy_dynamic = [0.707, 0.807, 0.793, 0.796, 0.804, 0.777, 0.811]
    F1_dynamic = [0.38, 0.591, 0.559, 0.551, 0.575, 0.519, 0.581]

    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 33}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(15)
    # using rc function
    plt.rc('font', **font)

    line3 = ax.plot(x, accuracy_dynamic, c='forestgreen', ls="solid", marker='v', label='Dynamic Accuracy',
                    markersize=15, linewidth=10)
    line4 = ax.plot(x, F1_dynamic, c='tan', ls="solid", marker='^', label='Dynamic F1',
                    markersize=15, linewidth=10)

    ax.set_yticks([0.4, 0.6, 0.8])
    ax.set_ylim([0.3, 0.9])
    ax.set_ylabel('Accuracy / F1', **font)
    ax.set_xlabel('Number of clients', **font)
    ax.set_xticks([10, 15, 20, 25, 30, 50, 100])
    # ax.yaxis.set_visible(False)
    ax.legend(line3 + line4, ['Dynamic accuracy', 'Dynamic F1'], loc=4,
              fontsize=30, ncol=1, framealpha=0.3)

    plt.savefig("plots/clients_plot_2.pdf", bbox_inches='tight')
    plt.show()
    pass


def clients_curve_combine_plot():
    accuracy_static = [0.765, 0.802, 0.813, 0.79, 0.805, 0.788, 0.813]
    F1_static = [0.494, 0.574, 0.586, 0.541, 0.576, 0.545, 0.585]
    accuracy_dynamic = [0.707, 0.807, 0.793, 0.796, 0.804, 0.777, 0.811]
    F1_dynamic = [0.38, 0.591, 0.559, 0.551, 0.575, 0.519, 0.581]
    x = [10, 15, 20, 25, 30, 50, 100]

    # plot
    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 33}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(15)
    # using rc function
    plt.rc('font', **font)

    line1 = ax.plot(x, accuracy_static, c='salmon', ls="solid", marker='o', label='Static Accuracy', markersize=15,
                    linewidth=10)
    line2 = ax.plot(x, F1_static, c='royalblue', ls="solid", marker='o', label='Static F1',
                    markersize=15, linewidth=10)
    line3 = ax.plot(x, accuracy_dynamic, c='forestgreen', ls="solid", marker='s', label='Dynamic Accuracy',
                    markersize=15, linewidth=10)
    line4 = ax.plot(x, F1_dynamic, c='tan', ls="solid", marker='s', label='Dynamic F1',
                    markersize=15, linewidth=10)

    ax.set_yticks([0.4, 0.6, 0.8])
    ax.set_ylim([0.3, 0.9])
    ax.set_ylabel('Accuracy / F1', **font)
    ax.set_xlabel('Number of clients', **font)
    ax.set_xticks([10, 15, 20, 25, 30, 50, 100])
    # ax.yaxis.set_visible(False)
    ax.legend(line1 + line2 + line3 + line4, ['Static accuracy', 'Static F1', 'Dynamic accuracy', 'Dynamic F1'], loc=4,
              fontsize=25, ncol=2, framealpha=0.3)

    plt.savefig("plots/clients_plot.pdf", bbox_inches='tight')
    plt.show()





if __name__ == "__main__":
    # folder_path = 'log_file/'
    # loss_list, acc_list = read_log_file(folder_path)
    # # print(len(loss_list[0]))
    # # print(loss_list)
    # curve_plot(loss_list, acc_list, plot_acc=False, plot_comp=False)
    #-------------------------------Bar plot for data-----------------------------------------------#
    # data = 2017
    # if data == 2017:
    #     data_dir = '2017_data/'
    #     num_class = 9
    # elif data == 2019:
    #     data_dir = '2019_data/'
    #     num_class = 11
    # pickle_saving_path = 'plots/'
    # num_clients = 20
    # training_data_name = str(num_clients) + '_training.pkl'
    # # load data
    # partition_data_list, testing_data, validation_data = utils.load_data(data_dir, training_data=training_data_name)
    # # show class distribution
    # for index, partition in enumerate(partition_data_list):
    #     (X_train, y_train) = partition
    #     unique, counts = np.unique(y_train, return_counts=True)
    #     print("Client", str(index), "training shape", dict(zip(unique, counts)))
    # # plot
    # if data == 2017:
    #     plot_name = "Partition_" + str(num_clients) + "_2017_ex_class_imbalanced.pdf"
    # elif data == 2019:
    #     plot_name = "Partition_" + str(num_clients) + "_2019_ex_class_imbalanced.pdf"
    # sampling.plot_stacked_bar(partition_data_list, pickle_saving_path, plot_name, number_class=num_class)
    #-------------------------------noise comparsion plot-----------------------------------------------#
    noise_curve_plot()
    # clients_curve_plot()
    # clients_curve_combine_plot()
