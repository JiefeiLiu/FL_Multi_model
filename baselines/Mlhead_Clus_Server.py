# This is my code to train n model and saved them

import random
import os
import tensorflow as tf
import numpy as np
import shutil
import tempfile

from tensorflow.python import pywrap_tensorflow

from mh_constants import VARIABLE_PARAMS, MODEL_PARAMS
from mlhead_utilfuncs import input_fn
from kmean_model import KmeanModel

from outlkm_algor import OutlierKmeansAlgor
from utils.args import parse_args


class Mlhead_Clus_Server:

    def __init__(self, client_model, dataset, model, num_clusters, num_clients, clus_seed=0):

        #        if num_clusters < 2:
        #            raise Exception("Sorry, cluster number must be 2 or more")
        if num_clusters > 10:
            raise Exception("Sorry, cluster number must less than 10")
        if num_clusters != -1:
            self.regul_term = 0.1
            self.model = client_model.get_params()  # global model of the server.
            self.selected_clients = []  # this variable keeps random, unclustered clients
            self.set_model_path(dataset, model)
            self._x_dimensions = self.get_model_x_dimensions(dataset, model)
            self._variable = self.get_model_variable(dataset, model)
            self._num_clusters = num_clusters
            self._learned = None
            self._clusterModel = KmeanModel(num_clients, self._x_dimensions, \
                                            self._num_clusters, clus_seed)

            self._outlkmalgor = OutlierKmeansAlgor(num_clients, self._x_dimensions,
                                                   num_clusters, max_iter=2, seed=clus_seed,
                                                   threshold_dis=self.regul_term)
        """
        cluster_membership is a list of cluster dictionary,
        each contains {'member':list of clients, 
            'center': a center vector, 'attention': a attention vector,
            'loss': mean valdation loss of each client of this cluster}
        """
        self._cluster_membership = list()
        for _ in range(num_clusters):
            self._cluster_membership.append({"member": [], "center": None, "attention": [], "loss": None})

    @property
    def path(self):
        return self._path

    @property
    def x_dimensions(self):
        return self._x_dimensions

    @property
    def variable(self):
        return self._variable

    @property
    def selected(self):
        return [c for c in self.selected_clients]

    @property
    def clusters(self):
        return [c for c in self._clusters_membership]

    def select_clients(self, my_round, possible_clients):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).
        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = len(possible_clients)
        np.random.seed(my_round + 20)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self):
        """Trains self.model on given clients.

        """
        clients = self.selected_clients

        tot_clients = len(clients)
        done_idx = 0
        for counter, c in enumerate(clients, 1):
            """
            Note: this is a trick. and it's equal to clear the session of client
            and make sure the graph has been re-initialized.
            """
            c_file = self.get_chkpfile('write_%s.ckpt' % c.id)

            """
            I think this one will run faster than training
            since now I have already initial the value with normal
            distribution but no training at first round
            """

            if os.path.exists(c_file):
                os.remove(c_file)
            c.save_model(c_file)

            done_percentil = float(done_idx + 1) * 25
            if self.get_percentil(counter, tot_clients) >= done_percentil:
                done_idx += 1
                print("%g%% clients has done" % done_percentil)

    def get_percentil(self, counter, len):
        fraction = float((counter + 1) / len)
        return fraction * 100

    def set_model_path(self, dataset, model):
        self._path = os.path.join('/scratch/leaf/ckpt_runtime', dataset, model)

    #     	if not os.path.exists(self._path):
    #         	os.makedirs(self._path)

    def get_model_x_dimensions(self, dataset, model):
        key = "%s.%s" % (dataset, model)
        d = MODEL_PARAMS[key]
        return d[0] * d[1]

    def get_model_variable(self, dataset, model):
        key = "%s.%s" % (dataset, model)
        v = VARIABLE_PARAMS[key]
        return v

    def get_chkpfile(self, id_ckpt):
        return os.path.join(self._path, id_ckpt)

    def train_iteation(self, data):
        train_data = lambda: input_fn(data)
        self._kmeans.train(train_data)
        cluster_centers = self._kmeans.cluster_centers()
        # print("cluster centers:", cluster_centers)
        score = self._kmeans.score(train_data)
        return score

    def get_init_point_data(self):
        # points = np.random.normal(loc=0.5, scale=0.5, size= (len(self.selected), self._x_dimensions))
        points = np.random.uniform(-0.01, 0.01, (len(self.selected), self._x_dimensions))
        c_dict = {}
        for x, client in enumerate(self.selected):
            c_dict[client.id] = points[x]

        return c_dict

    def run_clustering(self, data):
        labels = self._clusterModel.assign_clusters([data[k] for k in data])
        return self.eval_clustermembership(labels)

    def train_kmeans(self, prev_score, data):
        """We are using pre-made tensorflow estimators to
            train and predict.

           Args:
               prev_score: a sum of the distance between each sample
               to their nearest center
               data: list of weights of user model
           Return:
               updated score

        """

        seed = np.random.randint(5667799881, size=1)[0]
        temp_name = next(tempfile._get_candidate_names())
        temp_modeldir = os.path.join("./", temp_name)
        if not os.path.exists(temp_modeldir):
            os.makedirs(temp_modeldir)
        self._kmeans = tf.contrib.factorization.KMeansClustering(
            model_dir=temp_modeldir,
            random_seed=seed,
            num_clusters=self._num_clusters, use_mini_batch=False)

        # composed of weights of every client model

        score = self.train_iteation(data)

        # evaluate the samples to compute the distance between
        # each sample and each center, forming a matrix have each
        # row for each sample, and the column is distance to each
        # center.
        y = self._kmeans.transform(lambda: input_fn(data))
        point_distance = list(y)
        self._learned = np.argmin(point_distance, 1)
        # removing not used files of this model
        # shutil.rmtree(temp_modeldir, ignore_errors=True)
        return None, score

    def eval_clustermembership(self, labels):
        """Transfrom the input data,
        get the min distance of each point to cluster centers,
        then return the index of cluster center whose distance for a sample
        is the mininum

        return a list of (num_clients, clients)
        """

        for _, cluster in enumerate(self._cluster_membership):
            cluster["member"].clear()

        for x in range(len(labels)):
            grp_id = labels[x]
            self._cluster_membership[grp_id]["member"].append(self.selected_clients[x])
        clus = [(len(cluster["member"]), cluster["member"]) for cl_id, cluster in enumerate(self._cluster_membership)]
        for c in clus:
            # I wanna to set first flag to true if any
            # cluster is only = 0
            if c[0] == 0:
                self._clusterModel.first = True
                break;
        return clus

    def outlier_clustering(self, data):
        if not self._outlkmalgor.finalized:
            self._outlkmalgor.fit(np.array([data[k] for k in data]))

        return self.eval_clustermembership(self._outlkmalgor.labels)

    def is_unbalanced_clus(self, cluster_indices):
        lst_num_data = np.array([c[0] for c in cluster_indices])
        # check numpy array if any elements <= 1
        num_count = np.where(lst_num_data <= 1)
        if len(num_count[0]) == 0:
            return False
        else:
            return True