from config.config import Config
from get_nodes_table import get_nodes_table

import os
import csv
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class Graph(object):
    def __init__(self, directory, is_symmetric=True, init_table=True, is_link_predict=False, is_test_missing=False, missing_p=0.0):
        """
        a graph object.
        : param directory: data directory
        """
        self.directory = directory
        self.is_symmetirc = is_symmetric
        self.config = Config()
        self.is_test_missing = is_test_missing
        self.missing_p = missing_p
        if 'Flickr' in directory:
            self.config.train_node_size = 0.05
        if 'BlogCatalog' in directory:
            self.config.train_node_size = 0.6

        self.nodes_table_size = 1 * 10 ** 8
        self.nodes_labels_table_size = 1 * 10 ** 7

        self.nodes_file = os.path.join(self.directory, 'nodes.csv')
        if is_link_predict:
            self.edges_file = os.path.join(self.directory, 'train_edges.csv')
            self.test_edges_file = os.path.join(self.directory, 'test_edges.csv')

        else:
            self.edges_file = os.path.join(self.directory, 'edges.csv')
        self.labels_info_file = os.path.join(self.directory, 'labels_info.csv')
        self.train_labels_file = os.path.join(self.directory, 'train_labels.csv')
        self.validate_labels_file = os.path.join(self.directory, 'validate_labels.csv')
        self.test_labels_file = os.path.join(self.directory, 'test_labels.csv')
        self.labels_file = os.path.join(self.directory, 'labels.csv')
        self.train_nodes_file = os.path.join(self.directory, 'train_nodes{}.csv'.format(self.config.train_node_size))
        self.test_nodes_file = os.path.join(self.directory, 'test_nodes{}.csv'.format(self.config.train_node_size))

        self.nodes_set = self.get_nodes_set(self.nodes_file)
        self.nodes_num = len(self.nodes_set)
        self.nodes_ids = self.get_nodes_ids()
        self.nodes_adj_edges_set, self.nodes_in_degree, self.nodes_in_set = \
            self.get_nodes_adj_edges_set(self.nodes_set, self.edges_file, is_symmetric)
        if is_link_predict:
            self.test_nodes_adj_edges_set, _, _ = self.get_nodes_adj_edges_set(self.nodes_set, self.test_edges_file,
                                                                               is_symmetric)
        self.nodes_out_degree = self.get_out_degree()
        self.nodes_degree_prob = self.get_nodes_prob(self.nodes_out_degree)
        self.nodes_in_degree_prob = self.get_nodes_prob(self.nodes_in_degree)
        if init_table:
            self.nodes_degree_table = get_nodes_table(self.nodes_degree_prob, self.nodes_table_size)

        self.all_labels_set, self.train_labels_set, self.validate_labels_set, self.test_labels_set = self.get_labels()
        self.labels_num = len(self.train_labels_set)
        self.all_labels_ids, self.train_labels_ids, self.validate_labels_ids, self.test_labels_ids = self.get_labels_ids()
        self.all_nodes_labels, self.emb_nodes_labels, self.emb_validate_nodes_labels, self.clf_nodes_labels, self.train_nodes_labels, self.test_nodes_labels \
            = self.get_nodes_labels()
        # self.nodes_labels_edge = self.get_nodes_labels_edge()
        # self.nodes_labels_count = self.get_nodes_labels_count()
        # self.nodes_labels_prob = self.get_nodes_prob(self.nodes_labels_count)
        # if init_table:
        #     self.nodes_labels_table = get_nodes_table(self.nodes_labels_prob, self.nodes_labels_table_size)

    def get_nodes_set(self, nodes_file):
        nodes_set = list()
        with open(nodes_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) != 0:
                    nodes_set.append(row[0])

        return nodes_set

    def get_nodes_adj_edges_set(self, nodes_set, edges_file, is_symmetric):
        nodes_adj_edges_set = dict()
        nodes_in_set = dict()
        in_degree = dict()
        for node in nodes_set:
            nodes_adj_edges_set[node] = list()
            nodes_in_set[node] = list()
            in_degree[node] = 0

        with open(edges_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) != 0:
                    out_node = row[0]
                    for node in row[1:]:
                        nodes_adj_edges_set[out_node].append(node)
                        nodes_in_set[node].append(out_node)
                        in_degree[node] += 1

                        if is_symmetric:
                            nodes_adj_edges_set[node].append(out_node)
                            nodes_in_set[out_node].append(node)
                            in_degree[out_node] += 1

        return nodes_adj_edges_set, in_degree, nodes_in_set

    def get_out_degree(self):
        out_degree = dict()
        for key in self.nodes_adj_edges_set.keys():
            out_degree[key] = len(self.nodes_adj_edges_set[key])

        return out_degree

    def get_labels(self):
        all_labels_set = list()
        with open(self.labels_info_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) != 0:
                    all_labels_set.append(row[0])

        train_labels_set = list()
        with open(self.train_labels_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) != 0:
                    train_labels_set.append(row[0])

        validate_labels_set = list()
        with open(self.validate_labels_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) != 0:
                    validate_labels_set.append(row[0])

        test_labels_set = list()
        with open(self.test_labels_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) != 0:
                    test_labels_set.append(row[0])

        return all_labels_set, train_labels_set, validate_labels_set, test_labels_set


    def get_nodes_labels(self):
        all_nodes_labels = dict()
        for node in self.nodes_set:
            all_nodes_labels[node] = list()

        with open(self.labels_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) != 0:
                    all_nodes_labels[row[0]].append(row[1])

        emb_nodes_labels = dict()
        emb_validate_nodes_labels = dict()
        clf_nodes_labels = dict()
        for node in self.nodes_set:
            emb_nodes_labels[node] = list()
            emb_validate_nodes_labels[node] = list()
            clf_nodes_labels[node] = list()

        np.random.seed(10)
        for node in self.nodes_set:
            for label in all_nodes_labels[node]:
                if label in self.train_labels_set:
                    if self.is_test_missing is False:
                        emb_nodes_labels[node].append(label)
                    else:
                        tmp_rand = np.random.rand()
                        if tmp_rand > self.missing_p:
                            emb_nodes_labels[node].append(label)

                elif label in self.validate_labels_set:
                    emb_validate_nodes_labels[node].append(label)
                else:
                    clf_nodes_labels[node].append(label)

        train_nodes_set = list()
        test_nodes_set = list()

        with open(self.train_nodes_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) != 0:
                    train_nodes_set.append(row[0])

        with open(self.test_nodes_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) != 0:
                    test_nodes_set.append(row[0])

        train_nodes_labels = dict()
        test_nodes_labels = dict()

        for node in train_nodes_set:
            train_nodes_labels[node] = emb_validate_nodes_labels[node]

        for node in test_nodes_set:
            test_nodes_labels[node] = emb_validate_nodes_labels[node]

        return all_nodes_labels, emb_nodes_labels, emb_validate_nodes_labels, clf_nodes_labels, train_nodes_labels, test_nodes_labels


    def get_nodes_ids(self):
        nodes_ids = dict()
        for i, node in enumerate(self.nodes_set):
            nodes_ids[node] = i
        return nodes_ids

    def get_labels_ids(self):
        all_labels_ids = dict()
        for i, label in enumerate(self.all_labels_set):
            all_labels_ids[label] = i

        train_labels_ids = dict()
        for i, label in enumerate(self.train_labels_set):
            train_labels_ids[label] = i

        validate_labels_ids = dict()
        for i, label in enumerate(self.validate_labels_set):
            validate_labels_ids[label] = i

        test_labels_ids = dict()
        for i, label in enumerate(self.test_labels_set):
            test_labels_ids[label] = i
        return all_labels_ids, train_labels_ids, validate_labels_ids, test_labels_ids

    def get_nodes_prob(self, nodes_count):
        degree_sum = 0
        for node in self.nodes_set:
            if node in nodes_count.keys():
                degree_sum += nodes_count[node] ** 0.75
        nodes_degree_prob = np.zeros(self.nodes_num, dtype=np.float64)
        for node in self.nodes_set:
            if node in nodes_count.keys():
                nodes_degree_prob[self.nodes_ids[node]] = nodes_count[node] ** 0.75 / degree_sum
        return nodes_degree_prob


    def negative_sample(self, node_id_batch, sample_num, table):
        neg_batch = list()
        table_len = len(table)
        for node_id in node_id_batch:
            neg_list = list()
            for i in range(sample_num):
                neg_id = table[np.random.randint(table_len)]
                while node_id == neg_id:
                    neg_id = table[np.random.randint(table_len)]
                neg_list.append(neg_id)
            neg_batch.append(neg_list)
        neg = np.array(neg_batch, dtype=np.int)
        return neg

    def get_train_node_label(self, node_list):
        train_node_label = list()
        for i, node_id in enumerate(node_list):
            label_list = self.emb_nodes_labels[self.nodes_set[node_id]]
            train_node_label.append(label_list)

        MLB = MultiLabelBinarizer(classes=self.train_labels_set)
        one_hot_label = MLB.fit_transform(train_node_label)
        return one_hot_label


if __name__ == '__main__':
    # test
    import datetime

    begin = datetime.datetime.now()
    g = Graph(r'..\datasets\BlogCatalog-dataset', is_link_predict=False)
    labels_degree = []
    for node in g.emb_nodes_labels.keys():
        labels_degree.append(len(g.emb_nodes_labels[node]))
    print(1 - sum(labels_degree) / (len(g.emb_nodes_labels.keys()) * g.labels_num))
    print(g.labels_num)
    print(len(g.emb_nodes_labels.keys()))


