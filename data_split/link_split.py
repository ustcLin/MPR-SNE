import csv
import numpy as np
from config.config import Config

def link_split(edges_file, link_train_size, train_edge_file, test_edge_file, is_symmetric=True):
    edge_list = []
    with open(edges_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if len(row) != 0:
                edge_list.append(row)

    np.random.shuffle(edge_list)
    train_end = int(len(edge_list) * link_train_size)
    train_edge = edge_list[:train_end]
    test_edge = edge_list[train_end:]

    train_nodes_adj_edges_set = {}

    for row in train_edge:
        out_node = row[0]
        in_node = row[1]
        if out_node not in train_nodes_adj_edges_set.keys():
            train_nodes_adj_edges_set[out_node] = []

        train_nodes_adj_edges_set[out_node].append(in_node)

        if is_symmetric:
            if in_node not in train_nodes_adj_edges_set.keys():
                train_nodes_adj_edges_set[in_node] = []
            train_nodes_adj_edges_set[in_node].append(out_node)

    filter_test_edge = []
    for row in test_edge:
        out_node = row[0]
        in_node = row[1]
        is_filter = 0
        if out_node not in train_nodes_adj_edges_set.keys() or (is_symmetric and in_node not in train_nodes_adj_edges_set.keys()):
            train_edge.append([out_node, in_node])
            is_filter = 1
            train_nodes_adj_edges_set[out_node] = [in_node]
            if is_symmetric:
                train_nodes_adj_edges_set[in_node] = [out_node]

        if is_filter == 0:
            filter_test_edge.append(row)

    test_edge = filter_test_edge

    with open(train_edge_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in train_edge:
            csvwriter.writerow(row)

    with open(test_edge_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in test_edge:
            csvwriter.writerow(row)



if __name__ == '__main__':
    config = Config()
    link_split(r'..\datasets\Flickr-dataset\edges.csv',
               config.train_edge_size,
               r'..\datasets\Flickr-dataset\train_edges.csv',
               r'..\datasets\Flickr-dataset\test_edges.csv')
