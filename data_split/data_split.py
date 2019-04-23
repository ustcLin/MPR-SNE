import csv
import numpy as np
import os

from config.config import Config


def label_split(csv_file, train_size, validation_size, test_size, output_folder_name):
    if train_size + validation_size + test_size != 1:
        print('split size error!')
        return []

    label_dict = dict()
    with open(csv_file) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) != 0:
                if row[1] not in label_dict.keys():
                    label_dict[row[1]] = [row[0]]
                else:
                    label_dict[row[1]].append(row[0])

    train_dict = dict()
    validation_dict = dict()
    test_dict = dict()
    for label in label_dict.keys():
        np.random.shuffle(label_dict[label])
        label_data_num = len(label_dict[label])
        train_end = int(label_data_num * train_size) + 1
        validation_end = int(label_data_num * (train_size + validation_size)) + 1
        train_dict[label] = label_dict[label][:train_end]
        validation_dict[label] = label_dict[label][train_end:validation_end]
        test_dict[label] = label_dict[label][validation_end:]

    train_file_name = os.path.join(output_folder_name, 'train_labels{}.csv'.format(train_size))
    validation_file_name = os.path.join(output_folder_name, 'validation_labels{}.csv'.format(train_size))
    test_file_name = os.path.join(output_folder_name, 'test_labels{}.csv'.format(train_size))

    with open(train_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for label in train_dict.keys():
            if len(train_dict[label]) == 0:
                continue
            for node in train_dict[label]:
                csvwriter.writerow([node, label])

    with open(validation_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for label in validation_dict.keys():
            if len(validation_dict[label]) == 0:
                continue
            for node in validation_dict[label]:
                csvwriter.writerow([node, label])

    with open(test_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for label in test_dict.keys():
            if len(test_dict[label]) == 0:
                continue
            for node in test_dict[label]:
                csvwriter.writerow([node, label])

    return None


def node_label_split(nodes_file, train_size, test_size, output_folder_name):
    if train_size + test_size != 1:
        print('split size error!')
        return []

    nodes_set = []
    with open(nodes_file) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) != 0:
                nodes_set.append(row)

    for i in range(10):
        np.random.shuffle(nodes_set)

    train_nodes_end = int(len(nodes_set) * train_size)
    train_nodes_set = nodes_set[:train_nodes_end]
    test_nodes_set = nodes_set[train_nodes_end:]

    train_nodes_file_name = os.path.join(output_folder_name, 'train_nodes{}.csv'.format(train_size))
    test_nodes_file_name = os.path.join(output_folder_name, 'test_nodes{}.csv'.format(train_size))

    with open(train_nodes_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for node in train_nodes_set:
            csvwriter.writerow(node)

    with open(test_nodes_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for node in test_nodes_set:
            csvwriter.writerow(node)

    return None


def labels_split(label_info_file, train_size, validate_size, test_size, output_folder_name):
    if train_size + validate_size + test_size != 1:
        print('split size error!')
        return []

    labels_set = []
    with open(label_info_file) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) != 0:
                labels_set.append(row)

    for i in range(10):
        np.random.shuffle(labels_set)

    train_labels_end = int(len(labels_set) * train_size)
    validate_labels_end = int(len(labels_set) * (train_size + validate_size))
    train_labels_set = labels_set[:train_labels_end]
    validate_labels_set = labels_set[train_labels_end:validate_labels_end]
    test_labels_set = labels_set[validate_labels_end:]

    train_labels_file_name = os.path.join(output_folder_name, 'train_labels{}.csv'.format(train_size))
    validate_labels_file_name = os.path.join(output_folder_name, 'validate_labels{}.csv'.format(train_size))
    test_labels_file_name = os.path.join(output_folder_name, 'test_labels{}.csv'.format(train_size))

    with open(train_labels_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for label in train_labels_set:
            csvwriter.writerow(label)

    with open(validate_labels_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for label in validate_labels_set:
            csvwriter.writerow(label)

    with open(test_labels_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for label in test_labels_set:
            csvwriter.writerow(label)

    return None


def labels_top_split(data_file, top_n, validate_rate, output_folder_name):
    labels_file = os.path.join(data_file, 'labels.csv')
    labels_info_file = os.path.join(data_file, 'labels_info.csv')

    labels_nodes = dict()
    labels_id = list()

    with open(labels_info_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if len(row) != 0:
                labels_nodes[row[0]] = list()

    with open(labels_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if len(row) != 0:
                node = row[0]
                label = row[1]
                labels_nodes[label].append(node)

    labels_num = list()
    labels_id = list(labels_nodes.keys())
    for label in labels_id:
        labels_num.append(len(labels_nodes[label]))

    top_n_index = np.argsort(labels_num)[-top_n:]
    clf_label_list = list()
    for index in top_n_index:
        clf_label_list.append(labels_id[index])

    emb_label_list = list()
    for label in labels_id:
        if label not in clf_label_list:
            emb_label_list.append(label)

    np.random.shuffle(emb_label_list)
    validate_label_num = int(len(emb_label_list) * validate_rate)
    validate_label_list = emb_label_list[:validate_label_num]
    emb_label_list = emb_label_list[validate_label_num:]

    train_labels_file_name = os.path.join(output_folder_name, 'train_labels.csv')
    validate_labels_file_name = os.path.join(output_folder_name, 'validate_labels.csv')
    test_labels_file_name = os.path.join(output_folder_name, 'test_labels.csv')

    with open(train_labels_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for label in emb_label_list:
            csvwriter.writerow([label])

    with open(validate_labels_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for label in validate_label_list:
            csvwriter.writerow([label])

    with open(test_labels_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for label in clf_label_list:
            csvwriter.writerow([label])

    return None


if __name__ == '__main__':
    config = Config()
    train_label_size = config.train_label_size
    validate_label_size = config.validate_label_size
    test_label_size = config.test_label_size

    # label_split(r'..\datasets\BlogCatalog-dataset\labels.csv', 0.6, 0.1, 0.3,
    #             r'..\datasets\BlogCatalog-dataset')

    node_label_split(r'..\datasets\BlogCatalog-dataset\nodes.csv', train_label_size, test_label_size,
                     r'..\datasets\BlogCatalog-dataset')

    # labels_split(r'..\datasets\Flickr-dataset\labels_info.csv', train_label_size,
    #              validate_label_size,
    #              test_label_size,
    #              r'..\datasets\Flickr-dataset')

    labels_top_split(r'..\datasets\BlogCatalog-dataset',
                     5,
                     0.1,
                     r'..\datasets\BlogCatalog-dataset')
