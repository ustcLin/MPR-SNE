from graph.graph import Graph
from model.mprsne import MPRSNE
from node_classification.node_classification import *
from data_split.data_split import *
from config.config import Config
import csv
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"


if __name__ == '__main__':
    saveFile = 'blog_mprsne_test_node_classification'
    config = Config()

    # get train_nodes0.6.csv and test_nodes0.6.csv
    # node_label_split(r'datasets\BlogCatalog-dataset\nodes.csv', config.train_node_size, config.train_node_size,
    #                  r'datasets\BlogCatalog-dataset')


    # training emb
    g = Graph(r'datasets\BlogCatalog-dataset', is_link_predict=False)
    model = MPRSNE(g, saveFile=saveFile, init_emb_file=r'datasets\BlogCatalog-dataset\emb\deepwalk_emb.csv')
    model.train(10000, new_training=True)

    # save emb
    emb = model.get_emb()
    with open('{}.csv'.format(saveFile), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in emb:
            csvwriter.writerow(row)

    # test node classification
    test_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    performence_macro = []
    performence_micro = []
    for i in range(len(test_rate)):
        print(test_rate[i])
        tmp_rate = test_rate[i]
        acc_list = []
        macro_f1_list = []
        micro_f1_list = []
        for j in range(10):
            acc, macro_f1, micro_f1 = multi_label_node_classification_train_test_split(g,
                                                                                       '{}.csv'.format(saveFile),
                                                                                       tmp_rate)
            print('acc {}, macro {}, micro {}'.format(acc, macro_f1, micro_f1))
            # acc_list.append(acc)
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        performence_macro.append(np.mean(macro_f1_list))
        performence_micro.append(np.mean(micro_f1_list))

    print(saveFile)
    print('macro')
    print(performence_macro)
    print('micro')
    print(performence_micro)
