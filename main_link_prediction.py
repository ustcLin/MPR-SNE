from graph.graph import Graph
from model.mprsne import MPRSNE
from link_prediction.link_prediction import *
from data_split.data_split import *
from config.config import Config
import csv
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

if __name__ == '__main__':
    saveFile = 'blog_mprsne_test_link_prediction'
    config = Config()

    # get train_nodes0.6.csv and test_nodes0.6.csv
    # node_label_split(r'datasets\BlogCatalog-dataset\nodes.csv', config.train_node_size, config.train_node_size,
    #                  r'datasets\BlogCatalog-dataset')

    # training emb
    g = Graph(r'datasets\BlogCatalog-dataset', is_link_predict=True)
    model = MPRSNE(g, saveFile=saveFile)
    model.train(10000, new_training=True)

    # save emb
    emb = model.get_emb()
    with open('{}.csv'.format(saveFile), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in emb:
            csvwriter.writerow(row)

    # test link prediction
    MAP = evaluateStaticLinkPrediction(g,
                                       '{}.csv'.format(saveFile),
                                       getEdgeWeight3,
                                       1.0,
                                       0)
    print('{}_edge_weight3'.format(saveFile))
    print('MAP: ', MAP)
