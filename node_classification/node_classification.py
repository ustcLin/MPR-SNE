import csv
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



def multi_label_node_classification_train_test_split(g, emb_file, train_size):
    emb = list()
    with open(emb_file) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            emb.append(row)
    emb = np.array(emb, dtype=np.float32)


    classes = g.test_labels_set
    nodes_set = g.nodes_set

    x_ = list()
    for node in nodes_set:
        x_.append(emb[g.nodes_ids[node], :])
    x_ = np.array(x_)

    y_ = list()
    for node in nodes_set:
        y_.append(g.clf_nodes_labels[node])
    prepocess_y = MultiLabelBinarizer(classes=classes)
    y_ = prepocess_y.fit_transform(y_)


    train_x, test_x, train_y, test_y = train_test_split(x_, y_, train_size=train_size)


    multi_label_classifier = OneVsRestClassifier(LogisticRegression(C=2), n_jobs=1)
    multi_label_classifier.fit(train_x, train_y)
    pred_y = multi_label_classifier.predict(test_x)
    micro_f1 = f1_score(test_y, pred_y, average='micro')
    macro_f1 = f1_score(test_y, pred_y, average='macro')

    acc = 1 - sum(sum(test_y ^ pred_y)) / (len(test_x) * len(classes))
    return acc, macro_f1, micro_f1


