import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from graph.graph import Graph


def getEdgeWeight1(embedding):
    """
    :return:cosine similarity
    """
    tmp = np.dot(embedding, embedding.T)
    norm = np.linalg.norm(embedding, 2, axis=1, keepdims=True)
    norm_m = np.dot(norm, norm.T)
    return tmp / norm_m


def getEdgeWeight2(embedding):
    """
    :return: Inner product
    """
    return np.dot(embedding, embedding.T)


def getEdgeWeight3(embedding):
    """
    :return: (||v_i||_2 + ||v_j||_2)/2
    """
    l2norm = np.linalg.norm(embedding, 2, axis=1, keepdims=True)
    l2norm = np.repeat(l2norm, len(l2norm), axis=1)
    return (l2norm + l2norm.T) / 2



def getEdgeWeight4(embedding):
    """
    :return: 1/(||v_i||_2-||v_j||_2)
    """
    edgeWeight = np.zeros([len(embedding), len(embedding)], dtype=np.float32)
    for i in range(len(embedding)):
        for j in range(len(embedding)):
            if j <= i:
                continue
            else:
                tmp = 1.0 / np.linalg.norm(embedding[i, :] - embedding[j, :], 2)
                edgeWeight[i, j] = tmp
                edgeWeight[j, i] = tmp

    return edgeWeight



def get_reconstructed_adj(emb, get_edge_weight):
    adj_mtx_r = get_edge_weight(emb)
    return adj_mtx_r


def computePrecisionCurve(predicted_edge_list, test_edge, max_k=-1):
    if max_k == -1:
        max_k = len(predicted_edge_list)
    else:
        max_k = min(max_k, len(predicted_edge_list))

    sorted_edges = sorted(predicted_edge_list, key=lambda x: x[2], reverse=True)

    precision_scores = []
    delta_factors = []
    correct_edge = 0
    for i in range(max_k):
        if sorted_edges[i][1] in test_edge[sorted_edges[i][0]]:
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        precision_scores.append(1.0 * correct_edge / (i + 1))
    return precision_scores, delta_factors


def computeMAP(predicted_edge_list, test_edge, max_k=-1):
    node_num = len(test_edge)
    node_edges = []
    for i in range(node_num):
        node_edges.append([])
    for (st, ed, w) in predicted_edge_list:
        node_edges[st].append((st, ed, w))
    node_AP = [0.0] * node_num
    count = 0
    for i in range(node_num):
        if len(test_edge[i]) == 0:
            node_AP[i] = 0
            # count += 1
            continue
        count += 1
        precision_scores, delta_factors = computePrecisionCurve(node_edges[i], test_edge, max_k)
        precision_rectified = [p * d for p, d in zip(precision_scores, delta_factors)]
        if (sum(delta_factors) == 0):
            node_AP[i] = 0
        else:
            node_AP[i] = float(sum(precision_rectified) / sum(delta_factors))
            # print(node_AP[i])
    return sum(node_AP) / count


def getEdgeListFromAdjMtx(adj, threshold=0.0, is_undirected=True, edge_pairs=None):
    result = []
    node_num = adj.shape[0]
    if edge_pairs:
        for (st, ed) in edge_pairs:
            if adj[st, ed] >= threshold:
                result.append((st, ed, adj[st, ed]))
    else:
        for i in range(node_num):
            for j in range(node_num):
                if (j == i):
                    continue
                if (is_undirected and i >= j):
                    continue
                if adj[i, j] > threshold and i < j:
                    result.append((i, j, adj[i, j]))
    return result


def evaluateStaticLinkPrediction(g, embedding_file, get_edge_weight, sample_ratio, sample_seed=None):
    nodes_num = g.nodes_num
    is_symmetirc = g.is_symmetirc

    print('sample test node')
    node_set = g.nodes_set
    np.random.seed(sample_seed)
    node_set_id = []
    if np.abs(sample_ratio - 1.0) <= 0.000001:
        test_node_set = node_set
    else:
        np.random.shuffle(node_set)
        test_node_set = node_set[: int(nodes_num * sample_ratio)]

    node2sample_id = dict()
    for sample_id, node in enumerate(test_node_set):
        node_set_id.append(g.nodes_ids[node])
        node2sample_id[node] = sample_id

    print('load test edges')
    test_edge = []
    for i in range(len(test_node_set)):
        test_edge.append([])
    for node_id, node in enumerate(test_node_set):
        for in_node in g.test_nodes_adj_edges_set[node]:
            if in_node in node2sample_id.keys():
                test_edge[node_id].append(node2sample_id[in_node])

    emb = list()
    with open(embedding_file) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            emb.append(row)
    emb = np.array(emb, dtype=np.float32)
    emb = emb[node_set_id, :]


    print('reconstructed adj')
    eval_edge_pairs = None

    estimated_adj = get_reconstructed_adj(emb, get_edge_weight)
    predicted_edge_list = getEdgeListFromAdjMtx(
        estimated_adj,
        is_undirected=is_symmetirc,
        edge_pairs=eval_edge_pairs
    )

    filtered_edge_list = [e for e in predicted_edge_list if e[1] not in g.nodes_adj_edges_set[g.nodes_set[e[0]]]]

    print('compute MAP')
    MAP = computeMAP(filtered_edge_list, test_edge, max_k=-1)

    return MAP



