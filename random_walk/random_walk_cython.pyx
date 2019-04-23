from libc.stdlib cimport rand, srand
from libc.time cimport time
import numpy as np

def random_walk(g, long walk_length):
    cdef:
        long nodes_num = g.nodes_num
        long rand_num
        long edges_num


    walk = np.zeros([nodes_num, walk_length], dtype=int)
    srand(<unsigned> time(NULL))

    for i in range(nodes_num):
        first_node = g.nodes_set[i]
        walk[i][0] = g.nodes_ids[first_node]
        current_node = first_node
        for j in range(walk_length - 1):
            edges_num = len(g.nodes_adj_edges_set[current_node])
            rand_num = rand() % edges_num
            current_node = g.nodes_adj_edges_set[current_node][rand_num]
            walk[i][j + 1] = g.nodes_ids[current_node]

    return walk

