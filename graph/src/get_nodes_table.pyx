
def get_nodes_table(double [:] nodes_prob, long table_size):
    cdef:
        double prob_sum = 0.0
        long node_num = nodes_prob.shape[0]
        long i = 0

    table = [0] * table_size
    for node_id in range(node_num):
        while (i + 1.0) / table_size < prob_sum + nodes_prob[node_id] and i < table_size:
            table[i] = node_id
            i += 1
        prob_sum = prob_sum + nodes_prob[node_id]

    return table
