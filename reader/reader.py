from config.config import Config
from random_walk.random_walk_cython import random_walk

import numpy as np
import datetime


class Reader(object):
    def __init__(self, g):
        self.g = g
        self.config = Config()
        self.nodes_seq_generator = self.get_nodes_seq_generator()
        self.nodes_walk_generator = self.get_nodes_walk_generator()

    def get_nodes_seq_generator(self):
        # walk_times = -1
        # t_start = 1e-3
        while True:
            begin = datetime.datetime.now()
            walk = random_walk(self.g, self.config.walk_length)
            # walk_times += 1
            # t = np.max([5e-5, t_start * (0.95 ** (walk_times / 5.0))])
            t = 1e-3
            end = datetime.datetime.now()
            print('random walk time {}'.format(end - begin))
            np.random.shuffle(walk)
            for seq in walk:
                for i, node in enumerate(seq):
                    ran = np.sqrt(t / self.g.nodes_in_degree_prob[node]) + (t / self.g.nodes_in_degree_prob[node])
                    r = np.random.random()
                    if r > ran:
                        # print(1, t)
                        continue
                    mid_node = node
                    left_context = np.zeros([self.config.context_size], dtype=np.int32) + self.g.nodes_num
                    right_context = np.zeros([self.config.context_size], dtype=np.int32) + self.g.nodes_num
                    j = 1
                    while j <= self.config.context_size:
                        if i - j >= 0:
                            left_context[self.config.context_size - j] = seq[i - j]
                        if i + j < len(seq):
                            right_context[j - 1] = seq[i + j]
                        j += 1
                    yield left_context, mid_node, right_context


                # for i in range(len(seq) - self.config.context_size * 2 - 1):
                #     left_context = seq[i:i + self.config.context_size]
                #     mid_node = seq[i + self.config.context_size]
                #     right_context = seq[i + self.config.context_size + 1: i + self.config.context_size * 2 + 1]
                #     yield left_context, mid_node, right_context


    def nodes_seq_reader(self):
        left_context_batch = list()
        mid_node_batch = list()
        right_context_batch = list()
        for i in range(self.config.nodes_seq_batch_num):
            left_context, mid_node, right_context = next(self.nodes_seq_generator)
            left_context_batch.append(left_context)
            mid_node_batch.append(mid_node)
            right_context_batch.append(right_context)

        left_context_batch = np.array(left_context_batch)
        mid_node_batch = np.array(mid_node_batch)
        right_context_batch = np.array(right_context_batch)
        return left_context_batch, mid_node_batch, right_context_batch


    def get_nodes_walk_generator(self):
        while True:
            begin = datetime.datetime.now()
            walk = random_walk(self.g, self.config.walk_length)
            end = datetime.datetime.now()
            print('random walk time {}'.format(end - begin))
            np.random.shuffle(walk)
            for seq in walk:
                yield seq

    def nodes_walk_reader(self):
        nodes_walk_batch = list()
        for i in range(self.config.nodes_seq_batch_num):
            nodes_walk_batch.append(next(self.nodes_walk_generator))
        nodes_walk_batch = np.array(nodes_walk_batch)
        return nodes_walk_batch



