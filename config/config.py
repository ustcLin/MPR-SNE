class Config(object):
    def __init__(self):
        self.use_cpu_rate = 1.0

        self.train_label_size = 0.5
        self.validate_label_size = 0.1
        self.test_label_size = 0.4


        # self.train_node_size = 0.05
        self.train_node_size = 0.6

        self.train_edge_size = 0.7

        # random walk
        self.walks_per_node = 10
        self.walk_length = 20
        self.context_size = 5

        # network struction
        self.nodes_seq_batch_num = 128  # blog
        # self.nodes_seq_batch_num = 32 # flickr
        self.gru_layer_num = 1
        self.emb_dim = 128
        self.loss1_neg_sample_num = 5
        self.max_grad_norm = 3
        self.alpha1 = 0.5
        self.emb_l2_loss_wd = 0.001
        self.keep_prob = 0.8
        self.loss1_train_step = 50
        self.loss1_l2_wd = 0.01

        # network label
        self.alpha2 = 0.5
        self.label_l2_loss_wd = 0.001
        self.loss2_train_step = 50


        self.pos_weight = pow(10, 0)    # blog
        # self.pos_weight = pow(10, -0.5)    # flickr