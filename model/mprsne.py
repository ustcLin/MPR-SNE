from config.config import Config
from graph.graph import Graph
from reader.reader import Reader
from tflib.models import Model
from tflib.rnn_cell import MRUCell

import tensorflow as tf
import numpy as np
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.metrics
import sklearn.multiclass
import os
import csv

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def layer_norm(input_tensor, name='LayerNorm'):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name, reuse=None)



class MPRSNE(object):
    def __init__(self, g, saveFile, init_emb_file=None):
        self.config = Config()
        self.g = g
        self.init_emb_file = init_emb_file
        self.saveFile = saveFile
        self.reader = Reader(self.g)
        self.layer = self.setup_layer()
        self.train_op = self.setup_train_op()
        self.test_metrics = self.get_test_metrics()
        self.loss_train_merged = tf.summary.merge(tf.get_collection('loss_train_summary'))
        self.test_merged = tf.summary.merge(tf.get_collection('multi_label_classification'))
        self.train_writer = tf.summary.FileWriter('model/summary/{}'.format(self.saveFile))

    def setup_layer(self):
        layer = dict()
        batch_num = self.config.nodes_seq_batch_num
        emb_dim = self.config.emb_dim
        neg_sample_num = self.config.loss1_neg_sample_num
        walk_len = self.config.walk_length
        gru_layer_num = self.config.gru_layer_num
        keep_prob = self.config.keep_prob
        labels_num = self.g.labels_num
        context_size = self.config.context_size
        label_l2_loss_wd = self.config.label_l2_loss_wd
        emb_l2_loss_wd = self.config.emb_l2_loss_wd
        pos_weight = 1.0 / self.config.pos_weight


        walk_nodes = tf.placeholder(tf.int32, shape=[batch_num, walk_len], name='walk_nodes')
        walk_nodes_labels = tf.placeholder(tf.float32, shape=[batch_num, walk_len, labels_num],
                                           name='walk_nodes_labels')
        neg_walk_nodes = tf.placeholder(tf.int32, shape=[batch_num, walk_len, neg_sample_num], name='neg_walk_node')

        layer['walk_nodes'] = walk_nodes
        layer['walk_nodes_labels'] = walk_nodes_labels
        layer['neg_walk_nodes'] = neg_walk_nodes

        if self.init_emb_file is not None:
            emb = list()
            with open(self.init_emb_file) as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    emb.append(row)
            emb = np.array(emb, dtype=np.float32)
            emb_initializer = tf.constant_initializer(emb)
        else:
            emb_initializer = tf.glorot_uniform_initializer(dtype=tf.float32)

        with tf.variable_scope('emb'):
            emb = tf.get_variable(name='emb',
                                  shape=[self.g.nodes_num, emb_dim],
                                  initializer=emb_initializer)

            layer['emb'] = emb
            zeros_vec = tf.constant(0, dtype=tf.float32, shape=[1, emb_dim])
            emb = tf.concat([emb, zeros_vec], 0)

        with tf.variable_scope('sup_emb'):
            sup_emb = tf.get_variable(name='sup_emb',
                                      shape=[self.g.nodes_num, emb_dim],
                                      initializer=tf.glorot_normal_initializer(dtype=tf.float32))
            layer['sup_emb'] = sup_emb

        with tf.variable_scope('labels_emb'):
            labels_emb = tf.get_variable(name='labels_emb',
                                         shape=[self.g.labels_num, emb_dim],
                                         initializer=tf.glorot_normal_initializer(dtype=tf.float32))
            labels_l2_loss = label_l2_loss_wd * tf.nn.l2_loss(labels_emb)
            tf.add_to_collection('label_loss_weight_decay', labels_l2_loss)
            layer['labels_emb'] = labels_emb

        with tf.variable_scope('lookup'):
            walk_nodes_emb = tf.nn.embedding_lookup(emb, walk_nodes, name='context_nodes_emb')
            walk_nodes_emb = layer_norm(walk_nodes_emb)
            true_sup_emb = tf.nn.embedding_lookup(sup_emb, walk_nodes, name='true_sup_emb')
            neg_sup_emb = tf.nn.embedding_lookup(sup_emb, neg_walk_nodes, name='neg_sup_emb')


        fw_context_gru_cell_list = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.DeviceWrapper(
            MRUCell(num_units=emb_dim,
                    forget_bias=0.0,
                    state_is_tuple=True,
                    activation=tf.nn.tanh,
                    reuse=tf.get_variable_scope().reuse,
                    kernel_initializer=tf.glorot_normal_initializer(dtype=tf.float32)),
            "/gpu:%d" % (i % 2)),
            input_keep_prob=keep_prob, variational_recurrent=True, input_size=emb_dim, dtype=tf.float32)
            for i in range(gru_layer_num)]

        fw_context_gru_cell = tf.nn.rnn_cell.MultiRNNCell(
            fw_context_gru_cell_list, state_is_tuple=True)

        fw_context_Residual_gru_cell = fw_context_gru_cell

        bw_context_gru_cell_list = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.DeviceWrapper(
            MRUCell(num_units=emb_dim,
                    forget_bias=0.0,
                    state_is_tuple=True,
                    activation=tf.nn.tanh,
                    reuse=tf.get_variable_scope().reuse,
                    kernel_initializer=tf.glorot_normal_initializer(dtype=tf.float32)),
            "/gpu:%d" % (i % 2)),
            input_keep_prob=keep_prob, variational_recurrent=True, input_size=emb_dim, dtype=tf.float32)
            for i in range(gru_layer_num)]

        bw_context_gru_cell = tf.nn.rnn_cell.MultiRNNCell(
            bw_context_gru_cell_list, state_is_tuple=True)


        bw_context_Residual_gru_cell = bw_context_gru_cell

        with tf.variable_scope('context'):
            (context_outputs, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_context_Residual_gru_cell,
                                                                   cell_bw=bw_context_Residual_gru_cell,
                                                                   inputs=walk_nodes_emb,
                                                                   sequence_length=[walk_len] * batch_num,
                                                                   parallel_iterations=batch_num * 2,
                                                                   dtype=tf.float32)
            context_outputs = tf.concat(context_outputs, -1)



        with tf.variable_scope('reduce_layer'):
            emb_weight = tf.get_variable(name='emb_weight',
                                         shape=[emb_dim * 2, emb_dim],
                                         initializer=tf.glorot_normal_initializer(dtype=tf.float32))
            emb_bias = tf.get_variable(name='emb_bias',
                                       shape=[emb_dim],
                                       initializer=tf.constant_initializer(0.0))
            context_outputs = tf.reshape(context_outputs, [-1, emb_dim * 2])
            context_outputs = tf.matmul(context_outputs, emb_weight) + emb_bias
            context_outputs = tf.reshape(context_outputs, [batch_num, walk_len, emb_dim])


            emb_l2_loss = emb_l2_loss_wd * tf.nn.l2_loss(emb_weight)
            tf.add_to_collection('emb_loss_weight_decay', emb_l2_loss)
            context_outputs = tf.nn.sigmoid(context_outputs)

        with tf.variable_scope('output_gates'):
            o_weight = tf.get_variable(name='o_weight',
                                       shape=[emb_dim, emb_dim],
                                       initializer=tf.glorot_normal_initializer(dtype=tf.float32))
            o_diag = tf.get_variable(name='o_diag',
                                     shape=[emb_dim])
            o_bias = tf.get_variable(name='o_bias',
                                     shape=[emb_dim],
                                     initializer=tf.constant_initializer(0.0))
            o_emb = tf.reshape(walk_nodes_emb, [-1, emb_dim])
            o_outputs = tf.reshape(context_outputs, [-1, emb_dim])
            o_gates = tf.matmul(o_emb, o_weight) + o_diag * o_outputs + o_bias
            o_gates = tf.layers.batch_normalization(o_gates, axis=-1)
            o_gates = tf.sigmoid(o_gates)
            o_gates = tf.reshape(o_gates, [batch_num, walk_len, emb_dim])
            label_context = o_gates * context_outputs



        with tf.variable_scope('walk_context'):
            walk_context = list()

            for i in range(walk_len):
                tmp_context = tf.concat([context_outputs[:, :i, :], context_outputs[:, i + 1:, :]], axis=1)
                tmp_nodes_emb = tf.concat([walk_nodes_emb[:, :i, :], walk_nodes_emb[:, i + 1:, :]], axis=1)
                tmp_context = tmp_context + tmp_nodes_emb
                # [batch, walk_len - 1, emb]

                walk_context.append(tmp_context)

            context_vec = tf.stack(walk_context, axis=1)
            # [batch, walk_len, walk_len - 1, emb]


            context_mask = np.zeros([walk_len, walk_len - 1], dtype=np.float32)
            for i in range(walk_len):
                mask_min = np.max([0, i - context_size])
                mask_max = np.min([walk_len - 1, i + context_size])
                context_mask[i, mask_min: mask_max] = 1.0
            context_mask = tf.constant(context_mask, dtype=tf.float32)
            context_mask = tf.stack([context_mask] * batch_num, axis=0)
            context_mask = tf.stack([context_mask] * emb_dim, axis=3)
            # [batch, walk_len, walk_len - 1, emb]
            context_vec = context_mask * context_vec
            # [batch, walk_len, walk_len - 1, emb]

        with tf.variable_scope('loss'):
            label_context = tf.reshape(label_context, [-1, emb_dim])
            label_score = tf.matmul(label_context, labels_emb, transpose_b=True)
            label_bias = tf.get_variable(name='label_bias',
                                         shape=[labels_num],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            label_score = label_score + label_bias
            label_score = tf.reshape(label_score, [batch_num, walk_len, labels_num])
            label_loss = tf.nn.weighted_cross_entropy_with_logits(targets=walk_nodes_labels, logits=label_score,
                                                                  pos_weight=pos_weight)
            label_loss = label_loss / pos_weight
            # [batch, walk_len, label]
            label_loss = tf.reduce_sum(label_loss, 2)
            # [batch, walk_len]
            label_loss = tf.reduce_sum(label_loss, 1)
            # [batch]
            label_loss = tf.reduce_mean(label_loss, 0)
            label_loss_wd = tf.add_n(tf.get_collection('label_loss_weight_decay'))
            label_loss = label_loss + label_loss_wd
            loss_summ = tf.summary.scalar('label_loss', label_loss)
            tf.add_to_collection('loss_train_summary', loss_summ)
            layer['label_loss'] = label_loss * 1.0

            context_vec = tf.stack([context_vec] * (neg_sample_num + 1), axis=2)
            # [batch, walk_len, node, walk_len - 1, emb]
            true_emb = tf.expand_dims(true_sup_emb, axis=2)
            # [batch, walk_len, 1, emb]
            true_neg_emb = tf.concat([true_emb, tf.negative(neg_sup_emb)], 2)
            # [batch, walk_len, node, emb]
            true_neg_emb = tf.stack([true_neg_emb] * (walk_len - 1), axis=3)
            # [batch, walk_len, node, walk_len - 1, emb]


            sig = tf.sigmoid(tf.reduce_sum(tf.multiply(context_vec, true_neg_emb), 4))
            # [batch, walk_len, node, walk_len - 1]
            sig_log = tf.log(sig)
            sig_log = tf.reduce_sum(sig_log, 2)
            # [batch, walk_len, walk_len - 1]
            sig_log = tf.reduce_sum(sig_log, 2)
            # [batch, walk_len]

            sig_log_batch = tf.reduce_sum(sig_log, 1)
            # [batch]
            emb_loss = tf.reduce_mean(sig_log_batch, 0, name='emb_loss')
            emb_loss = tf.negative(emb_loss)
            emb_loss_wd = tf.add_n(tf.get_collection('emb_loss_weight_decay'))
            emb_loss = emb_loss + emb_loss_wd
            layer['emb_loss'] = emb_loss
            loss_summ = tf.summary.scalar('emb_loss', emb_loss)
            tf.add_to_collection('loss_train_summary', loss_summ)

            layer['loss'] = layer['label_loss'] + layer['emb_loss']
            loss_summ = tf.summary.scalar('loss', layer['loss'])
            tf.add_to_collection('loss_train_summary', loss_summ)

        return layer

    def setup_train_op(self):
        with tf.variable_scope('train_op'):
            loss_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.0005
            decay_steps = 10000
            learning_rate = tf.train.exponential_decay(starter_learning_rate, loss_step, decay_steps, 0.5,
                                                       staircase=True)
            learning_rate = tf.maximum(0.0001, learning_rate)
            loss_summ = tf.summary.scalar('learning_rate', learning_rate)
            tf.add_to_collection('loss_train_summary', loss_summ)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_pre, tvars = zip(*optimizer.compute_gradients(self.layer['loss']))
            grads, _ = tf.clip_by_global_norm(grads_pre, self.config.max_grad_norm)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=loss_step)


            pre_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            pre_tvars = [var for var in tf.trainable_variables() if var.name != 'emb/emb:0']
            pre_grads_pre, _ = zip(*pre_optimizer.compute_gradients(self.layer['loss'], var_list=pre_tvars))
            pre_grads_pre, _ = tf.clip_by_global_norm(pre_grads_pre, self.config.max_grad_norm)
            pre_train_op = pre_optimizer.apply_gradients(zip(pre_grads_pre, pre_tvars))
            return [train_op, pre_train_op]



    def train(self, passes, new_training=True):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        with tf.Session(config=sess_config) as sess:
            if new_training:
                saver, global_step = Model.start_new_session(sess)
            else:
                saver, global_step = Model.continue_previous_session(sess,
                                                                     model_file='model/saver/{}'.format(self.saveFile),
                                                                     ckpt_file='model/saver/{}/checkpoint'.format(
                                                                         self.saveFile))

            self.train_writer.add_graph(sess.graph, global_step=global_step)

            walk_times = 1
            for step in range(1 + global_step, 1 + passes + global_step):
                with tf.variable_scope('Train'):
                    walk_nodes = self.reader.nodes_walk_reader()
                    neg_walk_nodes = [self.g.negative_sample(walk_nodes[i],
                                                             self.config.loss1_neg_sample_num,
                                                             self.g.nodes_degree_table)
                                      for i in range(len(walk_nodes))]
                    neg_walk_nodes = np.array(neg_walk_nodes)
                    walk_nodes_labels = list()
                    for node_list in walk_nodes:
                        nodes_label_tmp = self.g.get_train_node_label(node_list)
                        walk_nodes_labels.append(nodes_label_tmp)
                    walk_nodes_labels = np.array(walk_nodes_labels)

                    # if (step - 1) % int(self.g.nodes_num / self.config.nodes_seq_batch_num) == 0:
                    #     print(walk_times)
                    #     walk_times += 1

                    if step < 200 and self.init_emb_file is not None:
                        train_op = self.train_op[1]
                    else:
                        train_op = self.train_op[0]
                    _, train_summary, loss = sess.run(
                        [train_op,
                         self.loss_train_merged,
                         self.layer['loss']],
                        feed_dict={self.layer['walk_nodes']: walk_nodes,
                                   self.layer['walk_nodes_labels']: walk_nodes_labels,
                                   self.layer['neg_walk_nodes']: neg_walk_nodes})

                    self.train_writer.add_summary(train_summary, step)

                    if step % 500 == 0 or step == 1:
                        [node_emb, sup_emb] = sess.run([self.layer['emb'],
                                                        self.layer['sup_emb']])
                        node_emb = np.concatenate((node_emb, sup_emb), axis=1)
                        print("gobal_step {},loss {}".format(step, loss))

                    if step % 1000 == 0 or step == 1:
                        micro_f1, macro_f1 = self.multi_label_node_classification(node_emb)
                        [test_summary] = sess.run([self.test_merged],
                                                  feed_dict={self.test_metrics['micro_f1']: micro_f1,
                                                             self.test_metrics['macro_f1']: macro_f1})
                        print("micro_f1 {},macro_f1 {}".format(micro_f1, macro_f1))
                        self.train_writer.add_summary(test_summary, step)
                        saver.save(sess, 'model/saver/{}/MPRSNE'.format(self.saveFile), global_step=step)
                        print('checkpoint saved')

    def get_emb(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            saver, global_step = Model.continue_previous_session(sess,
                                                                 model_file='model/saver/{}'.format(self.saveFile),
                                                                 ckpt_file='model/saver/{}/checkpoint'.format(self.saveFile))
            ids_set = np.array(range(self.g.nodes_num))
            emb_set = tf.nn.embedding_lookup(self.layer['emb'], ids_set)
            sup_emb_set = tf.nn.embedding_lookup(self.layer['sup_emb'], ids_set)
            [emb, sup_emb] = sess.run([emb_set, sup_emb_set])
            emb = np.concatenate([emb, sup_emb], axis=1)
        return emb

    def multi_label_node_classification(self, emb):
        g = self.g
        emb = emb[:, :self.config.emb_dim]
        classes = self.g.validate_labels_set
        train_nodes_id = list()
        train_y = list()
        train_x = list()
        for node in g.train_nodes_labels.keys():
            train_nodes_id.append(node)
            train_y.append(g.train_nodes_labels[node])
            train_x.append(emb[g.nodes_ids[node], :])

        train_x = np.array(train_x)
        prepocess_y = sklearn.preprocessing.MultiLabelBinarizer(classes=classes)
        train_y = prepocess_y.fit_transform(train_y)

        test_nodes_id = list()
        test_y = list()
        test_x = list()
        for node in g.test_nodes_labels.keys():
            test_nodes_id.append(node)
            test_y.append(g.test_nodes_labels[node])
            test_x.append(emb[g.nodes_ids[node], :])

        test_x = np.array(test_x)
        test_y = prepocess_y.fit_transform(test_y)

        multi_label_classifier = sklearn.multiclass.OneVsRestClassifier(sklearn.linear_model.LogisticRegression(),
                                                                        n_jobs=1)
        multi_label_classifier.fit(train_x, train_y)
        pred_y = multi_label_classifier.predict(test_x)
        mirco_f1 = sklearn.metrics.f1_score(test_y, pred_y, average='micro')
        marco_f1 = sklearn.metrics.f1_score(test_y, pred_y, average='macro')
        return mirco_f1, marco_f1

    def get_test_metrics(self):
        micro_f1 = tf.placeholder(dtype=tf.float32, shape=None, name='micro_f1')
        macro_f1 = tf.placeholder(dtype=tf.float32, shape=None, name='macro_f1')
        loss_summ = tf.summary.scalar('micro_f1', micro_f1)
        tf.add_to_collection('multi_label_classification', loss_summ)
        loss_summ = tf.summary.scalar('macro_f1', macro_f1)
        tf.add_to_collection('multi_label_classification', loss_summ)
        test_metrics = dict()
        test_metrics['micro_f1'] = micro_f1
        test_metrics['macro_f1'] = macro_f1
        return test_metrics


