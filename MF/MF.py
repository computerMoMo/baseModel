'''
Tensorflow implementation of Neural Factorization Machines as described in:
Xiangnan He, Tat-Seng Chua. Neural Factorization Machines for Sparse Predictive Analytics. In Proc. of SIGIR 2017.
This is a deep version of factorization machine and is more expressive than FM.
@author:
Xiangnan He (xiangnanhe@gmail.com)
Lizi Liao (liaolizi.llz@gmail.com)
@references:
'''
from __future__ import print_function
import os

import os
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from time import time
import argparse
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

from factory import *
from interaction import Interaction
from evaluation import eval_model_pro


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural FM.")
    parser.add_argument('--neg_num', type=int, default=100,
                        help='Number of negative samples per positive one.')
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-20m',
                        help='Choose a dataset.')
    parser.add_argument('--neg_samples', type=int, default=4,
                        help='Number of negative samples per positive one.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--keep_prob', nargs='?', default='[0.8,0.5]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0.01,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                        help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0 or 1)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of negative samples per positive one.')
    return parser.parse_args()


class MF(BaseEstimator, TransformerMixin):
    def __init__(self, features_U, features_M, args, random_seed=2018):
        # bind params to class
        self.batch_size = args.batch_size
        self.hidden_factor = args.hidden_factor
        self.features_U = features_U
        self.loss_type = args.loss_type
        self.features_M = features_M
        self.lambda_bilinear = args.lamda
        self.epoch = args.epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(eval(args.keep_prob))
        self.no_dropout = np.array([1 for i in range(len(eval(args.keep_prob)))])
        self.optimizer_type = args.optimizer
        self.learning_rate = args.lr
        self.batch_norm = args.batch_norm
        self.verbose = args.verbose
        self.activation_function = activation_function
        self.early_stop = args.early_stop
        # performance of each epoch
        self.train_loss, self.valid_hits, self.valid_ndcgs, self.test_hits, self.test_ndcgs = [], [], [], [], []

        self.margin = 1.0
        self.neg_samples = args.neg_samples
        self.row_len = args.neg_num + 1
        self.top_k = args.top_k

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.pos_features_U = tf.placeholder(tf.int32, shape=[None, None])  # (None, feature_M)
            self.pos_features_M = tf.placeholder(tf.int32, shape=[None, None])  # (None, feature_M)
            self.labels = tf.placeholder(tf.float32, shape=[None, 1])
            # self.neg_features_U = tf.placeholder(tf.int32, shape=[None, None]) # (None, feature_M)
            # self.neg_features_M = tf.placeholder(tf.int32, shape=[None, None]) # (None, feature_M)

            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ sum_square part _____________
            # get the summed up embeddings of features for positive & negative instances.
            pos_embeddings_U = tf.nn.embedding_lookup(self.weights['feature_embeddings_U'],
                                                      self.pos_features_U)  # (None, feature_M, K)
            pos_embeddings_M = tf.nn.embedding_lookup(self.weights['feature_embeddings_M'],
                                                      self.pos_features_M)  # (None, feature_M, K)
            pos_embeddings = tf.concat([pos_embeddings_U, pos_embeddings_M], 1)

            # neg_embeddings_U = tf.nn.embedding_lookup(self.weights['feature_embeddings_U'], self.neg_features_U) # (None, feature_M, K)
            # neg_embeddings_M = tf.nn.embedding_lookup(self.weights['feature_embeddings_M'], self.neg_features_M) # (None, feature_M, K)
            # neg_embeddings = tf.concat([neg_embeddings_U,neg_embeddings_M],1)

            self.pos_MF = tf.reduce_prod(pos_embeddings, axis=1)  # (None, K)
            # self.neg_MF = tf.reduce_prod(neg_embeddings, axis=1) # (None, K)

            self.pos_out = tf.reduce_sum(self.pos_MF, axis=1, keep_dims=True)
            # self.neg_out = tf.reduce_sum(self.neg_MF, axis=1, keep_dims=True)

            if self.loss_type == 'square_loss':
                self.loss = tf.reduce_sum(tf.nn.l2_loss(self.pos_out - self.labels))
            elif self.loss_type == 'log_loss':
                self.loss = tf.losses.log_loss(self.labels, self.pos_out)
            self.origin_loss = self.loss
            if self.lambda_bilinear > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.lambda_bilinear)(
                    self.weights['feature_embeddings_U'])  # regulizer
                self.loss += tf.contrib.layers.l2_regularizer(self.lambda_bilinear)(
                    self.weights['feature_embeddings_M'])  #
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['feature_embeddings_M'] = tf.Variable(
            tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.1), name='feature_embeddings_M')
        all_weights['feature_embeddings_U'] = tf.Variable(
            tf.random_normal([self.features_U, self.hidden_factor], 0.0, 0.1), name='feature_embeddings_U')
        all_weights['feature_bias_M'] = tf.Variable(tf.random_uniform([self.features_M, 1], 0.0, 0.0),
                                                    name='feature_bias_M')
        all_weights['feature_bias_U'] = tf.Variable(tf.random_uniform([self.features_U, 1], 0.0, 0.0),
                                                    name='feature_bias_U')

        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, batch_pos):  # fit a batch
        feed_dict = {self.pos_features_U: np.reshape(batch_pos[:, 0], [-1, 1]),
                     self.pos_features_M: np.reshape(batch_pos[:, 1], [-1, 1]),
                     self.labels: batch_pos[:, 2:], self.dropout_keep: self.keep_prob, self.train_phase: True}

        origin_loss, loss, opt = self.sess.run((self.origin_loss, self.loss, self.optimizer), feed_dict=feed_dict)
        return origin_loss, loss

    def train(self, interaction_data):
        for epoch in range(self.epoch):
            print("epoch:", epoch)
            t1 = time()
            total_batch = interaction_data.n_train // self.batch_size

            train_loss = 0
            for i in range(total_batch):
                print('[%d] over [%d] training done' % (i, total_batch))
                batch_pos = interaction_data.pos_batch_generator(phase='train', batch_size=self.batch_size)
                batch_origin_loss, batch_loss = self.partial_fit(batch_pos)
                train_loss += batch_loss / self.neg_samples
                print("train batch loss:", batch_origin_loss/self.neg_samples, "train batch reg loss:", batch_loss/self.neg_samples)
            t2 = time()
            # output validation
            print("train loss:", train_loss)
            if np.isnan(train_loss) == True:
                sys.exit()
            valid_hits, valid_ndcgs = 0., 0.
            # valid_hits, valid_ndcgs = self.evaluate(interaction_data.valid_ratings)
            t3 = time()

            test_hits, test_ndcgs = self.evaluate(interaction_data.test_ratings)
            t4 = time()

            # self.valid_hits.append(valid_hits)
            # self.valid_ndcgs.append(valid_ndcgs)

            self.test_hits.append(test_hits)
            self.test_ndcgs.append(test_ndcgs)

            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\t train_loss=%.4f valid=[%.4f %.4f %.1f] test=[%.4f %.4f %.1f]" % (epoch + 1,
                                                                                                            t2 - t1,
                                                                                                            train_loss,
                                                                                                            valid_hits,
                                                                                                            valid_ndcgs,
                                                                                                            t3 - t2,
                                                                                                            test_hits,
                                                                                                            test_ndcgs,
                                                                                                            t4 - t3))
            if self.early_stop > 0 and self.eva_termination(self.test_ndcgs, increas=True):
                break

    def eva_termination(self, valid, increas=True):
        if increas:
            if len(valid) > 5:
                if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        else:
            if len(valid) > 5:
                if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                    return True
        return False

    def evaluate(self, data):
        sample_num = data.shape[0]
        print('test/valid data number:', sample_num)
        total_batch = sample_num // self.batch_size
        print('total_batch num:', total_batch)
        y_pred = np.zeros((sample_num, 1))

        for j in range(total_batch):
            start = self.batch_size * j
            end = min(self.batch_size * (j + 1), sample_num)

            feed_dict = {self.pos_features_U: np.reshape(data[start:end, 0], [-1, 1]),
                         self.pos_features_M: np.reshape(data[start:end, 1], [-1, 1]),
                         self.dropout_keep: self.no_dropout, self.train_phase: False}
            y_pred[start:end, :] = self.sess.run((self.pos_out), feed_dict=feed_dict)

        y_gnd = np.zeros(y_pred.shape)
        y_gnd[::self.row_len] = 1
        hits, ndcgs = eval_model_pro(y_gnd, y_pred, K=self.top_k, row_len=self.row_len)
        return hits, ndcgs


if __name__ == '__main__':
    # Data loading
    args = parse_args()
    interaction_data = Interaction(int_cate=args.dataset, neg_num=args.neg_num)

    if args.verbose > 0:
        print(
            "Neural FM: dataset=%s, hidden_factor=%d, dropout_keep=%s,  loss_type=%s,#epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, batch_norm=%d, activation=%s, early_stop=%d"
            % (args.dataset, args.hidden_factor, args.keep_prob, args.loss_type, args.epoch, args.batch_size, args.lr,
               args.lamda, args.optimizer, args.batch_norm, args.activation, args.early_stop))
    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.sigmoid
    elif args.activation == 'tanh':
        activation_function = tf.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity

    # Training
    t1 = time()
    model = MF(interaction_data.features_U, interaction_data.features_M, args)
    model.train(interaction_data)

    # Find the best validation result across iterations
    best_valid_ndcgs = max(model.valid_ndcgs)
    best_epoch = model.valid_ndcgs.index(best_valid_ndcgs)

    final_results = "Best Iter(validation)=%d\t valid=[%.4f %.4f] test=[%.4f %.4f] @[%.1f s]" % (
    best_epoch + 1, model.valid_hits[best_epoch], model.valid_ndcgs[best_epoch], model.test_hits[best_epoch],
    model.test_ndcgs[best_epoch], time() - t1)
    print(final_results)

    save_path = 'Output/' + args.dataset + '/mf_%d.result' % (args.neg_num)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('MF: lambda=%.4f, lr=%.4f, top_k=%d, %s\n' % (args.lamda, args.lr, args.top_k, final_results))
    f.close()
