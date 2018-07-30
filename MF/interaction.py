from __future__ import print_function
import pandas as pd
import numpy as np
import random
from time import time


class Interaction(object):
    def __init__(self, int_cate, neg_num, test_file_name):
   
        self.map_movie = '../Data/movie_id.txt'
        self.map_user= '../Data/user_id.txt'

        self.features_M = 0
        self.all_ratings = None
        self.train_ratings = np.loadtxt( '../Data/user_item_train.txt')
        print("load train data done!")
        # self.valid_ratings = np.loadtxt( '../Data/valid_50.rate', dtype=int, delimiter=',')
        # self.test_ratings = np.loadtxt( '../Data/user_item_test_0.0.txt')
        # self.test_ratings = np.loadtxt('../Data/test_small.txt')
        self.test_ratings = np.loadtxt(test_file_name)

        print("load test data done!")

        
        self.n_train = len(self.train_ratings)
        self.n_test = len(self.test_ratings)
        self.n_all_neg =100

        np.random.shuffle(self.train_ratings)

        self.load_all_ratings()
        # self.unzip_ratings()

    def load_all_ratings(self):
        self.features_M = len(open(self.map_movie, 'r').readlines())+1
        self.features_U = len(open(self.map_user, 'r').readlines())+1

    def unzip_ratings(self):
        print('zipping valid/test ratings')
        self.valid_ratings = self.unzipping(phase='valid')
        self.test_ratings = self.unzipping(phase='test')
        print('zipping done')

    def unzipping(self, phase='valid'):
        if phase == 'valid':
            ratings = self.valid_ratings.copy()
        else:
            ratings = self.test_ratings.copy()

        unzipped_uid = np.array([[uid]*(self.n_all_neg+1) for uid in ratings[:,0]]).reshape([-1,1])
        unzipped_iid = ratings[:,1:].reshape([-1,1])
        unzipped = np.concatenate((unzipped_uid, unzipped_iid), axis=1)
        return unzipped

    def pos_batch_generator(self, phase='train', batch_size=1024):
        if phase == 'train':
            ratings = self.train_ratings.copy()
        else:
            ratings = self.valid_ratings.copy()

        size = len(ratings)
        indicies = np.random.choice(size, batch_size)

        batch_pos = ratings[indicies]
        # uids= ratings[indicies,0].reshape([-1,1])
        # itemids=ratings[indicies,2].reshape([-1,1])
        # batch_neg = np.concatenate((uids,itemids), axis=1)
        return batch_pos


if __name__ == '__main__':
    int_cate = 'ml-latest'
    batch_size = 1024

    t1 = time()
    kg = Interaction(int_cate=int_cate, neg_num=50)
    t2 = time()
    print('build graph', t2-t1)

    t1 = time()
    for i in range(kg.n_train//batch_size):
        i_batch_pos = kg.pos_batch_generator(phase='train', batch_size=batch_size)
        for j in range(5):
            i_batch_neg = kg.neg_batch_generator(i_batch_pos)
    t2 = time()
    print(t2 - t1)
