from __future__ import print_function
import pandas as pd
import numpy as np
import random
from time import time


class Interaction(object):
    def __init__(self, int_cate, neg_num):
   
        self.map_movie = '../../map_movie.csv'
        self.map_user= '../../map_user.csv'

        self.features_M = 0
        self.all_ratings = None
        self.train_ratings =np.loadtxt( '../../train.rate', dtype=int, delimiter=',')
        self.valid_ratings = np.loadtxt( '../../valid_50.rate', dtype=int, delimiter=',')
        self.test_ratings = np.loadtxt( '../../test_50.rate', dtype=int, delimiter=',')
        
        self.n_train = len(self.train_ratings)
        self.n_valid, self.n_all_neg = self.valid_ratings.shape
        self.n_test = len(self.test_ratings)
        self.n_all_neg -= 2

        np.random.shuffle(self.train_ratings)

        self.load_all_ratings()
        self.unzip_ratings()

    def load_all_ratings(self):
        links_df = pd.read_csv( self.map_movie)
        self.features_M = len(pd.unique(links_df['mapId']))
        self.features_U = len(pd.unique(pd.read_csv(self.map_user )['mapId']))
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

        batch_pos = ratings[indicies,0:2]
        uids= ratings[indicies,0].reshape([-1,1])
        itemids=ratings[indicies,2].reshape([-1,1])
        batch_neg = np.concatenate((uids,itemids), axis=1)
        return batch_pos, batch_neg


if __name__ == '__main__':
    int_cate = 'ml-latest'
    batch_size = 1024

    t1 = time()
    kg = Interaction(int_cate=int_cate, neg_num=50)
    t2 = time()
    print('build graph', t2-t1)

    t1 = time()
    for i in range(kg.n_train/batch_size):
        i_batch_pos = kg.pos_batch_generator(phase='train', batch_size=batch_size)
        for j in range(5):
            i_batch_neg = kg.neg_batch_generator(i_batch_pos)
    t2 = time()
    print(t2 - t1)
