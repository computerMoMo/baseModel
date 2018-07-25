import numpy as np
import pandas as pd
from time import time
from evaluation import eval_model_pro


def item_freq(train_ratings):
    pos_ratings = train_ratings[train_ratings[:,2]>0]
    item_dict = dict()

    for rate in pos_ratings:
        uid = int(rate[0])
        iid = int(rate[1])

        if iid in item_dict.keys():
            item_dict[iid] += 1
        else:
            item_dict[iid] = 1

    # item_id = sorted(item_dict, key=item_dict.__getitem__, reverse=True)
    return item_dict


def evaluate(data, item_freq_dict):
    sample_num = data.shape[0]
    print("test/valid data number:" + str(sample_num))

    y_gnd = data[:, 2:]

    y_pred = np.zeros(y_gnd.shape)

    for j in range(sample_num):
        i_id = int(data[j, 1])
        if i_id in item_freq_dict.keys():
            i_freq = item_freq_dict[i_id]
        else:
            i_freq = 1
        y_pred[j, 0] = i_freq

    hits, ndcgs = eval_model_pro(y_gnd, y_pred, K=10, row_len=100+1)
    return hits, ndcgs

if __name__ == '__main__':
    t1 = time()

    path = '../Data/'
    train_ratings = np.loadtxt(path + 'user_item_train.txt')
    test_ratings = np.loadtxt(path + 'test_small.txt')

    print('already load train and test data')

    item_freq_dict = item_freq(train_ratings)
    item_num = len(item_freq_dict.keys())

    test_hits, test_ndcgs = evaluate(test_ratings, item_freq_dict)

    final_results = "prediction results: test=[%.4f %.4f] @[%.1f s]" % (test_hits, test_ndcgs, time() - t1)
    print(final_results)

