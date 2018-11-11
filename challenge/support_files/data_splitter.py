"""
Created on 23/10/2018

@author: Antonio Chiappetta
"""

import numpy as np
import scipy.sparse as sps

def train_test_holdout(URM_all, train_perc = 0.8):


    numInteractions = URM_all.nnz
    URM_all = URM_all.tocoo()

    train_mask = np.random.choice([True, False], numInteractions, p=[train_perc, 1-train_perc])

    URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))
    URM_train = URM_train.tocsr()

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))
    URM_test = URM_test.tocsr()

    return URM_train, URM_test


def train_test_holdout_adjusted(URM_all, train_perc = 0.8):

    print('start_split')

    URM_all = URM_all.tocoo()

    temp_col_num = 0
    train_mask = np.array([]).astype(bool)
    prev_row = URM_all.row[0]

    for k in range(len(URM_all.row)):

        if URM_all.row[k] == prev_row:
            temp_col_num += 1
        else:
            if temp_col_num >= 10:
                temp_mask = np.random.choice([True, False], temp_col_num, p=[train_perc, 1 - train_perc])
            else:
                temp_mask = np.repeat(True, temp_col_num)

            train_mask = np.append(train_mask, temp_mask)
            temp_col_num = 1

        if k == len(URM_all.row)-1:
            temp_mask = np.random.choice([True, False], temp_col_num, p=[train_perc, 1 - train_perc])
            train_mask = np.append(train_mask, temp_mask)

        prev_row = URM_all.row[k]


    URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))
    URM_train = URM_train.tocsr()

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))
    URM_test = URM_all.tocsr()

    return URM_train, URM_test