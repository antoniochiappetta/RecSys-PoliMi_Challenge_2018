"""
Created on 23/10/2018

@author: Antonio Chiappetta
"""

import numpy as np
import scipy.sparse as sps
import sys
sys.path.insert(0, 'code/challenge/support_files')

from code.challenge.support_files.evaluate_function import evaluate_algorithm
from code.challenge.support_files.data_splitter import train_test_holdout

# LOAD INTERACTION DATA

URM_path = "../input data and submission/train.csv"
URM_file = open(URM_path, 'r')
URM_file.seek(0)
numberInteractions = 0

for _ in URM_file:
    numberInteractions += 1

print("The number of interactions is {}".format(numberInteractions))

def rowSplit(rowString):
    split = rowString.split(",")
    split[1] = split[1].replace("\n", "")
    split.append(1)
    try:
        split[0] = int(split[0])
    except ValueError:
        pass
    try:
        split[1] = int(split[1])
    except ValueError:
        pass
    return tuple(split)


URM_file.seek(0)
URM_tuples = []

for line in URM_file:
    URM_tuples.append(rowSplit(line))

print(URM_tuples[1:10])

playlist_list, track_list, rating_list = zip(*URM_tuples)

playlist_list = list(playlist_list)[1:]
track_list = list(track_list)[1:]
rating_list = list(rating_list)[1:]

print(playlist_list[0:10])
print(track_list[0:10])
print(rating_list[0:10])

playlist_list_unique = list(set(playlist_list))
track_list_unique = list(set(track_list))

print(playlist_list_unique[0:10])
print(track_list_unique[0:10])

URM_all = sps.coo_matrix((rating_list, (playlist_list, track_list)))

