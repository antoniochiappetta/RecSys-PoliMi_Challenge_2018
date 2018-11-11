"""
Created on 23/10/2018

@author: Antonio Chiappetta
"""

import numpy as np
import scipy.sparse as sps
import sys
sys.path.insert(0, 'code/challenge/support_files')

from code_recsys.challenge.support_files.evaluate_function import evaluate_algorithm
from code_recsys.challenge.support_files.data_splitter import train_test_holdout_adjusted
from code_recsys.challenge.support_files.compute_similarity import Compute_Similarity_Python

print("MARK: - Load interaction data")

URM_path = "../input/train.csv"
URM_file = open(URM_path, 'r')
URM_file.seek(0)
numberInteractions = 0

for _ in URM_file:
    numberInteractions += 1

print("The number of interactions is {}".format(numberInteractions))


def rowSplitTrain(rowString):
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
    URM_tuples.append(rowSplitTrain(line))

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
URM_train, URM_test = train_test_holdout_adjusted(URM_all)

print("URM_train len")
print(len(URM_train.indices))

print("URM_test len")
print(len(URM_test.indices))


# MARK: - Recommender

class ItemCFKNNRecommender(object):

    def __init__(self, URM, URM_all):
        self.URM = URM
        self.URM_all = URM_all

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object_track = Compute_Similarity_Python(self.URM_all, shrink=shrink, topK=topK, normalize=normalize,
                                                            similarity=similarity)

        self.W_sparse = similarity_object_track.compute_similarity()

    def recommend(self, playlist_id, at=10, exclude_duplicates=True):
        # compute the scores using the dot product
        playlist_profile = self.URM[playlist_id]
        scores = playlist_profile.dot(self.W_sparse).toarray().ravel()
        if exclude_duplicates:
            scores = self.filter_seen(playlist_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, playlist_id, scores):
        start_pos = self.URM.indptr[playlist_id]
        end_pos = self.URM.indptr[playlist_id + 1]

        playlist_profile = self.URM.indices[start_pos:end_pos]

        scores[playlist_profile] = -np.inf

        return scores


# MARK: - Train and evaluate algorithm

icfrecommendertest = ItemCFKNNRecommender(URM_train, URM_all)
icfrecommendertest.fit(shrink=2,topK=100)
evaluate_algorithm(URM_test, icfrecommendertest, at=10)

icfrecommender = ItemCFKNNRecommender(URM_test, URM_all)
icfrecommender.fit(shrink=2,topK=100)

# Let's generate recommendations for the target playlists

print("Generating recommendations...")

target_playlist_path = "../input/target_playlists.csv"
target_playlist_file = open(target_playlist_path, 'r')
target_playlist_file.seek(0)
target_playlist_tuples = []
numberOfTargets = 0

for line in target_playlist_file:
    line = line.replace("\n", "")
    try:
        playlist_id = int(line)
        target_playlist_tuples.append((playlist_id, list(icfrecommender.recommend(playlist_id))))
    except ValueError:
        pass


def get_description_from_recommendation(tuple):
    playlist_string = "{}".format(tuple[0])
    tracks_string = "{}".format(tuple[1]).replace(", ", " ").replace("[", "").replace("]", "")
    return "{},{}\n".format(playlist_string, tracks_string)


submission_path = "submission.csv"
submission_file = open(submission_path, 'w')
submission_file.write("playlist_id,track_ids\n")
for recommendation in target_playlist_tuples:
    submission_file.write(get_description_from_recommendation(recommendation))
    numberOfTargets += 1
submission_file.close()
print(numberOfTargets)