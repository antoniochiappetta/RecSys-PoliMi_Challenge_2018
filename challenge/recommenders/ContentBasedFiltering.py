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
from code.challenge.support_files.compute_similarity import Compute_Similarity_Python

# LOAD INTERACTION DATA

print("MARK: - Load interaction data")

URM_path = "../input data and submission/train.csv"
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
URM_train, URM_test = train_test_holdout(URM_all)

print("URM_train len")
print(len(URM_train.indices))

print("URM_test len")
print(len(URM_test.indices))

# LOAD CONTENT DATA

print("MARK: - Load Content Data")

ICM_path = "../input data and submission/tracks.csv"
ICM_file = open(ICM_path, 'r')
ICM_file.seek(0)

def rowSplitTracks(rowString):
    split = rowString.split(",")
    try:
        split[0] = int(split[0])
    except ValueError:
        pass
    try:
        split[1] = int(split[1])
    except ValueError:
        pass
    try:
        split[2] = int(split[2])
    except ValueError:
        pass
    try:
        split[3] = int(split[3])
    except ValueError:
        pass
    return tuple(split)

ICM_tuples = []

for line in ICM_file:
    ICM_tuples.append(rowSplitTracks(line))

print(ICM_tuples[1:10])

track_list_icm, album_list_icm, artist_list_icm, duration_list_icm = zip(*ICM_tuples)

track_list_icm = list(track_list_icm)[1:]
album_list_icm = list(album_list_icm)[1:]
artist_list_icm = list(artist_list_icm)[1:]
duration_list_icm = list(duration_list_icm)[1:]

print(track_list_icm[0:10])
print(album_list_icm[0:10])
print(artist_list_icm[0:10])
print(duration_list_icm[0:10])

ones = np.ones(len(track_list_icm))

ICM_album = sps.coo_matrix((ones, (track_list_icm, album_list_icm)))
ICM_album = ICM_album.tocsr()

ICM_artist = sps.coo_matrix((ones, (track_list_icm, artist_list_icm)))
ICM_artist = ICM_artist.tocsr()

ICM_duration = sps.coo_matrix((ones, (track_list_icm, duration_list_icm)))
ICM_duration = ICM_duration.tocsr()

print("ICM matrices generated")
print(ICM_album[0])
print(ICM_artist[0])
print(ICM_duration[0])

# MARK: - Data splitting

ICM_album_train, ICM_album_test = train_test_holdout(ICM_album)
ICM_artist_train, ICM_artist_test = train_test_holdout(ICM_artist)
ICM_duration_train, ICM_duration_test = train_test_holdout(ICM_duration)

# MARK: - Recommender

class ItemCBFKNNRecommender(object):

    def __init__(self, URM, ICM_album, ICM_artist, ICM_duration):
        self.URM = URM
        self.ICM_album = ICM_album
        self.ICM_artist = ICM_artist
        self.ICM_duration = ICM_duration

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object_album = Compute_Similarity_Python(self.ICM_album.T, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity)
        similarity_object_artist = Compute_Similarity_Python(self.ICM_artist.T, shrink=shrink, topK=topK,
                                                            normalize=normalize, similarity=similarity)
        similarity_object_duration = Compute_Similarity_Python(self.ICM_duration.T, shrink=shrink, topK=topK,
                                                            normalize=normalize, similarity=similarity)

        self.W_sparse_album = similarity_object_album.compute_similarity()
        self.W_sparse_artist = similarity_object_artist.compute_similarity()
        self.W_sparse_duration = similarity_object_duration.compute_similarity()
        self.W_sparse = self.W_sparse_album * 0.2 + self.W_sparse_artist * 0.8 + self.W_sparse_duration * 0.0

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
        end_pos = self.URM.indptr[playlist_id+1]

        playlist_profile = self.URM.indices[start_pos:end_pos]

        scores[playlist_profile] = -np.inf

        return scores

# MARK: - Train and evaluate algorithm

cbfrecommender = ItemCBFKNNRecommender(URM_train, ICM_album, ICM_artist, ICM_duration)
cbfrecommender.fit(shrink=0.0, topK=50)
evaluate_algorithm(URM_test, cbfrecommender)

# Let's generate recommendations for the target playlists

print("Generating recommendations...")

target_playlist_path = "../input data and submission/target_playlists.csv"
target_playlist_file = open(target_playlist_path, 'r')
target_playlist_file.seek(0)
target_playlist_tuples = []
numberOfTargets = 0

for line in target_playlist_file:
    line = line.replace("\n", "")
    try:
        playlist_id = int(line)
        target_playlist_tuples.append((playlist_id, list(cbfrecommender.recommend(playlist_id))))
    except ValueError:
        pass


def get_description_from_recommendation(tuple):
    playlist_string = "{}".format(tuple[0])
    tracks_string = "{}".format(tuple[1]).replace(", ", " ").replace("[", "").replace("]", "")
    return "{},{}\n".format(playlist_string, tracks_string)


submission_path = "../input data and submission/submission.csv"
submission_file = open(submission_path, 'w')
submission_file.write("playlist_id,track_ids\n")
for recommendation in target_playlist_tuples:
    submission_file.write(get_description_from_recommendation(recommendation))
    numberOfTargets += 1
submission_file.close()
print(numberOfTargets)


