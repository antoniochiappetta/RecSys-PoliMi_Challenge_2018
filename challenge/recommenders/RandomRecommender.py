import numpy as np
import scipy.sparse as sps
import sys
sys.path.insert(0, 'code/challenge/support_files')

from code_recsys.challenge.support_files.evaluate_function import evaluate_algorithm
from code_recsys.challenge.support_files.data_splitter import train_test_holdout

# IMPORT DATA AND SPLIT IN TRAIN AND TEST SET

URM_path = "../input/train.csv"
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
URM_train, URM_test = train_test_holdout(URM_all)

print("URM_train len")
print(len(URM_train.indices))

print("URM_test len")
print(len(URM_test.indices))

class RandomRecommender(object):

    def fit(self, URM_train):
        self.numItems = URM_train.shape[0]

    def recommend(self, user_id, at=10):
        recommended_items = np.random.choice(self.numItems, at)

        return recommended_items

# Evaluate algorithm

print("-- Random Recommender --")
randomRecommender = RandomRecommender()
randomRecommender.fit(URM_train)
evaluate_algorithm(URM_test, randomRecommender)

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
        target_playlist_tuples.append((playlist_id, list(randomRecommender.recommend(playlist_id))))
    except ValueError:
        pass


def get_description_from_recommendation(tuple):
    playlist_string = "{}".format(tuple[0])
    tracks_string = "{}".format(tuple[1]).replace(", ", " ").replace("[", "").replace("]", "")
    return "{},{}\n".format(playlist_string, tracks_string)


submission_path = "../input/submission.csv"
submission_file = open(submission_path, 'w')
submission_file.write("playlist_id,track_ids\n")
for recommendation in target_playlist_tuples:
    submission_file.write(get_description_from_recommendation(recommendation))
    numberOfTargets += 1
submission_file.close()
print(numberOfTargets)