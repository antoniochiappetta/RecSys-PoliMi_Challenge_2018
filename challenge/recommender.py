import numpy as np
import scipy.sparse as sps

# IMPORT DATA AND SPLIT IN TRAIN AND TEST SET

URM_path = "./input data and submission/train.csv"
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
URM_all.tocsr()

train_test_split = 0.8
numInteractions = URM_all.nnz
playlist_list = np.array(playlist_list)
track_list = np.array(track_list)
rating_list = np.array(rating_list)

train_mask = np.random.choice([True, False], numInteractions, p=[train_test_split, 1 - train_test_split])
URM_train = sps.coo_matrix((rating_list[train_mask], (playlist_list[train_mask], track_list[train_mask])))
URM_train = URM_train.tocsr()

print("URM_train len")
print(len(URM_train.indices))

test_mask = np.logical_not(train_mask)
URM_test = sps.coo_matrix((rating_list[test_mask], (playlist_list[test_mask], track_list[test_mask])))
URM_test = URM_test.tocsr()

print("URM_test len")
print(len(URM_test.indices))

# METRICS

# ### Precision: how many of the recommended items are relevant

def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


# ### Recall: how many of the relevant items I was able to recommend

def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


# ### Mean Average Precision

def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score

# ### Evaluation Algorithm

def evaluate_algorithm(URM_test, recommender_object, at=5):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    for playlist_id in playlist_list_unique:

        relevant_items = URM_test[playlist_id].indices

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(playlist_id, at=at)
            num_eval += 1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

#RECOMMENDERS

class RandomRecommender(object):

    def fit(self, URM_train):
        self.numItems = URM_train.shape[0]

    def recommend(self, user_id, at=10):
        recommended_items = np.random.choice(self.numItems, at)

        return recommended_items

class TopPopRecommender(object):

    def fit(self, URM_train):
        itemPopularity = (URM_train > 0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, user_id, at=10):
        recommended_items = self.popularItems[0:at]

        return recommended_items

# Evaluate algorithms

# print("-- Random Recommender --")
# randomRecommender = RandomRecommender()
# randomRecommender.fit(URM_train)
# evaluate_algorithm(URM_test, randomRecommender)

print("-- Top Pop Recommender --")
topPopRecommender = TopPopRecommender()
topPopRecommender.fit(URM_train)
evaluate_algorithm(URM_test, topPopRecommender)

# Once chosen the TopPop one, let's generate recommendations for the target playlists

print("Generating recommendations...")

target_playlist_path = "./input data and submission/target_playlists.csv"
target_playlist_file = open(target_playlist_path, 'r')
target_playlist_file.seek(0)
target_playlist_tuples = []
numberOfTargets = 0

for line in target_playlist_file:
    line = line.replace("\n", "")
    try:
        playlist_id = int(line)
        target_playlist_tuples.append((playlist_id, list(topPopRecommender.recommend(playlist_id))))
    except ValueError:
        pass


def get_description_from_recommendation(tuple):
    playlist_string = "{}".format(tuple[0])
    tracks_string = "{}".format(tuple[1]).replace(", ", " ").replace("[", "").replace("]", "")
    return "{},{}\n".format(playlist_string, tracks_string)


submission_path = "./input data and submission/submission.csv"
submission_file = open(submission_path, 'w')
submission_file.write("playlist_id,track_ids\n")
for recommendation in target_playlist_tuples:
    submission_file.write(get_description_from_recommendation(recommendation))
    numberOfTargets += 1
submission_file.close()
print(numberOfTargets)




