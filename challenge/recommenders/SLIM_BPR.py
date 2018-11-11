"""
Created on 23/10/2018

@author: Antonio Chiappetta
"""

import numpy as np
import scipy.sparse as sps
import sys
import time
sys.path.insert(0, 'code/challenge/support_files')

from code_recsys.challenge.support_files.evaluate_function import evaluate_algorithm
from code_recsys.challenge.support_files.data_splitter import train_test_holdout_adjusted
from code_recsys.challenge.support_files.compute_similarity import Compute_Similarity_Python

# LOAD INTERACTION DATA

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

# MARK: - Utilities

class BPR_Sampling(object):

    def __init__(self):
        super(BPR_Sampling, self).__init__()


    def sampleUser(self):
        """
        Sample a user that has viewed at least one and not all items
        :return: user_id
        """
        while (True):

            user_id = np.random.randint(0, self.n_users)
            numSeenItems = self.URM_train[user_id].nnz

            if (numSeenItems > 0 and numSeenItems < self.n_items):
                return user_id


    def sampleItemPair(self, user_id):
        """
        Returns for the given user a random seen item and a random not seen item
        :param user_id:
        :return: pos_item_id, neg_item_id
        """

        userSeenItems = self.URM_train[user_id].indices

        pos_item_id = userSeenItems[np.random.randint(0, len(userSeenItems))]

        while (True):

            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                return pos_item_id, neg_item_id


    def sampleTriple(self):
        """
        Randomly samples a user and then samples randomly a seen and not seen item
        :return: user_id, pos_item_id, neg_item_id
        """

        user_id = self.sampleUser()
        pos_item_id, neg_item_id = self.sampleItemPair(user_id)

        return user_id, pos_item_id, neg_item_id


    def initializeFastSampling(self, positive_threshold=0):
        print("Initializing fast sampling")

        self.eligibleUsers = []
        self.userSeenItems = dict()

        # Select only positive interactions

        URM_train_positive = self.URM_train.multiply(self.URM_train>positive_threshold)
        
        for user_id in range(self.n_users):

            if (URM_train_positive[user_id].nnz > 0):
                self.eligibleUsers.append(user_id)
                self.userSeenItems[user_id] = URM_train_positive[user_id].indices

        self.eligibleUsers = np.array(self.eligibleUsers)



    def sampleBatch(self):
        user_id_list = np.random.choice(self.eligibleUsers, size=(self.batch_size))
        pos_item_id_list = [None]*self.batch_size
        neg_item_id_list = [None]*self.batch_size

        for sample_index in range(self.batch_size):
            user_id = user_id_list[sample_index]

            pos_item_id_list[sample_index] = np.random.choice(self.userSeenItems[user_id])

            negItemSelected = False

            # It's faster to just try again then to build a mapping of the non-seen items
            # for every user
            while (not negItemSelected):
                neg_item_id = np.random.randint(0, self.n_items)

                if (neg_item_id not in self.userSeenItems[user_id]):
                    negItemSelected = True
                    neg_item_id_list[sample_index] = neg_item_id

        print("SAMPLE BATCH")
        print("User id list")
        print(user_id_list)
        print("pos item id list")
        print(pos_item_id_list)
        print("neg item id list")
        print(neg_item_id_list)
        return user_id_list, pos_item_id_list, neg_item_id_list

def sigmoidFunction(x):
  return 1 / (1 + np.exp(-x))

# MARK: - Recommender

class SLIM_BPR_Python(BPR_Sampling):

    def __init__(self, URM_train, positive_threshold=0, sparse_weights = False):
        super(SLIM_BPR_Python, self).__init__()

        """
          Creates a new object for training and testing a Bayesian
          Personalised Ranking (BPR) SLIM

          This object uses the Theano library for training the model, meaning
          it can run on a GPU through CUDA. To make sure your Theano
          install is using the GPU, see:

            http://deeplearning.net/software/theano/tutorial/using_gpu.html

          When running on CPU, we recommend using OpenBLAS.

            http://www.openblas.net/
        """
        """
        if objective!='sigmoid' and objective != 'logsigmoid':
            raise ValueError("Objective not valid. Acceptable values are 'sigmoid' and 'logsigmoid'. Provided value was '{}'".format(objective))
        self.objective = objective
        """

        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False
        self.sparse_weights = sparse_weights
        self.positive_threshold = positive_threshold


        self.URM_mask = self.URM_train.copy()

        self.URM_mask.data = self.URM_mask.data >= self.positive_threshold

    def check_matrix(X, format='csc', dtype=np.float32):
        if format == 'csc' and not isinstance(X, sps.csc_matrix):
            return X.tocsc().astype(dtype)
        elif format == 'csr' and not isinstance(X, sps.csr_matrix):
            return X.tocsr().astype(dtype)
        elif format == 'coo' and not isinstance(X, sps.coo_matrix):
            return X.tocoo().astype(dtype)
        elif format == 'dok' and not isinstance(X, sps.dok_matrix):
            return X.todok().astype(dtype)
        elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
            return X.tobsr().astype(dtype)
        elif format == 'dia' and not isinstance(X, sps.dia_matrix):
            return X.todia().astype(dtype)
        elif format == 'lil' and not isinstance(X, sps.lil_matrix):
            return X.tolil().astype(dtype)
        else:
            return X.astype(dtype)

    def similarityMatrixTopK(self, item_weights, forceSparseOutput=True, k=100, verbose=False, inplace=True):
        """
        The function selects the TopK most similar elements, column-wise

        :param item_weights:
        :param forceSparseOutput:
        :param k:
        :param verbose:
        :param inplace: Default True, WARNING matrix will be modified
        :return:
        """

        assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

        start_time = time.time()

        if verbose:
            print("Generating topK matrix")

        nitems = item_weights.shape[1]
        k = min(k, nitems)

        # for each column, keep only the top-k scored items
        sparse_weights = not isinstance(item_weights, np.ndarray)

        if not sparse_weights:

            idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

            if inplace:
                W = item_weights
            else:
                W = item_weights.copy()

            # index of the items that don't belong to the top-k similar items of each column
            not_top_k = idx_sorted[:-k, :]
            # use numpy fancy indexing to zero-out the values in sim without using a for loop
            W[not_top_k, np.arange(nitems)] = 0.0

            if forceSparseOutput:
                W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

                if verbose:
                    print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

                return W_sparse

            if verbose:
                print("Dense TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W

        else:
            # iterate over each column and keep only the top-k similar items
            data, rows_indices, cols_indptr = [], [], []

            item_weights = self.check_matrix(item_weights, format='csc', dtype=np.float32)

            for item_idx in range(nitems):
                cols_indptr.append(len(data))

                start_position = item_weights.indptr[item_idx]
                end_position = item_weights.indptr[item_idx + 1]

                column_data = item_weights.data[start_position:end_position]
                column_row_index = item_weights.indices[start_position:end_position]

                idx_sorted = np.argsort(column_data)  # sort by column
                top_k_idx = idx_sorted[-k:]

                data.extend(column_data[top_k_idx])
                rows_indices.extend(column_row_index[top_k_idx])

            cols_indptr.append(len(data))

            # During testing CSR is faster
            W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
            W_sparse = W_sparse.tocsr()

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse

    def updateSimilarityMatrix(self):

        if self.topK != False:
            if self.sparse_weights:
                self.W_sparse = self.similarityMatrixTopK(self.S.T, k=self.topK, forceSparseOutput=True)
            else:
                self.W = self.similarityMatrixTopK(self.S.T, k=self.topK, forceSparseOutput=False)

        else:
            if self.sparse_weights:
                self.W_sparse = sps.csr_matrix(self.S.T)
            else:
                self.W = self.S.T



    def updateWeightsLoop(self, u, i, j):
        """
        Define the update rules to be used in the train phase and compile the train function
        :return:
        """

        x_ui = self.S[i]
        x_uj = self.S[j]

        # The difference is computed on the whole row not only on the user_seen items
        # The performance seems to be higher this way
        x_uij = x_ui - x_uj

        # Sigmoid whose argument is minus in order for the exponent of the exponential to be positive
        sigmoid = sigmoidFunction(-x_uij)

        delta_i = sigmoid-self.lambda_i*self.S[i]
        delta_j = -sigmoid-self.lambda_j*self.S[j]

        # Since a shared variable may be the target of only one update rule
        # All the required updates are chained inside a subtensor
        for sampleIndex in range(self.batch_size):

            user_id = u[sampleIndex]

            for item_id in self.userSeenItems[user_id]:
                # Do not update items i
                if item_id != i[sampleIndex]:
                    self.S[i] += self.learning_rate * delta_i

                # Do not update j
                if item_id != j[sampleIndex]:
                    self.S[j] += self.learning_rate * delta_j


    def updateWeightsBatch(self, u, i, j):
        """
        Define the update rules to be used in the train phase and compile the train function
        :return:
        """

        if self.batch_size==1:
            seenItems = self.userSeenItems[u[0]]

            x_ui = self.S[i, seenItems]
            x_uj = self.S[j, seenItems]

            # The difference is computed on the user_seen items
            x_uij = x_ui - x_uj

            #x_uij = x_uij[0,seenItems]
            x_uij = np.sum(x_uij)

            # log(sigm(+x_uij))
            gradient = 1 / (1 + np.exp(x_uij))

            # sigm(-x_uij)
            #exp = np.exp(x_uij)
            #gradient = exp/np.power(exp+1, 2)

        else:

            x_ui = self.S[i]
            x_uj = self.S[j]

            # The difference is computed on the user_seen items
            x_uij = x_ui - x_uj

            x_uij = self.URM_mask[u,:].dot(x_uij.T).diagonal()

            gradient = np.sum(1 / (1 + np.exp(x_uij))) / self.batch_size

        # Sigmoid whose argument is minus in order for the exponent of the exponential to be positive
        # Best performance with: gradient = np.sum(expit(-x_uij)) / self.batch_size
        #gradient = np.sum(x_uij) / self.batch_size
        #gradient = expit(-gradient)
        #gradient = np.sum(expit(-x_uij)) / self.batch_size
        #gradient = np.sum(np.log(expit(x_uij))) / self.batch_size
        #gradient = np.sum(1/(1+np.exp(x_uij))) / self.batch_size
        #gradient = min(10, max(-10, gradient))+10


        if self.batch_size==1:

            userSeenItems = self.userSeenItems[u[0]]

            self.S[i, userSeenItems] += self.learning_rate * gradient
            self.S[i, i] = 0

            self.S[j, userSeenItems] -= self.learning_rate * gradient
            self.S[j, j] = 0



        else:
            itemsToUpdate = np.array(self.URM_mask[u, :].sum(axis=0) > 0).ravel()

            # Do not update items i, set all user-posItem to false
            # itemsToUpdate[i] = False

            self.S[i] += self.learning_rate * gradient * itemsToUpdate
            self.S[i, i] = 0

            # Now update i, setting all user-posItem to true
            # Do not update j

            # itemsToUpdate[i] = True
            # itemsToUpdate[j] = False

            self.S[j] -= self.learning_rate * gradient * itemsToUpdate
            self.S[j, j] = 0

    def fit(self, epochs=30, logFile=None, URM_test=None, filterTopPop = False, minRatingsPerUser=1,
            batch_size = 1000, validate_every_N_epochs = 1, start_validation_after_N_epochs = 0,
            lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05, topK = False):



        if self.sparse_weights:
            self.S = sps.csr_matrix((self.n_items, self.n_items), dtype=np.float32)
        else:
            self.S = np.zeros((self.n_items, self.n_items)).astype('float32')


        self.initializeFastSampling(positive_threshold=self.positive_threshold)


        self.fit_alreadyInitialized(epochs=epochs,
                                    logFile=logFile,
                                    URM_test=URM_test,
                                    filterTopPop = filterTopPop,
                                    minRatingsPerUser=minRatingsPerUser,
                                    batch_size = batch_size,
                                    validate_every_N_epochs = validate_every_N_epochs,
                                    start_validation_after_N_epochs = start_validation_after_N_epochs,
                                    lambda_i = lambda_i,
                                    lambda_j = lambda_j,
                                    learning_rate = learning_rate,
                                    topK = topK)



    def fit_alreadyInitialized(self, epochs=30, logFile=None, URM_test=None, filterTopPop = False, minRatingsPerUser=1,
            batch_size = 1000, validate_every_N_epochs = 1, start_validation_after_N_epochs = 0,
            lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05, topK = False):
        """
        Fits the model performing a round of testing at the end of each epoch
        :param epochs:
        :param batch_size:
        :param logFile:
        :param URM_test:
        :return:
        """


        if(topK != False and topK<1):
            raise ValueError("TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(topK))
        self.topK = topK


        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate


        start_time_train = time.time()

        for currentEpoch in range(epochs):

            start_time_epoch = time.time()

            if self.batch_size>0:
                self.epochIteration()
            else:
                print("No batch not available")


            if (URM_test is not None) and ((currentEpoch +1 )% validate_every_N_epochs == 0) and \
                            currentEpoch >= start_validation_after_N_epochs:

                print("Evaluation begins")

                self.updateSimilarityMatrix()

                results_run = self.evaluateRecommendations(URM_test, filterTopPop=filterTopPop,
                                                           minRatingsPerUser=minRatingsPerUser)

                self.writeCurrentConfig(currentEpoch, results_run, logFile)

                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs,
                                                                     float(time.time() - start_time_epoch) / 60))


            # Fit with no validation
            else:
                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs,
                                                                         float(time.time() - start_time_epoch) / 60))

        self.updateSimilarityMatrix()

        print("Fit completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))

        sys.stdout.flush()



    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'lambda_i': self.lambda_i,
                          'lambda_j': self.lambda_j,
                          'batch_size': self.batch_size,
                          'learn_rate': self.learning_rate,
                          'topK_similarity': self.topK,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))
        # print("Weights: {}\n".format(str(list(self.weights))))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            # logFile.write("Weights: {}\n".format(str(list(self.weights))))
            logFile.flush()



    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = int(self.URM_mask.nnz*1)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        totalNumberOfBatch = int(numPositiveIteractions/self.batch_size)+1

        # Uniform user sampling without replacement
        for numCurrentBatch in range(totalNumberOfBatch):

            sgd_users, sgd_pos_items, sgd_neg_items = self.sampleBatch()

            self.updateWeightsBatch(
                sgd_users,
                sgd_pos_items,
                sgd_neg_items
                )

            """
            self.updateWeightsLoop(
                sgd_users,
                sgd_pos_items,
                sgd_neg_items
                )
            """

            if(time.time() - start_time_batch >= 30 or numCurrentBatch==totalNumberOfBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numCurrentBatch*self.batch_size,
                    100.0* float(numCurrentBatch*self.batch_size)/numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(numCurrentBatch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()



        self.S[np.arange(0, self.n_items), np.arange(0, self.n_items)] = 0.0

    def recommend(self, playlist_id, at=10, exclude_duplicates=True):
        # compute the scores using the dot product
        playlist_profile = self.URM_train[playlist_id]
        user_ratings = self.URM_train[self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]

        relevant_weights = self.W[playlist_profile]
        scores = relevant_weights.T.dot(user_ratings)
        if exclude_duplicates:
            scores = self.filter_seen(playlist_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, playlist_id, scores):
        start_pos = self.URM_train.indptr[playlist_id]
        end_pos = self.URM_train.indptr[playlist_id + 1]

        playlist_profile = self.URM_train.indices[start_pos:end_pos]

        scores[playlist_profile] = -np.inf

        return scores


# MARK: - Train and evaluate algorithm

slimbprrecommendertest = SLIM_BPR_Python(URM_train, positive_threshold=0, sparse_weights=True)
slimbprrecommendertest.fit(epochs=2,topK=200)
evaluate_algorithm(URM_test, slimbprrecommendertest)

slimbprrecommender = SLIM_BPR_Python(URM_test, positive_threshold=0, sparse_weights=True)
slimbprrecommender.fit(epochs=2,topK=200)

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
        target_playlist_tuples.append((playlist_id, list(slimbprrecommender.recommend(playlist_id))))
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


