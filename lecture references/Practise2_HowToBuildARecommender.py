#!/usr/bin/env python
# coding: utf-8

# # Recommender Systems 2017/18
# 
# ### Practice 2 - Non personalized recommenders

# #### We will use the Movielens 10 million dataset. We download it and uncompress the file we need

# In[1]:


from urllib.request import urlretrieve
import zipfile

# In[2]:


urlretrieve("http://files.grouplens.org/datasets/movielens/ml-10m.zip", "lecture references/movielens_10m.zip")

# In[3]:


dataFile = zipfile.ZipFile("lecture references/movielens_10m.zip")

URM_path = dataFile.extract("lecture references/ml-10M100K/ratings.dat")

URM_file = open(URM_path, 'r')

# In[4]:


type(URM_file)

# #### Let's take a look at the data

# In[5]:


for _ in range(10):
    print(URM_file.readline())

# In[6]:


# Start from beginning of the file
URM_file.seek(0)
numberInteractions = 0

for _ in URM_file:
    numberInteractions += 1

print("The number of interactions is {}".format(numberInteractions))


# ### We split each row to separate user, item, rating and timestamp. We do that with a custom function creating a tuple for each interaction

# In[7]:


def rowSplit(rowString):
    split = rowString.split("::")
    split[3] = split[3].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])
    split[3] = int(split[3])

    result = tuple(split)

    return result


URM_file.seek(0)
URM_tuples = []

for line in URM_file:
    URM_tuples.append(rowSplit(line))

URM_tuples[0:10]

# ### We can easily separate the four columns in different independent lists

# In[8]:


userList, itemList, ratingList, timestampList = zip(*URM_tuples)

userList = list(userList)
itemList = list(itemList)
ratingList = list(ratingList)
timestampList = list(timestampList)

# In[9]:


userList[0:10]

# In[10]:


itemList[0:10]

# In[11]:


ratingList[0:10]

# In[12]:


timestampList[0:10]

# ### Now we can display some statistics

# In[13]:


userList_unique = list(set(userList))
itemList_unique = list(set(itemList))

numUsers = len(userList_unique)
numItems = len(itemList_unique)

print("Number of items\t {}, Number of users\t {}".format(numItems, numUsers))
print("Max ID items\t {}, Max Id users\t {}\n".format(max(itemList_unique), max(userList_unique)))
print("Average interactions per user {:.2f}".format(numberInteractions / numUsers))
print("Average interactions per item {:.2f}\n".format(numberInteractions / numItems))

print("Sparsity {:.2f} %".format((1 - float(numberInteractions) / (numItems * numUsers)) * 100))

# ##### Rating distribution in time

# In[14]:


import matplotlib.pyplot as pyplot

# Clone the list to avoid changing the ordering of the original data
timestamp_sorted = list(timestampList)
timestamp_sorted.sort()

pyplot.plot(timestamp_sorted, 'ro')
pyplot.ylabel('Timestamp ')
pyplot.xlabel('Item Index')
pyplot.show()

# #### To store the data we use a sparse matrix. We build it as a COO matrix and then change its format
# 
# #### The COO constructor expects (data, (row, column))

# In[15]:


import scipy.sparse as sps

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))

URM_all

# In[16]:


URM_all.tocsr()

# ### Item popularity

# In[17]:


import numpy as np

itemPopularity = (URM_all > 0).sum(axis=0)
itemPopularity

# In[18]:


itemPopularity = np.array(itemPopularity).squeeze()
itemPopularity

# In[19]:


itemPopularity = np.sort(itemPopularity)
itemPopularity

# In[20]:


pyplot.plot(itemPopularity, 'ro')
pyplot.ylabel('Num Interactions ')
pyplot.xlabel('Item Index')
pyplot.show()

# In[21]:


tenPercent = int(numItems / 10)

print("Average per-item interactions over the whole dataset {:.2f}".
      format(itemPopularity.mean()))

print("Average per-item interactions for the top 10% popular items {:.2f}".
      format(itemPopularity[-tenPercent].mean()))

print("Average per-item interactions for the least 10% popular items {:.2f}".
      format(itemPopularity[:tenPercent].mean()))

print("Average per-item interactions for the median 10% popular items {:.2f}".
      format(itemPopularity[int(numItems * 0.45):int(numItems * 0.55)].mean()))

# In[22]:


print("Number of items with zero interactions {}".
      format(np.sum(itemPopularity == 0)))

# In[23]:


itemPopularityNonzero = itemPopularity[itemPopularity > 0]

tenPercent = int(len(itemPopularityNonzero) / 10)

print("Average per-item interactions over the whole dataset {:.2f}".
      format(itemPopularityNonzero.mean()))

print("Average per-item interactions for the top 10% popular items {:.2f}".
      format(itemPopularityNonzero[-tenPercent].mean()))

print("Average per-item interactions for the least 10% popular items {:.2f}".
      format(itemPopularityNonzero[:tenPercent].mean()))

print("Average per-item interactions for the median 10% popular items {:.2f}".
      format(itemPopularityNonzero[int(numItems * 0.45):int(numItems * 0.55)].mean()))

# In[24]:


pyplot.plot(itemPopularityNonzero, 'ro')
pyplot.ylabel('Num Interactions ')
pyplot.xlabel('Item Index')
pyplot.show()

# ### User activity

# In[25]:


userActivity = (URM_all > 0).sum(axis=1)
userActivity = np.array(userActivity).squeeze()
userActivity = np.sort(userActivity)

pyplot.plot(userActivity, 'ro')
pyplot.ylabel('Num Interactions ')
pyplot.xlabel('User Index')
pyplot.show()


# ### Now that we have the data, we can build our first recommender. We need two things:
# * a 'fit' function to train our model
# * a 'recommend' function that uses our model to recommend
# 
# ### Let's start with a random recommender

# #### In a random recommend we don't have anything to learn from the data

# In[26]:


class RandomRecommender(object):

    def fit(self, URM_train):
        self.numItems = URM_train.shape[0]

    def recommend(self, user_id, at=5):
        recommended_items = np.random.choice(self.numItems, at)

        return recommended_items


# ### In order to evaluate our recommender we have to define:
# * A splitting of the data in URM_train and URM_test
# * An evaluation metric
# * A functon computing the evaluation for each user
# 
# ### The splitting of the data is very important to ensure your algorithm is evaluated in a realistic scenario by using test it has never seen.

# In[27]:


train_test_split = 0.80

numInteractions = URM_all.nnz

train_mask = np.random.choice([True, False], numInteractions, p=[train_test_split, 1 - train_test_split])
train_mask

# In[28]:


userList = np.array(userList)
itemList = np.array(itemList)
ratingList = np.array(ratingList)

URM_train = sps.coo_matrix((ratingList[train_mask], (userList[train_mask], itemList[train_mask])))
URM_train = URM_train.tocsr()
URM_train

# In[29]:


test_mask = np.logical_not(train_mask)

URM_test = sps.coo_matrix((ratingList[test_mask], (userList[test_mask], itemList[test_mask])))
URM_test = URM_test.tocsr()
URM_test

# ### Evaluation metric

# In[30]:


user_id = userList_unique[1]
user_id

# In[31]:


randomRecommender = RandomRecommender()
randomRecommender.fit(URM_train)

recommended_items = randomRecommender.recommend(user_id, at=5)
recommended_items

# #### We call items in the test set 'relevant'

# In[32]:


relevant_items = URM_test[user_id].indices
relevant_items

# In[33]:


is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
is_relevant


# ### Precision: how many of the recommended items are relevant

# In[34]:


def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


# ### Recall: how many of the relevant items I was able to recommend

# In[35]:


def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


# ### Mean Average Precision

# In[36]:


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


# ### And let's test it!

# In[37]:


# We pass as paramether the recommender class

def evaluate_algorithm(URM_test, recommender_object, at=5):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    for user_id in userList_unique:

        relevant_items = URM_test[user_id].indices

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))


# In[38]:


evaluate_algorithm(URM_test, randomRecommender)


# ### So the code works. The performance however...

# # Top Popular recommender
# 
# #### We recommend to all users the most popular items, that is those with the highest number of interactions
# #### In this case our model is the item popularity

# In[39]:


class TopPopRecommender(object):

    def fit(self, URM_train):
        itemPopularity = (URM_train > 0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, user_id, at=5):
        recommended_items = self.popularItems[0:at]

        return recommended_items


# ### Now train and test our model

# In[40]:


topPopRecommender = TopPopRecommender()
topPopRecommender.fit(URM_train)

# In[41]:


for user_id in userList_unique[0:10]:
    print(topPopRecommender.recommend(user_id, at=5))

# In[42]:


evaluate_algorithm(URM_test, topPopRecommender, at=5)