#!/usr/bin/env python

import csv
import operator
import pickle
import random
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from multiprocessing import Pool

RANDOM_SEED = 228
random.seed(RANDOM_SEED)

with np.load('train.npz') as npz:
    Y_all_train   = npz['Y_all_train']
    Z_index_train = npz['Z_index_train']
    Z_all_train   = npz['Z_all_train']

with np.load('test.npz') as npz:
    Y_all_test    = npz['Y_all_test']
    Z_index_test  = npz['Z_index_test']
    Z_all_test    = npz['Z_all_test']

X_all_train = Z_all_train[Z_index_train[:, 0], :]
X_all_test  = Z_all_test[Z_index_test[:, 0],   :]

train_offsets = Y_all_train - X_all_train
test_offsets  = Y_all_test  - X_all_test

if len(sys.argv) == 2:
    km = KMeans(n_clusters=int(sys.argv[1]), n_jobs=-1, random_state=RANDOM_SEED)
    km.fit_predict(train_offsets)
    pickle.dump(km, open('kmeans.pickle', 'wb'))
    print('Just written the k-means result for k=%d.' % (km.n_clusters))
    sys.exit(0)

kmeans = {}

for k in range(2, 20 + 1):
    kmeans[k] = KMeans(n_clusters=k, n_jobs=-1, random_state=RANDOM_SEED)
    kmeans[k].fit_predict(train_offsets)
    print('k-means for k=%d computed.' % (k))

def evaluate(k):
    km = kmeans[k]
    score = silhouette_score(train_offsets, km.labels_, metric='euclidean', random_state=RANDOM_SEED)
    print('Silhouette score for k=%d is %f.' % (k, score))
    return (k, score)

scores = {}

with open('kmeans-scores.txt', 'w', newline='') as f:
    writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
    writer.writerow(('k', 'silhouette'))
    with Pool(12) as pool:
        for k, score in pool.imap_unordered(evaluate, kmeans):
            scores[k] = score
            writer.writerow((k, score))

k, score = max(scores.items(), key=operator.itemgetter(1))
pickle.dump(kmeans[k], open('kmeans.pickle', 'wb'))
