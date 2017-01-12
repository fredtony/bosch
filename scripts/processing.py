# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 22:48:20 2016

@author: Tony

Feature ideas:
X Value Mask
X Total feature counts
X L and S counts
- Cat error codes as bits?
- Categorical -> OHE
- Numerical -> Z-Val
- Date -> difference?
"""

import numpy as np
from scipy import sparse
import pandas as pd
import cPickle
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import StratifiedKFold

with open('date_non_dupe.pkl', 'r') as f:
    date_index = cPickle.load(f)
with open('num_non_dupe.pkl', 'r') as f:
    num_index = cPickle.load(f)
with open('cat_non_dupe.pkl', 'r') as f:
    cat_index = cPickle.load(f)

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape)
             
def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

### L and S lists are lists of all features that have 
with open('LS_indices.pkl', 'r') as f:
    LS_lists = cPickle.load(f)
L_list = LS_lists[0]
S_list = LS_lists[1]
del LS_lists

DATADIR = '../input'
numfile = DATADIR + '/train_numeric.csv'
datefile = DATADIR + '/train_date.csv'
catfile = DATADIR + '/train_categorical.csv'
numfile_test = DATADIR + '/test_numeric.csv'
datefile_test = DATADIR + '/test_date.csv'
catfile_test = DATADIR + '/test_categorical.csv'
num_rows = 1183747
num_features = 4265
LS_counts = np.zeros((num_rows, len(L_list)+len(S_list)+10), dtype='int16')

num = pd.read_csv(numfile, usecols=num_index, dtype='float32').iloc[:,:-1]
date = pd.read_csv(datefile, usecols=date_index, dtype='float32').iloc[:,1:]
cat = pd.read_csv(catfile, usecols=cat_index).iloc[:,1:]

print "Starting processing training data"
### Create masks of whether each item has a value or not as features
num_mask = (~np.isnan(num.values[:,1:])).astype('int16')
num_mask = num_mask.sum(axis=1, dtype='int16').reshape(num_rows,1)
date_mask = (~np.isnan(date.values)).astype('int16')
date_mask = date_mask.sum(axis=1, dtype='int16').reshape(num_rows,1)
cat_mask = (~np.isnan(cat.values)).astype('int16')
cat_mask = cat_mask.sum(axis=1, dtype='int16').reshape(num_rows,1)

### Combine these features together with overall sum and raw num/date data
X_train_cv = np.hstack((num, date, cat, num_mask, date_mask, cat_mask,
    num_mask+date_mask+cat_mask)).astype('float16')
### Modify dates to be relative

### Delete all feature builders
del cat
del num
del date
del num_mask
del date_mask
del cat_mask

### Add in features detailing the counts from each line and stations
### as well as their mutual sums for each line
for i in xrange(4):
    LS_counts[:,i] = np.sum(X_train_cv[:,L_list[i]], axis=1, dtype='int16')
for i in xrange(4,56):
    LS_counts[:,i] = np.sum(X_train_cv[:,S_list[i-4]], axis=1, dtype='int16')
LS_counts[:,56] = LS_counts[:,0] + LS_counts[:,1]
LS_counts[:,57] = LS_counts[:,0] + LS_counts[:,2]
LS_counts[:,58] = LS_counts[:,0] + LS_counts[:,3]
LS_counts[:,59] = LS_counts[:,1] + LS_counts[:,2]
LS_counts[:,60] = LS_counts[:,1] + LS_counts[:,3]
LS_counts[:,61] = LS_counts[:,2] + LS_counts[:,3]
LS_counts[:,62] = LS_counts[:,0] + LS_counts[:,1] + LS_counts[:,2]
LS_counts[:,63] = LS_counts[:,0] + LS_counts[:,1] + LS_counts[:,3]
LS_counts[:,64] = LS_counts[:,0] + LS_counts[:,2] + LS_counts[:,3]
LS_counts[:,65] = LS_counts[:,1] + LS_counts[:,2] + LS_counts[:,3]
X_train_cv = np.hstack((X_train_cv, LS_counts)).astype('float16')
del LS_counts

### Remove cols with all identical values (will add no information to model)
# selector = VarianceThreshold()
# X_train_cv = selector.fit_transform(X_train_cv)

### Add in pre-calculated OHE matrix for cat data (see exploration.py)
cat_ohe_sp = load_sparse_csr('cat_train_ohe_sp.npz').toarray().astype('float16')
X_train_cv = np.hstack((X_train_cv, cat_ohe_sp)).astype('float16')
del cat_ohe_sp

y = pd.read_csv(numfile, usecols=['Response'], dtype='int8', squeeze=True).values

skf = StratifiedKFold(y, 10)
train_idx, cv_idx = next(iter(skf))

np.save('X_train.npy', X_train_cv[train_idx])
np.save('y_train.npy', y[train_idx])
np.save('X_cv.npy', X_train_cv[cv_idx])
np.save('y_cv.npy', y[cv_idx])
# with open('X_train.txt', 'wb') as f:
#     dump_svmlight_file(X_train_cv[train_idx], y[train_idx], f)
# with open('X_cv.txt', 'wb') as f:
#     dump_svmlight_file(X_train_cv[cv_idx], y[cv_idx], f)
del X_train_cv

#########################
### Building X_test #####
#########################
num_rows = 1183748
num_features = 4265
LS_counts = np.zeros((num_rows, len(L_list)+len(S_list)+10), dtype='int16')

num = pd.read_csv(numfile_test, dtype='float16').iloc[:,1:-1]
date = pd.read_csv(datefile_test, dtype='float16').iloc[:,1:]
cat = pd.read_csv(catfile_test, dtype=np.str).iloc[:,1:]

print "Starting processing training data"
num_mask = (~np.isnan(num.values)).astype('int16')
date_mask = (~np.isnan(date.values)).astype('int16')
cat_mask = (~np.isnan(cat.values)).astype('int16')
X_test = np.hstack((num_mask, date_mask, cat_mask)).astype('float16')
num_mask = num_mask.sum(axis=1, dtype='int16').reshape(num_rows,1)
date_mask = date_mask.sum(axis=1, dtype='int16').reshape(num_rows,1)
cat_mask = cat_mask.sum(axis=1, dtype='int16').reshape(num_rows,1)
X_test = np.hstack((X_test, num_mask, date_mask, cat_mask,
    num_mask+date_mask+cat_mask, num.values, date.values)).astype('float16')
del cat
del num
del date
del num_mask
del date_mask
del cat_mask

for i in xrange(4):
    LS_counts[:,i] = np.sum(X_test[:,L_list[i]], axis=1, dtype='int16')
for i in xrange(4,56):
    LS_counts[:,i] = np.sum(X_test[:,S_list[i-4]], axis=1, dtype='int16')
LS_counts[:,56] = LS_counts[:,0] + LS_counts[:,1]
LS_counts[:,57] = LS_counts[:,0] + LS_counts[:,2]
LS_counts[:,58] = LS_counts[:,0] + LS_counts[:,3]
LS_counts[:,59] = LS_counts[:,1] + LS_counts[:,2]
LS_counts[:,60] = LS_counts[:,1] + LS_counts[:,3]
LS_counts[:,61] = LS_counts[:,2] + LS_counts[:,3]
LS_counts[:,62] = LS_counts[:,0] + LS_counts[:,1] + LS_counts[:,2]
LS_counts[:,63] = LS_counts[:,0] + LS_counts[:,1] + LS_counts[:,3]
LS_counts[:,64] = LS_counts[:,0] + LS_counts[:,2] + LS_counts[:,3]
LS_counts[:,65] = LS_counts[:,1] + LS_counts[:,2] + LS_counts[:,3]
X_test = np.hstack((X_test, LS_counts)).astype('float16')
del LS_counts

cat_ohe_sp = load_sparse_csr('cat_test_ohe_sp.npz').toarray().astype('float16')
X_test = np.hstack((X_test, cat_ohe_sp)).astype('float16')
del cat_ohe_sp

selector = VarianceThreshold()
X_test = selector.fit_transform(X_test)


zscores = pd.read_csv(numfile, usecols=range(1, 970), dtype='float16')
with open('mu_sigma_train.pkl', 'r') as f:
    mu_sigma = cPickle.load(f)
zscores = zscores.apply(lambda x: (x-x.mean())/x.std(ddof=0))
X_test = np.hstack((X_test, zscores.values))
del zscores

y = pd.read_csv(numfile, usecols=['Response'], dtype='int8', squeeze=True).values

# np.save('X_test.npy', X_test)
with open('X_test.txt', 'wb') as f:
    dump_svmlight_file(X_test, f)
del X_test
