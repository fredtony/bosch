# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 06:13:41 2016

@author: Tony
"""
import cPickle
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats.mstats import rankdata
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import datetime


DATADIR = '../input'

numfile = DATADIR + '/train_numeric.csv'
datefile = DATADIR + '/train_date.csv'
catfile = DATADIR + '/train_categorical.csv'
magicfile = DATADIR + '/train_magic.csv'
numfile_test = DATADIR + '/test_numeric.csv'
datefile_test = DATADIR + '/test_date.csv'
catfile_test = DATADIR + '/test_categorical.csv'
magicfile_test = DATADIR + '/test_magic.csv'

num_rows = 1183747

with open('date_non_dupe.pkl', 'r') as f:
    date_index = cPickle.load(f)
with open('num_non_dupe.pkl', 'r') as f:
    num_index = cPickle.load(f)
with open('cat_non_dupe.pkl', 'r') as f:
    cat_index = cPickle.load(f)

def save_sparse_csr(filename,array):
    np.savez(filename, data = array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

####################################################
### Part 1: Find column indices for L's and S's ####
####################################################

### Make list of all indices of features
num_bare_idx = pd.read_csv(numfile, nrows=1)
num_bare_idx = num_bare_idx.drop(['Id','Response'], axis=1).columns.tolist()
cat_bare_idx = pd.read_csv(catfile, nrows=1)
cat_bare_idx = cat_bare_idx.drop(['Id'], axis=1).columns.tolist()
date_bare_idx = pd.read_csv(datefile, nrows=1)
date_bare_idx = date_bare_idx.drop(['Id'], axis=1).columns.tolist()

def count_LS_idx(index, save_file):
    LS_lists = [[[],[],[],[]], [[] for i in range(52)]]
    
    ### Make lists of features for each L and S feature to map
    for i in xrange(len(index)):
        split_name = index[i].split('_')
        L = int(split_name[0][1:])
        S = int(split_name[1][1:])
        LS_lists[0][L].append(i)
        LS_lists[1][S].append(i)
        
    with open(save_file, 'w') as f:
        cPickle.dump(LS_lists, f)
    return LS_lists

num_LS_lists = count_LS_idx(num_bare_idx, 'LS_idx_num.pkl')
date_LS_lists = count_LS_idx(date_bare_idx, 'LS_idx_date.pkl')
cat_LS_lists = count_LS_idx(cat_bare_idx, 'LS_idx_cat.pkl')

#########################################################################
### Part 2: Find all values in each column in order to OHE cat files ####
#########################################################################
len_cat_features = len(pd.read_csv(catfile, usecols=cat_index, nrows=1).columns)
tmp_array = pd.read_csv(catfile, usecols=cat_index, dtype=np.str)
tmp_array = tmp_array.append(pd.read_csv(catfile_test, usecols=cat_index, dtype=np.str))
cat_ohe_sp = sparse.csc_matrix(pd.get_dummies(tmp_array, 
                                              columns=tmp_array.columns[1:]),
                               dtype='int8')
del tmp_array

cat_test_ohe_sp = cat_ohe_sp[num_rows:,:]
cat_ohe_sp = cat_ohe_sp[0:num_rows,:]

save_sparse_csr('cat_train_ohe_sp.npz', cat_ohe_sp)
save_sparse_csr('cat_test_ohe_sp.npz', cat_test_ohe_sp)
del cat_test_ohe_sp
del cat_ohe_sp
    
##############################################################################
### Part 3: Find mean/stdev of each col in numerical to convert to Z-score ###
##############################################################################

### First find mean and std dev of each column in train
len_num_features = len(pd.read_csv(numfile, usecols=num_index, nrows=1).columns)
    
def get_mean_stdev_list(csv, num_cols, save_file_name):
    ### Finds mean & standard deviation of each column in csv file
    len_features = len(pd.read_csv(csv, nrows=1).columns)
    mu_sigma = [[],[]]
    for i in xrange(len_features//num_cols):
        print "{} features processed".format(i*num_cols)
        col_i = pd.read_csv(csv, usecols=range(num_cols*i+1, num_cols*i+num_cols+1)).describe()
        mu_sigma[0] += list(col_i.iloc[1,:])
        mu_sigma[1] += list(col_i.iloc[2,:])
    col_i = pd.read_csv(csv, usecols=range(num_cols*i+num_cols+1, len_features-1)).describe()
    mu_sigma[0] += list(col_i.iloc[1,:])
    mu_sigma[1] += list(col_i.iloc[2,:])
    with open(save_file_name, 'w') as f:
        cPickle.dump(mu_sigma, f)
    return mu_sigma

num_cols=40
mu_sigma_num_train = get_mean_stdev_list(numfile, num_cols, 'mu_sigma_num_train.pkl')
mu_sigma_num_test = get_mean_stdev_list(numfile_test, num_cols, 'mu_sigma_num_test.pkl')

diff_mean_stdev = [[],[]]
for i in xrange(len(mu_sigma_num_train[0])):
    diff_mean_stdev[0].append(mu_sigma_num_train[0][i] - mu_sigma_num_test[0][i])
    diff_mean_stdev[1].append(mu_sigma_num_train[1][i] - mu_sigma_num_test[1][i])

### Convert each column to z-score and save as sparse matrices

# col_i = pd.read_csv(numfile, usecols=range(1, 970), dtype='float16')
# col_i = col_i.apply(lambda x: (x-x.mean())/x.std(ddof=0))

    
# col_i = pd.read_csv(numfile_test, usecols=[1], squeeze=True)
# num_test_sp = (col_i - mu_sigma[0][0]) / mu_sigma[0][1]
# for i in xrange(1,len_num_features):
#     if i % 100 == 0:
#         print "{} features processed".format(i)
#     col_i = pd.read_csv(numfile_test, usecols=[i+1], squeeze=True)
#     num_test_sp = sparse.hstack((num_test_sp, (col_i - mu_sigma[i][0]) / mu_sigma[i][1]),
#                                  format = 'csc', dtype='int16')

# num_test_sp = num_test_sp.tocsr()
# save_sparse_csr('num_test_sp.npz', num_test_sp)
# del num_test_sp

#########################################################################
### Part 4: Explore how dates progress in the data set               ####
#########################################################################
len_date_features = len(pd.read_csv(datefile, usecols=date_index, nrows=1).columns)
total_rows = 1183747
nrows = 20000
def find_range(x, axis=0):
    return np.nanmax(x, axis=axis) - np.nanmin(x, axis=axis)
    
date_index_noid = list(date_index)
date_index_noid.remove('Id')
LS_lists_date = [[[],[],[],[]], [[] for i in range(52)]]

### Make lists of features for each L and S feature to map
for i in xrange(len(date_index_noid)):
    split_name = date_index_noid[i].split('_')
    L = int(split_name[0][1:])
    S = int(split_name[1][1:])
    F = int(split_name[2][1:])
    LS_lists_date[0][L].append(i)
    LS_lists_date[1][S].append(i)
L = LS_lists_date[0]
S = LS_lists_date[1]

with open('LS_indices_date.pkl', 'w') as f:
    cPickle.dump(LS_lists_date, f)
del LS_lists_date

### Use when RAM is limited on home computer
### Does not include actual date files
def featurize_date_part(nrows, start=0):
    if start == 0:
        date = pd.read_csv(datefile, usecols=date_index, nrows=nrows, dtype='float32')
    else:
        date = pd.read_csv(datefile, usecols=date_index, nrows=nrows,
                           skiprows=range(1,start), dtype='float32')
    
    date_names = date.columns[1:]

    date['L0_min'] = date.iloc[:,1:][L[0]].min(axis=1, skipna=True)
    date['L1_min'] = date.iloc[:,1:][L[1]].min(axis=1, skipna=True)
    date['L2_min'] = date.iloc[:,1:][L[2]].min(axis=1, skipna=True)
    date['L3_min'] = date.iloc[:,1:][L[3]].min(axis=1, skipna=True)
    
    date['L0_max'] = date.iloc[:,1:][L[0]].max(axis=1, skipna=True)
    date['L1_max'] = date.iloc[:,1:][L[1]].max(axis=1, skipna=True)
    date['L2_max'] = date.iloc[:,1:][L[2]].max(axis=1, skipna=True)
    date['L3_max'] = date.iloc[:,1:][L[3]].max(axis=1, skipna=True)
#    
#    date['L0_mean'] = date.iloc[:,1:][L[0]].mean(axis=1, skipna=True)
#    date['L1_mean'] = date.iloc[:,1:][L[1]].mean(axis=1, skipna=True)
#    date['L2_mean'] = date.iloc[:,1:][L[2]].mean(axis=1, skipna=True)
#    date['L3_mean'] = date.iloc[:,1:][L[3]].mean(axis=1, skipna=True)
#    
#    date['L0_median'] = date.iloc[:,1:][L[0]].median(axis=1, skipna=True)
#    date['L1_median'] = date.iloc[:,1:][L[1]].median(axis=1, skipna=True)
#    date['L2_median'] = date.iloc[:,1:][L[2]].median(axis=1, skipna=True)
#    date['L3_median'] = date.iloc[:,1:][L[3]].median(axis=1, skipna=True)
#    
#    date['L0_range'] = date['L0_max'] - date['L0_min']
#    date['L1_range'] = date['L1_max'] - date['L1_min']
#    date['L2_range'] = date['L2_max'] - date['L2_min']
#    date['L3_range'] = date['L3_max'] - date['L3_min']
    
    date['L0_L1_gap'] = date['L1_min'] - date['L0_max']
    date['L0_L2_gap'] = date['L2_min'] - date['L0_max']
    date['L0_L3_gap'] = date['L3_min'] - date['L0_max']
    
    date['L1_L2_gap'] = date['L2_min'] - date['L1_max']
    date['L1_L3_gap'] = date['L3_min'] - date['L1_max']
    date['L2_L3_gap'] = date['L3_min'] - date['L2_max']
    
    date['MIN'] = date[date_names].min(axis=1, skipna=True)
    date['MAX'] = date[date_names].max(axis=1, skipna=True)
    date['RANGE'] = date['MAX'] - date['MIN']
#    date['MEAN'] = date[date_names].mean(axis=1, skipna=True)
#    date['MEDIAN'] = date[date_names].median(axis=1, skipna=True)
    date['STDEV'] = date[date_names].std(axis=1, skipna=True)

#    selector = VarianceThreshold()
#    date = selector.fit_transform(date)
    
    #date.iloc[:,1:len_date_features+1] = date.iloc[:,1:len_date_features+1]-date['MEAN']
    
#    if start == 0:
#        date['Response'] = pd.read_csv(numfile, usecols=['Response'], nrows=nrows,
#                                       dtype='int8', squeeze=True)
#    else:
#        date['Response'] = pd.read_csv(numfile, nrows=nrows, usecols=['Response'],
#                                skiprows=range(1,start), dtype='int8', squeeze=True)
    
    date.drop(date_names, axis=1, inplace=True)
    return date

date = featurize_date_part(nrows, start=0)
for i in tqdm.trange(1,total_rows//nrows+1):
    if i == (total_rows//nrows):
        date = pd.concat([date, featurize_date_part(total_rows%nrows+1, start=i*nrows)])
    else:
        date = pd.concat([date, featurize_date_part(nrows, start=i*nrows)])
date.drop_duplicates(inplace=True)


##############################################################
### Used for instance with enough memory to load whole set ###
##############################################################
date = pd.read_csv(datefile, usecols=date_index, dtype='float32')
date_names = date.columns[1:]

date['L0_min'] = date.iloc[:,1:][L[0]].min(axis=1, skipna=True)
date['L1_min'] = date.iloc[:,1:][L[1]].min(axis=1, skipna=True)
date['L2_min'] = date.iloc[:,1:][L[2]].min(axis=1, skipna=True)
date['L3_min'] = date.iloc[:,1:][L[3]].min(axis=1, skipna=True)

date['L0_max'] = date.iloc[:,1:][L[0]].max(axis=1, skipna=True)
date['L1_max'] = date.iloc[:,1:][L[1]].max(axis=1, skipna=True)
date['L2_max'] = date.iloc[:,1:][L[2]].max(axis=1, skipna=True)
date['L3_max'] = date.iloc[:,1:][L[3]].max(axis=1, skipna=True)

date['L0_mean'] = date.iloc[:,1:][L[0]].mean(axis=1, skipna=True)
date['L1_mean'] = date.iloc[:,1:][L[1]].mean(axis=1, skipna=True)
date['L2_mean'] = date.iloc[:,1:][L[2]].mean(axis=1, skipna=True)
date['L3_mean'] = date.iloc[:,1:][L[3]].mean(axis=1, skipna=True)

date['L0_median'] = date.iloc[:,1:][L[0]].median(axis=1, skipna=True)
date['L1_median'] = date.iloc[:,1:][L[1]].median(axis=1, skipna=True)
date['L2_median'] = date.iloc[:,1:][L[2]].median(axis=1, skipna=True)
date['L3_median'] = date.iloc[:,1:][L[3]].median(axis=1, skipna=True)

date['L0_range'] = date['L0_max'] - date['L0_min']
date['L1_range'] = date['L1_max'] - date['L1_min']
date['L2_range'] = date['L2_max'] - date['L2_min']
date['L3_range'] = date['L3_max'] - date['L3_min']

date['L0_L1_gap'] = date.iloc[:,1:][L[1]].min(axis=1, skipna=True) - \
                    date.iloc[:,1:][L[0]].max(axis=1, skipna=True)
date['L0_L2_gap'] = date.iloc[:,1:][L[2]].min(axis=1, skipna=True) - \
                    date.iloc[:,1:][L[0]].max(axis=1, skipna=True)
date['L0_L3_gap'] = date.iloc[:,1:][L[3]].min(axis=1, skipna=True) - \
                    date.iloc[:,1:][L[0]].max(axis=1, skipna=True)

date['L1_L2_gap'] = date.iloc[:,1:][L[2]].min(axis=1, skipna=True) - \
                    date.iloc[:,1:][L[1]].max(axis=1, skipna=True)
date['L1_L3_gap'] = date.iloc[:,1:][L[3]].min(axis=1, skipna=True) - \
                    date.iloc[:,1:][L[1]].max(axis=1, skipna=True)
date['L2_L3_gap'] = date.iloc[:,1:][L[3]].min(axis=1, skipna=True) - \
                    date.iloc[:,1:][L[2]].max(axis=1, skipna=True)

date['MIN'] = date[date_names].min(axis=1, skipna=True)
date['MAX'] = date[date_names].max(axis=1, skipna=True)
date['RANGE'] = date['MAX'] - date['MIN']
date['MEAN'] = date[date_names].mean(axis=1, skipna=True)
date['MEDIAN'] = date[date_names].median(axis=1, skipna=True)
date['STDEV'] = date[date_names].std(axis=1, skipna=True)

selector = VarianceThreshold()
date = selector.fit_transform(date)

#date.iloc[:,1:len_date_features+1] = date.iloc[:,1:len_date_features+1]-date['MEAN']

# date['Response'] = pd.read_csv(numfile, usecols=['Response'], dtype='int8', squeeze=True)

skf = StratifiedKFold(y_train_cv, n_folds=10)
train_idx, cv_idx = next(iter(skf))
y_train_cv = pd.read_csv(numfile, usecols=['Response'], dtype='float16', squeeze=True).values

xgb_params = {
"objective": "binary:logistic",
"booster": "gbtree",
"max_depth": 15,
"eval_metric": "auc",
"eta": 0.07,
"silent": 1,
"lambda": 0.3,
}
num_round = 200
early_stopping_rounds = 100
dtrain = xgb.DMatrix(date[train_idx], y_train_cv[train_idx])
dcv = xgb.DMatrix(date[cv_idx], y_train_cv[cv_idx])
watchlist = [(dtrain,'train'), (dcv,'cv')]
bst = xgb.train(xgb_params, dtrain, num_round, watchlist, maximize=True)



######################################
################### Plotting##########
######################################

import gc

date_features = date.columns.tolist()
date_features.remove('Id')
date_features.remove('Response')
date_features.remove('MIN')
date_features.remove('Unnamed: 0')
fig = plt.Figure()
for elem in date_features:
    if date[elem].count() > 0:
        g=sns.swarmplot(x='MIN', y=elem, hue='Response', data=date)
        fig=g.get_figure()
        fig.savefig('./images/explore/num_ordered_explore_{}.png'.format(elem))
        plt.close(fig)
        plt.close('all')
        gc.collect()
    
num_features = num.columns.tolist()
num_features.remove('Id')
num_features.remove('Response')
for elem in num_features:
    g = sns.FacetGrid(num[[elem,'Response']], hue='Response', size=10)
    g.map(sns.kdeplot, elem, shade=True).add_legend()
    g.savefig('./images/numtrain_{}.png'.format(elem))
    plt.close()
    
### After adding time features, plot as a time series
num_features = num.columns.tolist()
num_features.remove('Id')
num_features.remove('Response')
num_features.remove('MIN')
fig = plt.Figure()
for elem in num_features:
    sns.palplot(sns.color_palette('bright',2))
    g = sns.FacetGrid(num[[elem,'MIN','Response']], hue='Response', size=10)
    g.map(plt.scatter, 'MIN', elem).add_legend()
    g.savefig('./images/numtrain_time_{}.png'.format(elem))
    plt.close('all')


##############################
### Find column duplicates ###
##############################
def remove_dupes_col(df, col, cols_to_check):
    return [elem for elem in cols_to_check if df[col].equals(df[elem])]

def remove_dupes(df, skip=['Id']):
    cols_non_dupe = df.columns.tolist()
    for elem in skip:
        cols_non_dupe.remove(elem)
    cols_dupe = []
    i=0
    while i < len(cols_non_dupe)-1:
        print('i={}'.format(i))
        cols_to_remove = remove_dupes_col(df, cols_non_dupe[i], cols_non_dupe[i+1:])
        for elem in cols_to_remove:
            cols_non_dupe.remove(elem)
            cols_dupe.append(elem)
        i += 1
    return cols_non_dupe, cols_dupe

num = pd.concat([pd.read_csv(numfile), pd.read_csv(numfile_test)])
num_non_dupe, num_dupe = remove_dupes(num)
#del num
with open('num_dupe.pkl','w') as f:
    cPickle.dump(num_dupe, f)
with open('num_non_dupe.pkl','w') as f:
    cPickle.dump(num_non_dupe, f)

date = pd.concat([pd.read_csv(datefile), pd.read_csv(datefile_test)])
date_non_dupe, date_dupe = remove_dupes(date)
#del date
with open('date_dupe.pkl','w') as f:
    cPickle.dump(date_dupe, f)
with open('date_non_dupe.pkl','w') as f:
    cPickle.dump(date_non_dupe, f)

cat = pd.concat([pd.read_csv(catfile), pd.read_csv(catfile_test)])
cat_non_dupe, cat_dupe = remove_dupes(cat)
#del cat
with open('cat_dupe.pkl','w') as f:
    cPickle.dump(cat_dupe, f)
with open('cat_non_dupe.pkl','w') as f:
    cPickle.dump(cat_non_dupe, f)

######################################
### Create features related to S38 ###
######################################

test=pd.read_csv(datefile_test, usecols=date_index, dtype=np.float32)
test = test[date_index]
test['L3_S38_F3952'] = pd.read_csv(numfile_test, usecols=['L3_S38_F3952'],
                                   squeeze=True, dtype=np.float32)
test['Response'] = np.nan
test['MIN']=test.iloc[:,1:-3].min(axis=1)
test['MAX']=test.iloc[:,1:-4].max(axis=1)
test['RANGE'] = test.MAX-test.MIN
test['COUNT'] = test.count(axis=1)

train = pd.read_csv(datefile, usecols=date_index, dtype=np.float32)
train = train[date_index]
train['L3_S38_F3952'] = pd.read_csv(numfile, usecols=['L3_S38_F3952'],
                                    squeeze=True, dtype=np.float32)
train['Response']=pd.read_csv(numfile, usecols=['Response'], squeeze=True, dtype=np.int8)
train['MIN']=train.iloc[:,1:-3].min(axis=1)
train['MAX']=train.iloc[:,1:-4].max(axis=1)
train['RANGE'] = train.MAX-train.MIN
train['COUNT'] = train.count(axis=1)
train_test = pd.concat([train, test], ignore_index=True)

train_test.sort_values(train_test.columns.tolist()[1:-7], inplace=True)
train_test.reset_index(inplace=True)
train_test['ORDER'] = train_test.index
train_test['L3_S38_D3953_shift_back'] = train_test.L3_S38_D3953.shift(-1)
train_test['L3_S38_F3952_shift_back'] = train_test.L3_S38_F3952.shift(-1)
train_test['38_mask_back'] = ~(train_test.L3_S38_D3953_shift_back.isnull()).astype(np.int8)
train_test['same_min_next'] = (train_test.MIN == train_test.MIN.shift(-1))\
                                .fillna(False).astype(np.int8)
train_test['same_max_next'] = (train_test.MAX == train_test.MAX.shift(-1))\
                                .fillna(False).astype(np.int8)
                                
train_test['L3_S38_D3953_shift_ahead'] = train_test.L3_S38_D3953.shift(1)
train_test['L3_S38_F3952_shift_ahead'] = train_test.L3_S38_F3952.shift(1)
train_test['38_mask_ahead'] = ~(train_test.L3_S38_D3953_shift_ahead.isnull()).astype(np.int8)
train_test['same_min_last'] = (train_test.MIN == train_test.MIN.shift(1))\
                                .fillna(False).astype(np.int8)
train_test['same_max_last'] = (train_test.MAX == train_test.MAX.shift(1))\
                                .fillna(False).astype(np.int8)

train = train_test[~train_test.Response.isnull()]
train.to_csv('train_date_mod.csv', index=False)
test = train_test[train_test.Response.isnull()].drop(['Response'], axis=1)
test.to_csv('test_date_mod.csv', index=False)

### The above gives inaccurate information for the shift and so the resulting
### algorithm is severely overfitting


#class heirarchical_sorted_df(object):
#    '''
#    This class takes a pandas DataFrame that has some time order in all
#    of the columns cols. However, many columns 
#    have gaps (nan) and simple sorts (i.e. forward order, backwards order, most
#    common first) corrupt the order of other columns. All of the columns 
#    start unordered and none are considered ordered. Finding the correct order
#    of the rows will provide significant predictive power, so this class helps
#    to finds the order that the columns should be sorted in.
#    
#    order: Finds the order of columns and outputs the order if one exists.
#    
#    col_ordered: Checks if one particular column of df is ordered and returns
#    a boolean.
#    
#    df_not_ordered_list: For particular columns, checks if they are all
#    correctly sorted and if not, returns the columns that are not sorted.
#    
#    order_df: The workhorse of the class. This takes a current order (ordered)
#    with columns that are still unordered and returns the order if it exists.
#    Uses recursion on the list names (using OOP to ensure the large dataframe
#    does not have to be copied reeatedly and did not want to risk globals)
#    
#    '''
#    def __init__(self, df, cols):
#        self.df = df
#        self.ordered = []
#        self.unordered = cols[:]
#        self.order_check = []
#        self.unordered_check = []
#        self.iter = 0
#        return
#    
#    def col_ordered(self, col):
#        arr = self.df[col].dropna().values
#        return np.all(arr[:-1] <= arr[1:])
#    
#    def df_not_ordered_list(self, cols):
#        not_ordered = []
#        for col in cols:
#            if not self.col_ordered(col):
#                not_ordered.append(col)
#        return not_ordered
#    
#    def order_df(self, ordered_list, unordered_list):
#        ordered = ordered_list[:]
#        unordered = unordered_list[:]
#        if len(unordered) == 1:
#            self.iter += 1
#            if self.iter % 50 == 0:
#                print(ordered)
#            ordered.append(unordered[0])
#            self.df.sort_values(ordered, inplace=True)
#            if self.df_not_ordered_list(ordered) == []:
#                return ordered
#            else:
#                return []
#        else:
#            for col in unordered:
#                self.iter += 1
#                if self.iter % 50 == 0:
#                    print(ordered)
#                ordered.append(col)
#                self.order_check.append(ordered[:])
#                self.df.sort_values(ordered, inplace=True)
#                if self.df_not_ordered_list(ordered) == []:
#                    unordered.remove(col)
#                    new_ordered = self.order_df(ordered, unordered)
#                    if new_ordered == []:
#                        unordered.append(ordered.pop())
#                    else:
#                        return new_ordered
#                else:
#                    ordered.remove(col)
#            return []
#                
#    def order(self):
#        if self.unordered == []:
#            print("Already ordered!")
#        else:
#            self.ordered = self.order_df([], self.unordered)
#            if self.ordered != []:
#                print("Ordered!")                
#                self.unordered = []
#            else:
#                print("Unable to be ordered!")
#            return self.ordered
#
#sorted_train_test = heirarchical_sorted_df(train_test, date_index)
#sorted_train_test.order()
#
#
#def row_ordered(row):
#    row = np.dropna(row)
#    return np.all(row[:-1] <= row[1:])
#
#train_test_sort = train_test[date_index[1:]].copy()
#train_test_sort['COUNT'] = train_test_sort.count(1)
#train_test_sort['ORDER'] = train_test_sort.apply(lambda x: np.argsort(x), raw=True)
