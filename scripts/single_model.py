# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 23:38:09 2016

@author: Tony
"""

import datetime
import cPickle
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from numba import jit
import xgboost as xgb
import copy
#import matplotlib.pyplot as plt

with open('date_non_dupe.pkl', 'r') as f:
    date_index = cPickle.load(f)
with open('num_non_dupe.pkl', 'r') as f:
    num_index = ['Id'] + cPickle.load(f)
with open('cat_non_dupe.pkl', 'r') as f:
    cat_index = cPickle.load(f)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def out_from_probs(train_ids, cv_pred, test_ids, test_pred, best_proba,
                   folds, best_mcc, clf_type):
    now = datetime.datetime.now()
    
    pred = pd.DataFrame({'Id': test_ids, 'Response': pd.Series(test_pred)})
    pred_file = 'preds_{}fold-keras-{:.4f}-{}.csv.gz'.format(
        folds, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing test probabilities: {}".format(pred_file))
    pred.to_csv(pred_file, index=False, compression='gzip')
    
    oof_pred = pd.DataFrame({'Id': train_ids, 'Response': pd.Series(cv_pred)})
    oof_pred_file = 'oof_preds_{}fold-xgb-{:.4f}-{}.csv.gz'.format(
        folds, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing oof probabilities: {}".format(oof_pred_file))
    oof_pred.to_csv(oof_pred_file, index=False, compression='gzip')
    
    test_pred = (test_pred >= best_proba).astype(int)
    
    result = pd.DataFrame({'Id': test_ids, 'Response': pd.Series(test_pred)})
    sub_file = 'submission_{}fold-xgb-{:.4f}-{}.csv.gz'.format(
        folds, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing submission: {}".format(sub_file))
    result.to_csv(sub_file, index=False, compression='gzip')
    return

@jit
def eval_mcc(y_true, y_prob):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)
    numn = n - nump
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    prev_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            sup = tp * tn - fp * fn
            inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            if inf==0:
                new_mcc = 0
            else:
                new_mcc = float(sup) / float(np.sqrt(inf))
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    return best_mcc
        
def eval_mcc2(y_true, y_prob):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)
    numn = n - nump
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            sup = tp * tn - fp * fn
            inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            if inf==0:
                new_mcc = 0
            else:
                new_mcc = float(sup) / float(np.sqrt(inf))
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    y_pred = (y_prob >= best_proba).astype(int)
    score = matthews_corrcoef(y_true, y_pred)
    print(score, best_mcc)
#    plt.plot(mccs)
    return best_proba, best_mcc, y_pred

def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc
        
DATADIR = '../input'
numfile = DATADIR + '/train_numeric.csv'
datefile = DATADIR + '/train_date.csv'
catfile = DATADIR + '/train_categorical.csv'
magicfile = DATADIR + '/train_magic.csv'
numfile_test = DATADIR + '/test_numeric.csv'
datefile_test = DATADIR + '/test_date.csv'
catfile_test = DATADIR + '/test_categorical.csv'
magicfile_test = DATADIR + '/test_magic.csv'

train = pd.read_csv('train_date_mod.csv',dtype=np.float32)
y = train.loc[:,'Response']
train.drop('Response',1,inplace=True)
num = pd.read_csv(numfile, dtype=np.float32, usecols=num_index[:-1])
train = train.merge(num, on='Id', how='left')
del num
cat = pd.read_csv(catfile, dtype=np.float32, usecols=cat_index)
train = train.merge(cat, on='Id', how='left')
del cat
magic = pd.read_csv(magicfile)
magic['Id'] = pd.read_csv(datefile, usecols=['Id'], squeeze=True)
magic.drop(['StationPath','LinePath'], 1, inplace=True)
magic.loc[:,'StationStartEndSame0':'StationStartEndSame51'] =\
    magic.loc[:,'StationStartEndSame0':'StationStartEndSame51'].astype(np.float32)
train = train.merge(magic, on='Id', how='left')
del magic
#y = y.values
#train_ids = train.iloc[:,0].values
#train = np.nan_to_num(train.as_matrix().astype(np.float32))

test = pd.read_csv('test_date_mod.csv',dtype=np.float32)
num = pd.read_csv(numfile_test, dtype=np.float32, usecols=num_index[:-1])
test = test.merge(num, on='Id', how='left')
del num
cat = pd.read_csv(catfile_test, dtype=np.float32, usecols=cat_index)
test = test.merge(cat, on='Id', how='left')
del cat
magic = pd.read_csv(magicfile_test)
magic['Id'] = pd.read_csv(datefile_test, usecols=['Id'], squeeze=True)
magic.drop(['StationPath','LinePath'], 1, inplace=True)
magic.loc[:,'StationStartEndSame0':'StationStartEndSame51'] =\
    magic.loc[:,'StationStartEndSame0':'StationStartEndSame51'].astype(np.float32)
test = test.merge(magic, on='Id', how='left')
del magic
#test_ids = test.iloc[:,0].values
#test = np.nan_to_num(test.as_matrix().astype(np.float32))


# X_train_cv = np.nan_to_num(X_train_cv)
# num = pd.read_csv(numfile, dtype='float16').iloc[:,1:-1]
# date = pd.read_csv(datefile, dtype='float16').iloc[:,1:]
# cat = load_sparse_csr('cat_train_ohe_sp.npz').toarray().astype('float16')
# X_train_cv = np.hstack(num, date, cat)
# del num
# del date
# del cat

#del X_train_cv
#del y_train_cv
#dcv = xgb.DMatrix(X_cv, label=y_cv)
#del X_cv
#del y_cv
#dtrain = xgb.DMatrix(train, label=y)
#del X_train
#del y_train

def eval_xgb(train, y, test, nfolds=5, random_state=0):
    xgb_params = {
    "base_score": 0.005,
    "booster": "gbtree",
    "objective": "binary:logistic",
    "max_depth": 10,
    "subsample": 0.8,
    #"eval_metric": 'auc',
    "eta": 0.01,
    "min_child_weight": 3,
    "silent": 1,
    "lambda": 3.0,
    }
    num_round = 1000
    early_stopping_rounds = 20
    #nfolds = nfolds
    #watchlist = [(dtrain,'train'), (dcv,'cv')]
    #bst = xgb.train(xgb_params, dtrain, num_round, watchlist, maximize=True)
    
    skf = StratifiedKFold(y, n_folds=nfolds, shuffle=True, random_state=random_state)
    y_pred = np.zeros((1183748, len(skf)))
    y_oof_pred = np.zeros(y.shape)
    xgb_models = []
    best_probas = np.zeros(len(skf))
    best_mccs = np.zeros(len(skf))
    dtest = xgb.DMatrix(test)
    for i, (train_idx, cv_idx) in enumerate(skf):
        dtrain = xgb.DMatrix(train.iloc[train_idx,:], y[train_idx])
        dcv = xgb.DMatrix(train.iloc[cv_idx,:], y[cv_idx])
        watchlist = [(dtrain,'train'), (dcv,'cv')]
        bst = xgb.train(xgb_params, dtrain, num_round, watchlist, maximize=True, feval=mcc_eval,
                        verbose_eval=5, early_stopping_rounds=early_stopping_rounds)
        y_oof_pred[cv_idx] = bst.predict(dcv)
        best_probas[i], best_mccs[i], _ = eval_mcc2(y[cv_idx].values, y_oof_pred[cv_idx])
        print("Fold {} mcc: {}".format(i+1, best_mccs[i]))
        y_pred[:,i] = bst.predict(dtest)
        xgb_models.append(bst)
    del dtrain
    del dcv
    #del train
    
    #y_pred = np.zeros((1183748, len(skf)))
    #for i in xrange(len(skf)):
    #    y_pred[:,i] = xgb_models[i].predict(dtest)
    y_pred = np.average(y_pred, 1)
    
    best_proba, best_mcc, _ = eval_mcc2(y.values, y_oof_pred)
    print("Out of fold (predicted LB) CV mcc: {}".format(best_mcc))
    
    now = datetime.datetime.now()
    
    pred = pd.DataFrame({'Id': test['Id'].astype(np.int32), 'Response': pd.Series(y_pred)})
    pred_file = 'preds_{}fold-xgb-{:.4f}-{}.csv.gz'.format(
        nfolds, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing test probabilities: {}".format(pred_file))
    pred.to_csv(pred_file, index=False, compression='gzip')
    
    oof_pred = pd.DataFrame({'Id': train['Id'].astype(np.int32), 'Response': pd.Series(y_oof_pred)})
    oof_pred_file = 'oof_preds_{}fold-xgb-{:.4f}-{}.csv.gz'.format(
        nfolds, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing oof probabilities: {}".format(oof_pred_file))
    oof_pred.to_csv(oof_pred_file, index=False, compression='gzip')
    
    y_pred = (y_pred >= best_proba).astype(int)
    
    result = pd.DataFrame({'Id': test['Id'].astype(np.int32), 'Response': pd.Series(y_pred)})
    sub_file = 'submission_{}fold-xgb-{:.4f}-{}.csv.gz'.format(
        nfolds, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing submission: {}".format(sub_file))
    result.to_csv(sub_file, index=False, compression='gzip')

eval_xgb(train, y, test, nfolds=10, random_state=45)
   

def eval_clf(clf, train_cv, y, test, train_ids, test_ids, clf_type='clf', nfolds=10, random_state=0):
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    test_pred = np.zeros((test.shape[0], nfolds))
    cv_pred = np.zeros(y.shape)
    clfs = []
    best_probas = np.zeros(nfolds)
    best_mccs = np.zeros(nfolds)
    for i, (train_idx, cv_idx) in enumerate(skf.split(train_cv, y)):
        clfs.append(copy.deepcopy(clf))
        train, y_train = train_cv[train_idx,:], y[train_idx]
        cv, y_cv = train_cv[cv_idx,:], y[cv_idx]
        clfs[i].fit(train, y_train)
        cv_pred[cv_idx] = clfs[i].predict_proba(cv)[:,1]
        best_probas[i], best_mccs[i], _ = eval_mcc2(y_cv, cv_pred[cv_idx])
        print("Fold {} mcc: {}".format(i+1, best_mccs[i]))
        test_pred[:,i] = clfs[i].predict_proba(test)[:,1]
    del train, cv
 
    test_pred = np.average(test_pred, 1)
    
    best_proba, best_mcc, _ = eval_mcc2(y, cv_pred)
    print("Out of fold (predicted LB) CV mcc: {}".format(best_mcc))
    
    out_from_probs(train_ids, cv_pred, test_ids, test_pred, best_proba,
                   nfolds, best_mcc, clf_type)
                   
#clf = LogisticRegression()
#clf = RandomForestClassifier(n_estimators=40, criterion='gini', n_jobs=-1, random_state=0)
#clf = RandomForestClassifier(n_estimators=40, criterion='entropy', n_jobs=-1, random_state=0)
#clf = ExtraTreesClassifier(n_estimators=40, criterion='gini', n_jobs=-1, random_state=0)
#clf = ExtraTreesClassifier(n_estimators=100, criterion='entropy', n_jobs=-1, random_state=0)
#clf = GradientBoostingClassifier()
#clf = AdaBoostClassifier(random_state=0)
clf = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
#clf = MLPClassifier(hidden_layer_sizes=(400,100,20), random_state=0, verbose=True)
eval_clf(clf, train, y, test, train_ids, test_ids, 'KNN', 10, 0)


############################################################
###################### Neural Nets #########################
############################################################

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.optimizers import Adadelta
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
import theano.tensor as T

def eval_mcc_keras(y_true, y_prob):
    idx = T.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape
    nump = 1.0 * T.sum(y_true)
    numn = n - nump
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    prev_proba = -1
    mccs = T.zeros(n)
    for i in range(n):
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            sup = tp * tn - fp * fn
            inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            if inf==0:
                new_mcc = 0
            else:
                new_mcc = sup / T.sqrt(inf)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    return best_mcc

def create_model(n_input, nodes=[50], reg=1.0, dropouts=[.5], acts=['relu']):
    n_in = n_input    
    model = Sequential()
    for i in xrange(len(nodes)):
        n_out = nodes[i]
        dropout = dropouts[i]
        act = acts[i]
        model.add(Dense(output_dim=n_out, input_dim=n_in, init='he_normal', W_regularizer=l2(reg)))
        model.add(act)
        model.add(Dropout(dropout))
        n_in = n_out
    model.add(Dense(output_dim=1, init='he_normal', W_regularizer=l2(reg)))
    model.add(Activation("softmax"))
    # Compile model
    adadelta = Adadelta(lr=10.0, rho=0.95, epsilon=1e-08)
    sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=adadelta, metrics=[eval_mcc_keras])
    return model

def keras_eval(X_train_cv, y, X_test, train_ids, test_ids, folds=5, nbags=2,
               nepochs=55, batch_size=500, random_state=123):
    np.random.seed(random_state)
    
    # set up KFold that matches xgb.cv number of folds
    cv_pred = np.zeros((X_train_cv.shape[0], folds, nbags))
    test_pred = np.zeros((X_test.shape[0], folds, nbags))
    cv_score = np.zeros((folds, nbags))
    skf = StratifiedKFold(y, n_folds=folds, shuffle=True, random_state=random_state)
    for i, (train_index, cv_index) in enumerate(skf):
        X_train, X_cv = X_train_cv[train_index,:], X_train_cv[cv_index,:]
        y_train, y_cv = y[train_index], y[cv_index]

        ## train models
        for j in range(nbags):
            model = create_model(X_train.shape[1], nodes=[200,20], reg=2.0,
                                 dropouts=[.4,.2], acts=[PReLU(),PReLU()])
            callback = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
            model.fit(X_train, y_train, nb_epoch=nepochs, batch_size=batch_size,
                      callbacks=[callback], validation_data=(X_cv,y_cv), verbose=1)
            cv_pred[cv_index,i,j] = model.predict(X_cv, batch_size=batch_size, verbose=1)\
                                    .reshape((X_cv.shape[0],))
            test_pred[:,i,j] = model.predict(X_test, batch_size=batch_size, verbose=1)\
                                    .reshape((X_test.shape[0],))
            best_proba, best_mcc, y_pred = eval_mcc2(y_cv, cv_pred[cv_index,i,j])
            cv_score[i,j] = best_mcc
            print('Fold {}, Bag {} - MCC: {}'.format(i, j, cv_score[i,j]))
        print(' Fold {} - MCC: {}\n'.format(i, cv_score.mean(1)[i]))
    best_proba, best_mcc, y_pred = eval_mcc2(y, cv_pred.mean(2).mean(1))
    score = best_mcc
    print('Total - MCC: {}'.format(score))
    
    out_from_probs(train_ids, cv_pred, test_ids, test_pred, best_proba,
                   folds, best_mcc, clf_type='keras')
    
keras_eval(train, y, test, train[:,0], test[:,0], folds=5, nbags=2, nepochs=55,
           batch_size=500, random_state=6)