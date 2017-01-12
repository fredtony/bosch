`# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 19:35:54 2016

@author: Tony
"""

import cPickle
import os
import pandas as pd
import numpy as np
import tqdm
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from numba import jit
import datetime
import copy

MODELDIR = '../models/'
DATADIR = '../input/'

numfile = DATADIR + 'train_numeric.csv'
datefile = DATADIR + 'train_date.csv'
catfile = DATADIR + 'train_categorical.csv'
magicfile = DATADIR + 'train_magic.csv'
numfile_test = DATADIR + 'test_numeric.csv'
datefile_test = DATADIR + 'test_date.csv'
catfile_test = DATADIR + 'test_categorical.csv'
magicfile_test = DATADIR + 'test_magic.csv'

num_rows = 1183747

with open('date_non_dupe.pkl', 'r') as f:
    date_index = cPickle.load(f)
with open('num_non_dupe.pkl', 'r') as f:
    num_index = cPickle.load(f)
with open('cat_non_dupe.pkl', 'r') as f:
    cat_index = cPickle.load(f)
    
def out_from_probs(train_ids, cv_pred, test_ids, test_pred, best_proba,
                   folds, best_mcc, clf_type):
    now = datetime.datetime.now()
    
    pred = pd.DataFrame({'Id': test_ids, 'Response': pd.Series(test_pred)})
    pred_file = '../models1/preds_{}fold-{}-{:.4f}-{}.csv.gz'.format(
        folds, clf_type, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing test probabilities: {}".format(pred_file))
    pred.to_csv(pred_file, index=False, compression='gzip')
    
    oof_pred = pd.DataFrame({'Id': train_ids, 'Response': pd.Series(cv_pred)})
    oof_pred_file = '../models1/oof_preds_{}fold-{}-{:.4f}-{}.csv.gz'.format(
        folds, clf_type, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing oof probabilities: {}".format(oof_pred_file))
    oof_pred.to_csv(oof_pred_file, index=False, compression='gzip')
    
    test_pred = (test_pred >= best_proba).astype(int)
    
    result = pd.DataFrame({'Id': test_ids, 'Response': pd.Series(test_pred)})
    sub_file = 'submission_{}fold-{}-{:.4f}-{}.csv.gz'.format(
        folds, clf_type, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing submission: {}".format(sub_file))
    result.to_csv(sub_file, index=False, compression='gzip')

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


current_model_list = os.listdir(MODELDIR)
len_models = len(current_model_list)/2
oof_list = current_model_list[:len_models]
preds_list = current_model_list[len_models:]
train_ids = pd.read_csv(MODELDIR+oof_list[1], usecols=['Id'], squeeze=True).values
test_ids = pd.read_csv(MODELDIR+preds_list[1], usecols=['Id'], squeeze=True).values

def import_models(model_list):
    model = pd.read_csv(MODELDIR+model_list[0])
    for elem in model_list[1:]:
        next_model = pd.read_csv(MODELDIR+elem)
        model = model.merge(next_model, on='Id', how='left')
    return model

train = import_models(oof_list)
train = train.merge(pd.read_csv('train_date_mod.csv', usecols=['Id','Response']),
                    on='Id', how='left')
y = train.iloc[:,-1].values
train = train.iloc[:,1:-1].values
test = import_models(preds_list).values[:,1:]

def eval_xgb(train, y, test, train_ids, test_ids, nfolds=5, random_state=0):
    xgb_params = {
    "base_score": 0.005,
    "booster": "gblinear",
    "objective": "binary:logistic",
    "max_depth": 8,
    #"eval_metric": 'auc',
    "eta": 0.1,
    "silent": 1,
    }
    num_round = 1000
    early_stopping_rounds = 10
    #nfolds = nfolds
    #watchlist = [(dtrain,'train'), (dcv,'cv')]
    #bst = xgb.train(xgb_params, dtrain, num_round, watchlist, maximize=True)
    
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    y_pred = np.zeros((1183748, nfolds))
    y_oof_pred = np.zeros(y.shape)
    xgb_models = []
    best_probas = np.zeros(nfolds)
    best_mccs = np.zeros(nfolds)
    dtest = xgb.DMatrix(test)
    for i, (train_idx, cv_idx) in enumerate(skf.split(train, y)):
        dtrain = xgb.DMatrix(train[train_idx,:], y[train_idx])
        dcv = xgb.DMatrix(train[cv_idx,:], y[cv_idx])
        watchlist = [(dtrain,'train'), (dcv,'cv')]
        bst = xgb.train(xgb_params, dtrain, num_round, watchlist, maximize=True, feval=mcc_eval,
                        verbose_eval=5, early_stopping_rounds=early_stopping_rounds)
        y_oof_pred[cv_idx] = bst.predict(dcv)
        best_probas[i], best_mccs[i], _ = eval_mcc2(y[cv_idx], y_oof_pred[cv_idx])
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
    
    best_proba, best_mcc, _ = eval_mcc2(y, y_oof_pred)
    print("Out of fold (predicted LB) CV mcc: {}".format(best_mcc))
    
    now = datetime.datetime.now()
    
    pred = pd.DataFrame({'Id': test_ids.astype(np.int32), 'Response': pd.Series(y_pred)})
    pred_file = '../models1/preds_{}fold-xgb_ensemble1-{:.4f}-{}.csv.gz'.format(
        nfolds, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing test probabilities: {}".format(pred_file))
    pred.to_csv(pred_file, index=False, compression='gzip')
    
    oof_pred = pd.DataFrame({'Id': train_ids.astype(np.int32), 'Response': pd.Series(y_oof_pred)})
    oof_pred_file = '../models1/oof_preds_{}fold-xgb_ensemble1-{:.4f}-{}.csv.gz'.format(
        nfolds, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing oof probabilities: {}".format(oof_pred_file))
    oof_pred.to_csv(oof_pred_file, index=False, compression='gzip')
    
    y_pred = (y_pred >= best_proba).astype(int)
    
    result = pd.DataFrame({'Id': test_ids.astype(np.int32), 'Response': pd.Series(y_pred)})
    sub_file = 'submission_{}fold-xgb_ensemble1-{:.4f}-{}.csv.gz'.format(
        nfolds, best_mcc, now.strftime("%Y-%m-%d-%H-%M"))
    print("\n Writing submission: {}".format(sub_file))
    result.to_csv(sub_file, index=False, compression='gzip')
    return

eval_xgb(train, y, test, train_ids, test_ids, nfolds=10, random_state=936)


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
#clf = ExtraTreesClassifier(n_estimators=40, criterion='entropy', n_jobs=-1, random_state=0)
#clf = GradientBoostingClassifier()
#clf = AdaBoostClassifier(random_state=0)
#clf = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
clf = MLPClassifier(hidden_layer_sizes=(400,20), random_state=0,
                    tol=.0001, verbose=True)
eval_clf(clf, train, y, test, train_ids, test_ids, 'LRensemble1', 10, 18)