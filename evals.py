#!/usr/bin/env python
#==============================================================================
#
#          FILE:  evals.py
# 
#         USAGE:  import evals (from hib.py)
# 
#   DESCRIPTION:  evaluation routines
# 
#       UPDATES:  170339: removed shuffle from getfolds()
#                 170410: added reclass()
#                 170417: renamed reclass() to reclass_result()
#                         reworked reclass_result()
#                 170510: reclass_result() convert result to numpy array before
#                         attaching to pandas DataFrame
#                 170801: added odds_ratio()
#        AUTHOR:  Pete Schmitt (hershey), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.3
#       CREATED:  Sun Mar 19 11:34:09 EDT 2017
#      REVISION:  Wed Aug 16 13:55:04 EDT 2017
#==============================================================================
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
###############################################################################
def subsets(x,percent):
    """ take a subset of "percent" of x """
    p = percent / 100
    xa = np.array(x)
    subsample_indices = np.random.choice(xa.shape[1], int(xa.shape[1] * p), 
                                         replace=False)
    return (xa[:, subsample_indices]).tolist()
###############################################################################
def getfolds(x, num):
    """ return num folds of size 1/num'th of x """
    folds = []
    fsize = end = int(len(x[0]) / num)
    xa = np.array(x)
    start = 0
    for i in range(num):
        folds.append(xa[:,start:end])
        start += fsize
        end += fsize  
    return folds
###############################################################################
def addnoise(x,pcnt):
    """ add some percentage of noise to data """
    xa = np.array(x)
    val = pcnt/100
    rep = {}
    rep[0] = [1,2]
    rep[1] = [0,2]
    rep[2] = [0,1]

    for i in range(len(xa)):
        indices = np.random.choice(xa.shape[1], int(xa.shape[1] * val), 
                                   replace=False)
        for j in list(indices):
            xa[i][j] = np.random.choice(rep[xa[i][j]])

    return xa.tolist()
###############################################################################
def reclass_result(x, result, pct):
    """ reclassify data """
    d = np.array(x).transpose()
    columns = [0]*len(x)
    # create columns names for variable number of columns.
    for i in range(len(x)):
        columns[i] = 'X' + str(i)
    
    df = pd.DataFrame(d, columns=columns)
    dflen = len(df)
    np_result = np.array(result)

    df['Class'] = np_result

    df.sort_values('Class', ascending=True, inplace=True)
    
    cntl_cnt = dflen - int(dflen * (pct/100.0))
    c = np.zeros(dflen, dtype=np.int)
    c[cntl_cnt:] = 1

    df.Class = c
    df.sort_index(inplace=True)  # put data back in index order
    return df['Class'].tolist()
###############################################################################
def oddsRatio(data, class_labels, dlen):
    """ returns sums of difference between fixed and odd_ratio """
#   fixed = np.array([4,2,1.5,1.4,1.3,1.2,1.18,1.16,1.14,1.12])
    fixed = np.array([1.5, 1.3, 1.2, 1.15, 1.1, 1.09, 1.08, 1.07, 1.06, 1.05])
    odds_ratio = []
    
    features = np.array(data)
    class_labels = np.array(class_labels)

    for feature_index in range(dlen):
        clf = LogisticRegression()
        x = features[feature_index].reshape(-1,1)
        clf.fit(x, class_labels)
        odds_ratio.append(np.exp(clf.coef_[0][0]))

    odds_ratio = np.array(sorted(odds_ratio, reverse=True))
    sum_of_diffs = np.sum(np.absolute(np.subtract(fixed, odds_ratio)))
    return sum_of_diffs, odds_ratio
