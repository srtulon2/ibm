from sklearn.metrics import accuracy_score, precision_score, make_scorer, f1_score, roc_auc_score, recall_score, precision_recall_curve, average_precision_score, plot_roc_curve, roc_curve, auc, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, brier_score_loss, log_loss

from sklearn.metrics._ranking import _binary_clf_curve

from prg import prg

from hmeasure import h_score

import numpy as np

import copy


import math
from sklearn.metrics import confusion_matrix


def get_pr(y_real, y_probabilities):
    precision, recall, _ = precision_recall_curve(y_real, y_probabilities)
    return precision, recall

def gmean_score(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    TN = matrix[0][0]
    FN = matrix[1][0]
    TP = matrix[1][1]
    FP = matrix[0][1]

    return math.sqrt((TP/(TP+FN))*(TN/(TN+FP)))



def pr_davis(y_real, y_probabilities,return_pr=False):
    fps, tps, thresholds = _binary_clf_curve(y_real, y_probabilities,pos_label=None,sample_weight=None)

    #Interpolate new TPs and FPs when diff between successive TP is >1
    for i in range(len(tps)-1):
        if (tps[i+1] - tps[i]) >= 2:
            local_skew = (fps[i+1]-fps[i])/(tps[i+1]-tps[i])
        
        for x in range(1,int(tps[i+1] - tps[i])):
            new_fp = fps[i]+(local_skew*x)
            tps = np.insert(tps, i+x, tps[i]+x)
            fps = np.insert(fps, i+x, new_fp)


    precision_davis = tps / (tps + fps)
    precision_davis[np.isnan(precision_davis)] = 0
    recall_davis = tps / tps[-1]
        
    # Stop when full recall is attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    precision_davis = np.r_[precision_davis[sl], 1]
    recall_davis = np.r_[recall_davis[sl], 0]
    pr_auc_davis = auc(recall_davis, precision_davis)

    if return_pr:
        return precision_davis, recall_davis, pr_auc_davis
    
    else:
        return pr_auc_davis


def pr_manning(y_real, y_probabilities,return_pr=False):
    precision, recall = get_pr(y_real, y_probabilities)
    
    precision_manning = copy.deepcopy(precision)
    recall_manning = copy.deepcopy(recall)
    prInv = np.fliplr([precision_manning])[0]
    recInv = np.fliplr([recall_manning])[0]
    j = recall_manning.shape[0]-2

    while j>=0:
        if prInv[j+1]>prInv[j]:
            prInv[j]=prInv[j+1]
        j=j-1

    decreasing_max_precision = np.maximum.accumulate(prInv[::-1])[::-1]
    pr_auc_manning = auc(recInv, decreasing_max_precision)

    if return_pr:
        return decreasing_max_precision, recInv, pr_auc_manning
    
    else:
        return pr_auc_manning


def pr_gain_curve(y_real, y_probabilities,return_prg=False):
    prg_curve = prg.create_prg_curve(y_real, y_probabilities)

    if return_prg:
        return prg_curve['precision_gain'], prg_curve['recall_gain'], prg_curve
  
    else:
        return prg_curve


def prg_auc(y_real, y_probabilities):

    prg_curve = prg.create_prg_curve(y_real, y_probabilities)
    return prg.calc_auprg(prg_curve)
