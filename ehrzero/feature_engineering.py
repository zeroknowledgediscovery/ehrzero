import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import roc_auc_score, roc_curve, auc

def augment_agg(X):
    mean = np.array(X.mean(axis = 1)).reshape(X.shape[0],1)
    std = np.array(X.std(axis = 1)).reshape(X.shape[0],1)
    rang = np.array((X.max(axis = 1) - X.min(axis = 1))).reshape(X.shape[0],1)

    X = np.append(X,mean,1)
    X = np.append(X,std,1)
    X = np.append(X,rang,1)
    
    return X

def longest_one_streak(lst):
    return max(sum(1 for x in l if x == 1) for n, l in itertools.groupby(lst))

def optimal_cutoff(labels, preds):
    ####################################
    # The optimal cut off would be where tpr is high and fpr is low
    # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    ####################################
    fpr, tpr, cutoff = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    #print("Area under the ROC curve : %f" % roc_auc)
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(cutoff, index = i)})
    return float(roc.ix[(roc.tf-0).abs().argsort()[:1]]["thresholds"])

def get_dynamics(VALUES, TEST_SEQ_LENGTH):
                        first_half = pd.Series([np.mean(i[:int(TEST_SEQ_LENGTH/2)]) for i in VALUES])
                        second_half = pd.Series([np.mean(i[int(TEST_SEQ_LENGTH/2):TEST_SEQ_LENGTH]) for i in VALUES])
                        dynamics = (second_half/first_half).fillna(0)
                        dynamics[np.isinf(dynamics)] = 0
                        return dynamics

def get_max_streak_length(arr, x): 
    # intitialize count 
    count = 0 
    # initialize max 
    result = 0 
    for i in arr: 
        # Reset count when 0 is found 
        if (i == x): 
            count += 1
        # If 1 is found, increment count 
        # and update result if count  
        # becomes more. 
        else: 
            count = 0
        result = max(result, count)  
    return result  