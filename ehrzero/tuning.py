import numpy as np
import pandas as pd
import numpy as np
from random import randint, choice
from random import uniform as randfloat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Normalizer

def split_combine(df, ratio = 0.81):
    X = df.drop("target", 1)
    y = df.target
    Xtt, Xtv, ytt, ytv = train_test_split(X, y, test_size=0.81, random_state=1321)
    TRA = Xtt
    TRA["target"] = ytt
    TES = Xtv
    TES["target"] = ytv
    return TRA, TES

def class_weights(arr):
    dict(pd.Series([i[0] for i in arr]).value_counts(normalize = True))
"""
params = {'boosting_type': 'gbdt',
          'max_depth' : randint(222, 330),
          'objective': 'binary',
          'nthread': 28, # Updated from nthread
          'num_leaves': randint(80, 125),
          'learning_rate': randfloat(0.22,0.25),
          'max_bin': randint(50, 450),
          'subsample_for_bin': randint(8000, 10500), # 
          'subsample': randfloat(0.9925, 1),
          'subsample_freq': randint(35, 45),
          'colsample_bytree': randfloat(0.5, 0.57),
          'reg_alpha': randfloat(7, 9),
          'reg_lambda': randfloat(8, 14),
          'min_split_gain': 0.0012,#randfloat(0.001, 0.003),  #
          'min_child_weight': randfloat(6, 8), #
          'min_child_samples': randint(450, 530),
          'scale_pos_weight': randfloat(0.62, 0.72),
          'min_data_in_leaf': randint(100, 500),
          'metric' : 'auc',
          'early_stopping_round' : 37,
          'verbosity' : 0}
"""
    
def RUN_XG(train_set, test_set, k_splits):
    hyperparameters = {'boosting_type': 'gbdt',
          'max_depth' : randint(222, 330),
          'objective': 'binary',
          'nthread': 28, # Updated from nthread
          'num_leaves': randint(80, 125),
          'learning_rate': randfloat(0.22,0.25),
          'max_bin': randint(50, 450),
          'subsample_for_bin': randint(8000, 10500), # 
          'subsample': randfloat(0.9925, 1),
          'subsample_freq': randint(35, 45),
          'colsample_bytree': randfloat(0.5, 0.57),
          'reg_alpha': randfloat(7, 9),
          'reg_lambda': randfloat(8, 14),
          'min_split_gain': 0.0012,#randfloat(0.001, 0.003),  #
          'min_child_weight': randfloat(6, 8), #
          'min_child_samples': randint(450, 530),
          'scale_pos_weight': randfloat(0.62, 0.72),
          'min_data_in_leaf': randint(100, 500),
          'metric' : 'auc',
          'early_stopping_round' : 37,
          'verbosity' : 0}
    auc_scores = []
    DELELIST = ['target']
    TOTAL = pd.concat([train_set, test_set])
    X = TOTAL.drop(DELELIST, 1)
    y = TOTAL.target
    
    skf = StratifiedKFold(n_splits=k_splits)
    skf.get_n_splits(X, y)
    
    for train_index, test_index in skf.split(X, y):
        Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
        Xt, Xv, yt, yv = train_test_split(Xtrain, ytrain, test_size=0.38)
        Xt_array = Xt.values
        yt_array = yt.values.astype('int').flatten()
        Xv_array = Xv.values
        yv_array = yv.values.astype('int').flatten()
        
        clf = lgb.LGBMClassifier(**hyperparameters)
        clf.fit(Xt_array, yt_array, 
                eval_set = [(Xt_array, yt_array), (Xv_array, yv_array)],
                eval_metric = 'auc',
                early_stopping_rounds = 37, 
                verbose = False)
        y_predict = clf.predict_proba(Xtest, num_iteration=clf.best_iteration_)
        auc = roc_auc_score(ytest, y_predict[:,1])
        auc_scores.append(auc)
        
    output = hyperparameters.copy()
    output["AUC"] = np.mean(auc_scores)
    print(" XXXXX X X >>", np.mean(auc_scores))
    return output

def RUN_NB(train_set, test_set, k_splits):
    hyperparameters = {"var_smoothing": randfloat(0.01,5)}
    auc_scores = []
    DELELIST = ['target']
    ftest = test_set
    Xtest = ftest.drop(DELELIST, 1)
    ytest = ftest.target
    for i in range(k_splits):
        ftrain = train_set
        X = ftrain.drop(DELELIST, 1)
        y = ftrain.target
        scaler = MinMaxScaler()
        scaler.fit(X)
        del ftrain
        clf = GaussianNB(**hyperparameters)
        clf.fit(scaler.transform(X) + 10, y)
        y_predict = clf.predict_proba(scaler.transform(Xtest) + 10)
        auc = roc_auc_score(ytest, y_predict[:,1])
        auc_scores.append(auc)
        
    output = hyperparameters.copy()
    output["AUC"] = np.mean(auc_scores)
    print(" XXXXX X X >>", np.mean(auc_scores))
    return output

def random_search(num_iters, train_set, test_set, model, k_splits = 5, downsample = 0):
    learning_results = []
    for i in range(num_iters):
        if i % 14 == 0:
            print(" {  %d  }  " % i)
        if model == "LGBM":
            learning_results.append(RUN_XG(train_set, test_set, k_splits))
        elif model == "NB":
            learning_results.append(RUN_NB(train_set, test_set, k_splits))
        else:
            print("Model not recognized.\nCurrently available models:\n'LGBM' : LightGBM\n'NB' : GaussianNaiveBayes\n")
            return
    DF = pd.DataFrame(learning_results)
    print("======\nMEAN ::: ",np.mean(DF.AUC))
    return DF
