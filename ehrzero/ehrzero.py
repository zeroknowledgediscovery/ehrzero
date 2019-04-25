import os
import numpy as np
import itertools
import pickle
import tempfile

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score


import pandas as pd
from sklearn.preprocessing import Imputer
from functools import reduce

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score

import lightgbm as lgb
from lightgbm import LGBMClassifier
from random import randint, choice
from random import uniform as randfloat

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

from ehrzero.feature_engineering import get_max_streak_length, get_dynamics
from ehrzero.tools import combine_disease_groups, bash
from ehrzero.z3 import Z3Classifier
from ehrzero.preprocessing import convert_to_dx, optimize_memory, retrieve_raw_records
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def prediction_outcome(true, pred):
    if pred == true and pred == 1:
        return "TP"
    elif pred == true and pred == 0:
        return "TN"
    elif pred == 0 and true == 1:
        return "FN"
    elif pred == 1 and true == 0:
        return "FP"

def roc_auc(tru, pred):
    try:
        return roc_auc_score(tru,pred)
    except:
        return 0

def load_sequence_data(GENDER,
              DISEASE_GROUPS,
              short_path,
              full_path):
    short_seqs = {}
    full_seqs = {}
    for KEY in DISEASE_GROUPS:
            print(KEY)
            df_short = pd.read_csv(short_path % (GENDER, KEY))
            # Compute the average digit of the sequence (to detect all-zeros later)
            df_short["MEAN"] = [np.mean([int(i) for i in rec.split()]) for rec in df_short.record]
            short_seqs[KEY] = df_short

            df_full = pd.read_csv(full_path % KEY)
            df_full["MEAN"] = [np.mean([int(i) for i in rec.split()]) for rec in df_full.record]
            full_seqs[KEY] = df_full
    return short_seqs, full_seqs

def combine_disease_groups(dfs, ID = "patient_id", prediction = False):
                df = reduce(lambda left,right: pd.merge(left, right, on = ID,
                                                            how='outer').drop_duplicates(), dfs)
                if prediction:
                    return df
                else:
                    all_cols = list(df.columns)
                    num = 0
                    for i in range(len(all_cols)):
                        if "target" in all_cols[i]:
                            all_cols[i] = "target_%d" % num
                            num += 1
                    df.columns = all_cols
                    targets = [df[var] for var in list(df.columns) if "target" in var]
                    full_target = reduce(lambda left, right : left.combine_first(right), targets)
                    df.drop([i for i in list(df.columns) if "target" in i], 1, inplace = True)
                    df['target'] = full_target
                    return df
"""
    NEW PIPELINE RUN
    [NOT YET TESTED], TEST ON [22AUT]>[COMPUTE_COUNITES]
"""
def run_pipeline(
    RUNS,
    GENDER,
    DISEASE_GROUPS,
    SHORT_SEQUENCES,
    FULL_SEQUENCES,
    RESULT_DIR = "SUBMISSION/PFSA",
    HYPERPARAMETERS = "NEXTONE_TUNING.csv",
    PFSA_PATH = "SUBMISSION/PFSA/%s/%s",
    LLK_PATH = "bin/llk",
    MODEL_PATH = "",
    APPEND_PREDICTIONS_TO = "",
    FIPS = "00000",
    TUNING_LABEL = "SAMPLE",
    POS_EPSILON = 0.24,
    NEG_EPSILON = 0.2,
    PFSA_SPLIT = 0.42,
    TEST_SPLIT = 0.4,
    VALID_SPLIT = 0.15,
    LLK_SEQUENCE_LENGTH = 150,
    downsample = 0,
    verbose = False,
    save_preds = False):

    start = True
    patient_data = []
    if APPEND_PREDICTIONS_TO:
        PREDICTIONS = pd.read_csv(APPEND_PREDICTIONS_TO)
    else:
        PREDICTIONS = 0
    # Dictionary for the feature importances
    IMP = {}
    # Run the pipeline for multiple times
    for x in range(RUNS):
        # Get downsamples of the imput data
        if downsample:
            downsamples = {}
            for disease in DISEASE_GROUPS:
                downsamples[disease] = SHORT_SEQUENCES[disease].copy()
                POS = downsamples[disease][downsamples[disease].target == 1]
                NEG = downsamples[disease][downsamples[disease].target == 0].sample(frac = downsample)
                downsamples[disease] = pd.concat([POS,NEG])
            SHORT_SEQUENCES = downsamples
        print(x+1)
        dfs = []
        KEYS = []
        ### Compute the following for every disease group
        for KEY in DISEASE_GROUPS:
                # Declare the llk predictions dictionary
                preds = {}
                DATASET = SHORT_SEQUENCES[KEY]
                features = DATASET.drop(['fips', 'target'], 1)
                labels = DATASET.target
                """
                    Split the data into PFSA and LLK (TRAIN+TEST) sets
                """
                X_pfsa, X_llk, y_pfsa, y_llk = train_test_split(features, labels, test_size=PFSA_SPLIT)
                pfsa_set = X_pfsa.copy()
                pfsa_set['target'] = list(y_pfsa)
                llk_set = X_llk.copy()
                llk_set["target"] = y_llk
                # Leave only those records with at least one nonzero digit in the sequence
                llk_set = llk_set[llk_set.MEAN > 0]
                X_llk = llk_set.drop("target", 1)
                y_llk = llk_set.target
                """
                    Generate PFSAs with pfsa_set
                """
                full_sequences = FULL_SEQUENCES[KEY]
                Z=Z3Classifier(result_path = PFSA_PATH % (GENDER, KEY), llk_path = LLK_PATH)
                if verbose:
                    print(pfsa_set.shape)
                # Fetch the full-length versions of the sequences by patient_id
                fit_set = full_sequences[full_sequences["patient_id"].isin(pfsa_set.patient_id)]
                fit_set["MEAN"] = [np.mean([int(i) for i in rec.split()]) for rec in fit_set.record]
                fit_set = fit_set[fit_set.MEAN > 0]
                Z.fit(fit_set,
                    peps = POS_EPSILON,
                    neps = NEG_EPSILON) # fits with BOTH feats and labels
                preds["patient_id"] = list(llk_set.patient_id)
                """
                    Generate LLs with llk_set
                """
                LL = Z.predict_loglike(llk_set) # raw outpuit is zip of POS and NEG LLs
                ## LogLikelihoods of a sequence belonging to a NEG and POS PFSAs
                preds[KEY] = np.array(list(LL[1])) - np.array(list(LL[0]))
                """
                    Generate record-based features
                """
                VALUES = np.array([[int(int(i) == 1) for i in rec.split()] for rec in llk_set.record])
                RAW_VAL = np.array([[int(i) for i in rec.split()] for rec in llk_set.record])
                PREVAL = []
                for arr in RAW_VAL:
                    try:
                        ONES = np.mean([int(i == 1) for i in arr])
                        TWOS = np.mean([int(i == 2) for i in arr])
                        NONZERO = np.mean([int(i != 0) for i in arr])
                        if ONES == 0:
                            PREVAL.append(0)
                        elif TWOS == 0:
                            PREVAL.append(1)
                        else:
                            PREVAL.append(ONES/NONZERO)
                    except:
                        PREVAL.append(0)
                PREVAL = np.array(PREVAL)
                preds[KEY + "_prevalence"] = PREVAL
                preds[KEY + "_proportion"] = [np.mean(i) for i in VALUES]
                preds[KEY + "_streak"] = [get_max_streak_length(i, 1) for i in VALUES]
                preds[KEY + "_intermission"] = [get_max_streak_length(i, 0) for i in RAW_VAL]
                preds[KEY + "_dynamics_p"] = get_dynamics(VALUES, LLK_SEQUENCE_LENGTH)
                # --------------------------------------------------------
                df = pd.DataFrame(preds)
                df['target'] = list(y_llk)
                dfs.append(df)
                KEYS.append(KEY)

        """ ========================================================= """
        """ Compile all the disease group features into one dataframe """
        """ ========================================================= """
        df = combine_disease_groups(dfs)
        """
            Initialize feature importance and AUC dataframe
        """
        params = list(df.drop(["patient_id", "target"], 1).columns)\
            + ["MEAN", "STD", "RANGE"] + ["AUC", "AUC_lgbm", "AUC_nbayes"]
        if len(IMP) == 0:
            IMP = {param: [] for param in params}
        """
            CONSTRUCT X-y pair, aggregate, impute
        """
        X = df.drop(["patient_id", "target"], 1)
        IMPUTER = Imputer(missing_values=np.nan, strategy='mean')
        IMPUTER.fit(X)
        X=IMPUTER.transform(X)
        y = []
        for i in df.target:
            try:
                y.append(int(i))
            except:
                y.append(0)
        # Get back the column names
        X = pd.DataFrame(X, columns = df.drop(["patient_id", "target"], 1).columns)
        X['patient_id'] = df["patient_id"]
        X["MEAN"] = X[DISEASE_GROUPS].mean(1)
        X["STD"] = X[DISEASE_GROUPS].std(1)
        X["RANGE"] = X[DISEASE_GROUPS].max(1) - X[DISEASE_GROUPS].min(1)
        """
            Retrieve the tuned hyperparameters
        """
        TUNED_PARAMS = pd.read_csv(HYPERPARAMETERS)#.drop("target", 1)
        BEST_PARAMS = dict(TUNED_PARAMS.sort_values(by = 'AUC', ascending = False).reset_index().ix[0,:])
        del BEST_PARAMS['AUC']
        """
            LIGHTGBM
        """
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.41)
        Xtrain.drop(["patient_id"], 1, inplace = True)
        TEST_IDS = Xtest['patient_id']
        Xtest.drop(["patient_id"], 1, inplace = True)
        Xtt, Xtv, ytt, ytv = train_test_split(Xtrain, ytrain, test_size=0.11)
        LGBM = lgb.LGBMClassifier(**BEST_PARAMS)
        LGBM.fit(
            Xtt, ytt,
            eval_set = [(Xtt, ytt), (Xtv, ytv)],
            eval_metric = 'auc',
            early_stopping_rounds = 100,
            verbose=False
        )
        lgbm_predict = LGBM.predict_proba(Xtest, num_iteration=LGBM.best_iteration_)
        """
            GAUSSIAN NAIVE BAYES
        """
        NB = GaussianNB(var_smoothing = 0.05)
        NB_SCALER = MinMaxScaler()
        NB_SCALER.fit(Xtrain)
        NB.fit(NB_SCALER.transform(Xtrain) + 1, ytrain)
        nb_predict = NB.predict_proba(NB_SCALER.transform(Xtest) + 1)
        """
            COMBINE THE NB AND LGBM SCORES INTO A FINAL ONE
        """
        f = 0.99
        final_predict = lgbm_predict[:,1] * f + nb_predict[:,1] * (1 - f)
        auc_nb = roc_auc_score(ytest, nb_predict[:,1])
        auc_lgbm = roc_auc_score(ytest, lgbm_predict[:,1])
        auc_X = roc_auc_score(ytest, final_predict)
        for i in range(len(params) - 3):
            IMP[params[i]].append(LGBM.feature_importances_[i])
        print("LGBM_SCORE    > %f" % auc_lgbm)
        print("NB_SCORE      > %f" % auc_nb)
        print("FINAL_SCORE   > %f" % auc_X)
        print("------------------------------")
        IMP["AUC_lgbm"].append(auc_lgbm)
        IMP["AUC_nbayes"].append(auc_nb)
        IMP["AUC"].append(auc_X)
        """
            OPTIONAL :: SAVE PREDICTIONS
            Save the whole set of predictions for the first run
            add to the existing dataset, excluding already added patient_ids
            for the following runs
        """
        if save_preds:
            XXX = Xtest.copy()
            XXX['patient_id'] = TEST_IDS
            XXX['predicted_risk'] = final_predict
            XXX['truth'] = ytest
            if start:
                PREDICTIONS = XXX
                print(" > ADDING %d new ids" % XXX.shape[0])
                start = False
            else:
                XXX = XXX[~XXX['patient_id'].isin(PREDICTIONS["patient_id"])]
                print(" >>> > > > ADDING %d new ids" % XXX.shape[0])
                PREDICTIONS = pd.concat([PREDICTIONS, XXX])
        """
            PICKLE THE MODELS
        """
        pickle.dump(NB_SCALER, open( "%s/%s/NB_SCALER.pickle" % (MODEL_PATH, GENDER), "wb" ) )
        pickle.dump(IMPUTER, open( "%s/%s/IMPUTER.pickle" % (MODEL_PATH, GENDER), "wb" ) )
        pickle.dump(LGBM, open( "%s/%s/LGBM.pickle" % (MODEL_PATH, GENDER), "wb" ) )
        pickle.dump(NB, open( "%s/%s/NB.pickle" % GENDER, "wb" ) )
        """
            SAVE FINAL TRAIN & TEST SETS FOR TUNING
        """
        XXT = pd.DataFrame(Xtrain)
        XXT['target'] = ytrain
        XXT.to_csv("%s_TRAIN.csv" % TUNING_LABEL, index = False)
        XXTS = pd.DataFrame(Xtest)
        XXTS['target'] = ytest
        XXTS.to_csv("%s_TEST.csv" % TUNING_LABEL, index = False)
        ############################################################
    """
        Get averages of the feature importance values, AUC scores
    """
    lag = {param:np.mean(IMP[param]) for param in params}
    Idf=pd.DataFrame.from_dict(lag,orient='index',columns=['value']).transpose()

    if save_preds:
        return Idf, PREDICTIONS
    else:
        return Idf

def train_models(GENDER,
                 DISEASE_GROUPS,
                 LLK_SEQUENCE_LENGTH,
                 PFSA_SET,
                 LLK_SET,
                 PFSA_PATH,
                 MODEL_PATH,
                 HYPERPARAMETERS,
                 POS_EPSILON = 0.24,
                 NEG_EPSILON = 0.2):
    test_set_disease_groups = []
    IMP = {}
    for KEY in DISEASE_GROUPS:
        test_preds = {}
        test_preds['patient_id'] = list(LLK_SET[KEY].patient_id)
        print(KEY)
        Z=Z3Classifier(result_path = PFSA_PATH % (GENDER, KEY), llk_path = "bin/llk")
        Z.fit(PFSA_SET[KEY], peps = POS_EPSILON, neps = NEG_EPSILON)

        LL = Z.predict_loglike(LLK_SET[KEY]) # raw outpuit is zip of POS and NEG LLs
        ## LogLikelihoods of a sequence belonging to a NEG and POS PFSAs
        test_preds[KEY] = list(np.array(list(LL[1])) - np.array(list(LL[0])))
        """
            Generate record-based features
        """
        VALUES = np.array([[int(int(i) == 1) for i in rec.split()] for rec in LLK_SET[KEY].record])
        RAW_VAL = np.array([[int(i) for i in rec.split()] for rec in LLK_SET[KEY].record])
        PREVAL = []
        for arr in RAW_VAL:
            try:
                ONES = np.mean([int(i == 1) for i in arr])
                TWOS = np.mean([int(i == 2) for i in arr])
                NONZERO = np.mean([int(i != 0) for i in arr])
                if ONES == 0:
                    PREVAL.append(0)
                elif TWOS == 0:
                    PREVAL.append(1)
                else:
                    PREVAL.append(ONES/NONZERO)
            except:
                PREVAL.append(0)
        PREVAL = np.array(PREVAL)
        test_preds[KEY + "_prevalence"] = PREVAL
        test_preds[KEY + "_proportion"] = [np.mean(i) for i in VALUES]
        test_preds[KEY + "_streak"] = [get_max_streak_length(i, 1) for i in VALUES]
        test_preds[KEY + "_intermission"] = [get_max_streak_length(i, 0) for i in RAW_VAL]
        test_preds[KEY + "_dynamics_p"] = get_dynamics(VALUES, LLK_SEQUENCE_LENGTH)
        # --------------------------------------------------------
        try:
            df = pd.DataFrame(test_preds)
        except:
            return test_preds
        df['target'] = list(LLK_SET[KEY].target)
        test_set_disease_groups.append(df)
    """ ========================================================= """
    """ Compile all the disease group features into one dataframe """
    """ ========================================================= """
    try:
        fit_df = combine_disease_groups(test_set_disease_groups)
    except:
        return test_set_disease_groups
    """
        Initialize feature importance and AUC dataframe
    """
    params = list(fit_df.drop(["patient_id", "target"], 1).columns)  + ["MEAN", "STD", "RANGE"] + ["AUC", "AUC_lgbm", "AUC_nbayes"]
    if len(IMP) == 0:
        IMP = {param: [] for param in params}
    """
        CONSTRUCT X-y pair, aggregate, impute
    """
    X = fit_df.drop(["patient_id", "target"], 1)
    IMPUTER = SimpleImputer(missing_values=np.nan, strategy='mean')
    IMPUTER.fit(X)
    X=IMPUTER.transform(X)
    y = []
    for i in fit_df.target:
        try:
            y.append(int(i))
        except:
            y.append(0)
    # Get back the column names

    X = pd.DataFrame(X, columns = fit_df.drop(["patient_id", "target"], 1).columns)
    X['patient_id'] = fit_df["patient_id"]
    X["MEAN"] = X[DISEASE_GROUPS].mean(1)
    X["STD"] = X[DISEASE_GROUPS].std(1)
    X["RANGE"] = X[DISEASE_GROUPS].max(1) - X[DISEASE_GROUPS].min(1)
    """
        Retrieve the tuned hyperparameters
    """
    TUNED_PARAMS = pd.read_csv(HYPERPARAMETERS)#.drop("target", 1)
    BEST_PARAMS = dict(TUNED_PARAMS.sort_values(by = 'AUC', ascending = False).reset_index().ix[0,:])
    del BEST_PARAMS['AUC']
    """
        LIGHTGBM
    """
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
    Xtrain.drop(["patient_id"], 1, inplace = True)
    TEST_IDS = Xtest['patient_id']
    Xtest.drop(["patient_id"], 1, inplace = True)
    Xtt, Xtv, ytt, ytv = train_test_split(Xtrain, ytrain, test_size=0.11)
    LGBM = lgb.LGBMClassifier(**BEST_PARAMS)
    LGBM.fit(
        Xtt, ytt,
        eval_set = [(Xtt, ytt), (Xtv, ytv)],
        eval_metric = 'auc',
        early_stopping_rounds = 100,
        verbose=False
    )
    lgbm_predict = LGBM.predict_proba(Xtest, num_iteration=LGBM.best_iteration_)
    """
        GAUSSIAN NAIVE BAYES
    """
    NB = GaussianNB(var_smoothing = 1.31)
    NB_SCALER = MinMaxScaler()
    NB_SCALER.fit(Xtrain)
    NB.fit(NB_SCALER.transform(Xtrain) + 1, ytrain)
    nb_predict = NB.predict_proba(NB_SCALER.transform(Xtest) + 1)
    """
        COMBINE THE NB AND LGBM SCORES INTO A FINAL ONE
    """
    f = 0.42
    final_predict = lgbm_predict[:,1] * f + nb_predict[:,1] * (1 - f)
    auc_nb = roc_auc_score(ytest, nb_predict[:,1])
    auc_lgbm = roc_auc_score(ytest, lgbm_predict[:,1])
    auc_X = roc_auc_score(ytest, final_predict)
    for i in range(len(params) - 3):
        IMP[params[i]].append(LGBM.feature_importances_[i])
    print("LGBM_SCORE    > %f" % auc_lgbm)
    print("NB_SCORE      > %f" % auc_nb)
    print("FINAL_SCORE   > %f" % auc_X)
    print("------------------------------")
    IMP["AUC_lgbm"].append(auc_lgbm)
    IMP["AUC_nbayes"].append(auc_nb)
    IMP["AUC"].append(auc_X)
    """
        PICKLE THE MODELS
    """
    pickle.dump(NB_SCALER, open(MODEL_PATH + "/%s/NB_SCALER.pickle" % GENDER, "wb" ) )
    pickle.dump(IMPUTER, open(MODEL_PATH + "/%s/IMPUTER.pickle" % GENDER, "wb" ) )
    pickle.dump(LGBM, open(MODEL_PATH + "/%s/LGBM.pickle" % GENDER, "wb" ) )
    pickle.dump(NB, open(MODEL_PATH + "/%s/NB.pickle" % GENDER, "wb" ) )
    """
        SAVE FINAL TRAIN & TEST SETS FOR TUNING
    """
    FIPS = "FEMM_TUNN"
    XXT = pd.DataFrame(Xtrain)
    XXT['target'] = ytrain
    XXT.to_csv("%s_TRAIN.csv" % FIPS, index = False)
    XXTS = pd.DataFrame(Xtest)
    XXTS['target'] = ytest
    XXTS.to_csv("%s_TEST.csv" % FIPS, index = False)

    return IMP

def ccombine_disease_groups(dfs, ID = "patient_id", prediction = False):
    df = reduce(lambda left,right: pd.merge(left, right, on = ID,
                                                how='outer').drop_duplicates(), dfs)
    #if not prediction:
    all_cols = list(df.columns)
    num = 0
    for i in range(len(all_cols)):
        if "target" in all_cols[i]:
            all_cols[i] = "target_%d" % num
            num += 1
    df.columns = all_cols
    targets = [df[var] for var in list(df.columns) if "target" in var]
    full_target = reduce(lambda left, right : left.combine_first(right), targets)
    df.drop([i for i in list(df.columns) if "target" in i], 1, inplace = True)
    df['target'] = full_target
    return df

def run_saved_pipeline(PIPELINE, evaluate = False, verbose = False,work_dir='./'):
    dfs = []
    for KEY in PIPELINE["CONDITIONS"]:
        preds = {}
        llk_set = PIPELINE["DATA"][KEY].copy()
        Z = Z3Classifier(use_own_pfsa = True,
                         posmod = PIPELINE["PFSA_POS"] % KEY,
                         negmod = PIPELINE["PFSA_NEG"] % KEY)
        llk_set.to_csv(work_dir+"/SEQ_"+KEY+".csv", index = False)
        llk_set["record"] = [" ".join(A.split()[1:PIPELINE["LENGTH"]+1]) for A in llk_set["record"]]
        LL = Z.predict_loglike(llk_set[["record"]]) # raw output is zip
        preds[KEY] = np.array(list(LL[1])) - np.array(list(LL[0]))
        preds["patient_id"] = list(llk_set.patient_id)
        VALUES = np.array([[int(int(i) == 1) for i in rec.split()] for rec in llk_set.record])
        RAW_VAL = np.array([[int(i) for i in rec.split()] for rec in llk_set.record])
        PREVAL = []
        for arr in RAW_VAL:
            try:
                ONES = np.mean([int(i == 1) for i in arr])
                TWOS = np.mean([int(i == 2) for i in arr])
                NONZERO = np.mean([int(i != 0) for i in arr])
                if ONES == 0:
                    PREVAL.append(0)
                elif TWOS == 0:
                    PREVAL.append(1)
                else:
                    PREVAL.append(ONES/NONZERO)
            except:
                PREVAL.append(0)
        PREVAL = np.array(PREVAL)
        preds[KEY + "_prevalence"] = PREVAL
        preds[KEY + "_proportion"] = [np.mean(i) for i in VALUES]
        preds[KEY + "_streak"] = [get_max_streak_length(i, 1) for i in VALUES]
        preds[KEY + "_intermission"] = [get_max_streak_length(i, 0) for i in RAW_VAL]
        preds[KEY + "_dynamics_p"] = get_dynamics(VALUES, PIPELINE["LENGTH"])
        if evaluate:
            preds['target'] = list(llk_set.target)
        df = pd.DataFrame(preds)
        dfs.append(df)

    df = combine_disease_groups(dfs, prediction = not evaluate)
    ################ CONSTRUCT X-y pair, aggregate, impute ##############################
    if evaluate:
        X = df.drop(["patient_id", "target"], 1)
        target = df.target
    else:
        X = df.drop(["patient_id"], 1)
    IMPUTER = PIPELINE['Imputer']
    X=IMPUTER.transform(X)
    # ############
    CONDITIONS = PIPELINE["CONDITIONS"]
    # Get back the column names
    if evaluate:
        X = pd.DataFrame(X, columns = df.drop(["patient_id", "target"], 1).columns)
    else:
        X = pd.DataFrame(X, columns = df.drop(["patient_id"], 1).columns)
    X['patient_id'] = df["patient_id"]
    X["MEAN"] = X[CONDITIONS].mean(1)
    X["STD"] = X[CONDITIONS].std(1)
    X["RANGE"] = X[CONDITIONS].max(1) - X[CONDITIONS].min(1)
    ##########################
    IDSS = X['patient_id']
    X.drop(["patient_id"], 1, inplace = True)
    LGBM = PIPELINE['LGBM']
    lgbm_predict = LGBM.predict_proba(X, num_iteration=LGBM.best_iteration_)
    XXX = X.copy()
    XXX['patient_id'] = IDSS
    XXX['risk'] = lgbm_predict[:,1]
    if evaluate:
        XXX['target'] = [int(i) for i in target]
    return XXX

def predict(patients_file,
            out_file,
            disease_groups,
            phenotypes,
            model_path,
            pfsa_path,
            separator,
            delimiter,
            n_first_weeks = 9999,
            save_features = False,
            procdb = 'bin/procdb',
            verbose = 0,
            optimal_thresholds = {'M': .017379, 'F': .00617},
            dx_input = "dx_input.dat",  # intermediary file path to store converted DX source
            work_dir = "STANDALONE",
            T = 1, N = 2, z = 1, a = 5): # params for the procdb

    if not work_dir in os.listdir():
        os.makedirs(work_dir)
    ## Step 1. Convert files into DX format, write into the file
    with open(patients_file, 'r') as input_file:
        patients = input_file.readlines()
        dx_already = all(["|" in patients[0], "^" in patients[0], ":" in patients[0]])
        with open(os.path.join(work_dir, dx_input), 'w') as dx_file:
            for i, line in enumerate(patients):
                if dx_already:
                    dx_file.write(line)
                else:
                    dx_file.write(convert_to_dx(i + 1, line, separator, delimiter))

    ## Step 2. Convert DX files to ternary encodings, then to dfs

    dx_FILE = os.path.join(work_dir, dx_input)
    procdb_line = "%s -D %s -P %s -T %d -N %d -I encoding_%s.dat -w %s -z %d -a %d"

    for KEY in disease_groups:
        bash(procdb_line % (procdb, dx_FILE,
                         phenotypes % KEY,
                         T, N, KEY,
                         work_dir, z, a), mute = 1)
    ENCODINGS = {}
    encoding_path = os.path.join(work_dir, "encoding_%s.dat")
    for KEY in disease_groups:
        DATAPATH = '%s/encoding_%s.dat' % (work_dir, KEY)
        ENCODING = optimize_memory(retrieve_raw_records(encoding_path % KEY, n_first_weeks, full = True))
        """"""
        ENCODING['record'] = [' '.join(A.split()[1:n_first_weeks+1]) for A in ENCODING['record']]
        """"""
        ENCODINGS[KEY] = ENCODING
        # Infer the gender from one of the homogenous records
        try:
            gender = list(ENCODINGS[KEY].head().gender)[0]
        except:
            continue
    ## Step 3. Produce features and predictions with dataframes
    LGBM = pickle.load( open(model_path + "/%s/LGBM.pickle" % gender, "rb" ) )
    NB = pickle.load( open(model_path + "/%s/NB.pickle" % gender, "rb" ) )
    MINMAX = pickle.load( open(model_path + "/%s/NB_SCALER.pickle" % gender, "rb" ) )
    IMPUTER = pickle.load( open(model_path + "/%s/IMPUTER.pickle" % gender, "rb" ) )
    ROC = pd.read_csv(model_path + "/%s/ROC.csv" % gender)
    predictions = run_saved_pipeline({
        "LENGTH": n_first_weeks,
        "CONDITIONS": disease_groups,
        "DATA": ENCODINGS,
        "LGBM": LGBM,
        "NB": NB,
        "Scaler": MINMAX,
        "Imputer": IMPUTER,
        "PFSA_POS": pfsa_path % gender + "/%s" + "/POS.pfsa",
        "PFSA_NEG": pfsa_path % gender + "/%s" + "/NEG.pfsa"
    }, evaluate = True, verbose = verbose,work_dir=work_dir)
    threshold = optimal_thresholds[gender]
    predictions['diagnosis'] = [int(pred > threshold) for pred in predictions.predicted_risk]
    predictions['confidence'] = pd.Series([p0(row[2], row[3],
                                              ROC) for row in predictions[['patient_id',
                                                                           'predicted_risk',
                                                                           'diagnosis']].itertuples()]).values
    predictions['relative_risk'] = predictions.predicted_risk/threshold
    if not save_features:
        OUT = predictions[['patient_id', 'predicted_risk', 'relative_risk', 'diagnosis', 'confidence']]
    else:
        OUT = predictions
    OUT.to_csv(out_file, index = False)
    return OUT

def detect_allzeros(threshold, diseases, work_dir):
    """ Return 0 for allzeros """
    allzeros = {}
    for disease in diseases:
        with open(os.path.join(work_dir, "encoding_%s.dat" % disease), 'r') as file:
            lines = file.readlines()
        for line in lines:
            data = line.split()
            patient_id = "A" + data[1]
            if patient_id not in allzeros.keys():
                allzeros[patient_id] = int(np.mean([int(i) for i in data[2:threshold+2]]) != 0)
    return pd.DataFrame({'patient_id': list(allzeros.keys()), "allzero": list(allzeros.values())})

def predict_with_confidence(patients_file,
            out_file,
            separator,
            delimiter,
            n_first_weeks = [9999],
            disease_groups = None,
            phenotypes = None,
            model_path = None,
            pfsa_path = None,
            procdb = None,
            verbose = 0,
            optimal_thresholds = {'M': .017379, 'F': .00617},
            dx_input = "dx_input.dat",  # intermediary file path to store converted DX source
            work_dir = None,
            T = 1, N = 2, z = 1, a = 5): # params for the procdb

    if disease_groups is None:
        disease_groups = ['Infectious_Disease','Cardiovascular',
                  'Development','Digestive','Endocrine',
                  'Hematologic','Immune','Integumentary',
                  'Metabolic', 'Musculoskeletal', 'Ophthalmological',
                  'Otic', 'Reproductive', 'Respiratory']
    if phenotypes is None:
        phenotypes=os.path.dirname(os.path.realpath(__file__))+'/PHENOTYPES/%s.dat'
    if procdb is None:
        procdb=os.path.dirname(os.path.realpath(__file__))+'/bin/procdb'
    if model_path is None:
        model_path=os.path.dirname(os.path.realpath(__file__))+'/MODELS'
    if pfsa_path is None:
        pfsa_path=os.path.dirname(os.path.realpath(__file__))+'/PFSA/%s'

    if work_dir is None:
        work_dir="/tmp/xxx/"
    #work_dir_=tempfile.TemporaryDirectory()
    #work_dir=work_dir_.name
    
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    ## Step 1. Convert files into DX format, write into the file
    with open(patients_file, 'r') as input_file:
        patients = input_file.readlines()
        dx_already = all(["|" in patients[0], "^" in patients[0], ":" in patients[0]])
        with open(os.path.join(work_dir, dx_input), 'w') as dx_file:
            for i, line in enumerate(patients):
                if dx_already:
                    dx_file.write(line)
                else:
                    dx_file.write(convert_to_dx(i + 1, line, separator, delimiter))

    ## Step 2. Convert DX files to ternary encodings, then to dfs

    dx_FILE = os.path.join(work_dir, dx_input)
    procdb_line = "%s -D %s -P %s -T %d -N %d -I encoding_%s.dat -w %s -z %d -a %d"
    
    for KEY in disease_groups:
        bash(procdb_line % (procdb, dx_FILE,
                         phenotypes % KEY,
                         T, N, KEY,
                         work_dir, z, a), mute = 1)
    multicols = []
    all_zeros = {}
    WEEK_PREDS = []
    for seq_length in n_first_weeks:
        all_zeros[str(seq_length)] = detect_allzeros(seq_length, disease_groups, work_dir)
        ENCODINGS = {}
        encoding_path = os.path.join(work_dir, "encoding_%s.dat")
        for KEY in disease_groups:
            DATAPATH = '%s/encoding_%s.dat' % (work_dir, KEY)
            ENCODING = optimize_memory(retrieve_raw_records(encoding_path % KEY, seq_length, full = True))
            """"""
            ENCODING['record'] = [' '.join(A.split()[1:seq_length+1]) for A in ENCODING['record']]
            ENCODING['all_zero'] = [int(np.mean([int(i) for i in seq.split()]) == 0) for seq in ENCODING['record']]
            """"""
            ENCODINGS[KEY] = ENCODING
            # Infer the gender from one of the homogenous records
            try:
                gender = list(ENCODINGS[KEY].head().gender)[0]
            except:
                continue
        ## Step 3. Produce features and predictions with dataframes
        LGBM = pickle.load( open(model_path + "/%s/LGBM.pickle" % gender, "rb" ) )
        NB = pickle.load( open(model_path + "/%s/NB.pickle" % gender, "rb" ) )
        MINMAX = pickle.load( open(model_path + "/%s/NB_SCALER.pickle" % gender, "rb" ) )
        IMPUTER = pickle.load( open(model_path + "/%s/IMPUTER.pickle" % gender, "rb" ) )
        # Save all columns on the first run
        predictions = run_saved_pipeline({
                "LENGTH": seq_length,
                "CONDITIONS": disease_groups,
                "DATA": ENCODINGS,
                "LGBM": LGBM,
                "NB": NB,
                "Scaler": MINMAX,
                "Imputer": IMPUTER,
                "PFSA_POS": pfsa_path % gender + "/%s" + "/POS.pfsa",
                "PFSA_NEG": pfsa_path % gender + "/%s" + "/NEG.pfsa"
            }, evaluate = False, verbose = verbose,work_dir=work_dir)[
                ['patient_id', 'risk']
        ]

        ROC_WEEKS = ['25', '50', '75', '100', '125', '150']
        ROCS = {seq_len : pd.read_csv(model_path + "/%s/ROC_%s_%s.csv" % (gender, gender, seq_len)) for seq_len in ROC_WEEKS}
        threshold = optimal_thresholds[gender]
        predictions['relative_risk'] = predictions['risk']/threshold
        predictions['diagnosis'] = [int(pred > threshold) for pred in predictions['risk']]
        predictions['confidence'] = pd.Series(
            [confidence(row[2],
                row[3], seq_length,
                ROCS) for row in predictions[['patient_id',
                                              'risk',
                                              'diagnosis']].itertuples()]).values
        multicols.append('confidence_%d' % seq_length)
        # ALL ZEROS
        predictions = predictions.merge(all_zeros[str(seq_length)], on = "patient_id")
        # Nullify predictions where input is all-zeros
        predictions['diagnosis'] *= predictions["allzero"] ###
        predictions['risk'] *= predictions["allzero"]
        predictions['relative_risk'] *= predictions["allzero"]
        predictions['confidence'] *= predictions["allzero"]
        predictions.drop(["allzero"], 1, inplace = True)
        predictions['week'] = seq_length
        WEEK_PREDS.append(predictions[['patient_id', 'week', 'risk', 'relative_risk', 'confidence']])
    return pd.concat(WEEK_PREDS).sort_values(by = "patient_id")

def get_nearest_fpr_tpr(ROC, threshold):
    ROC_curve = ROC.copy()
    ROC_curve['diff'] = [abs(i - threshold) for i in ROC_curve.threshold]
    optimum = ROC_curve.sort_values(by = "diff").head()
    return optimum['tpr'].iloc[0], optimum['fpr'].iloc[0]

def confidence(risk, diagnosis, week, ROCS):
    if str(week) in ROCS.keys():
        return p0(risk, diagnosis, week, ROCS)
    else:
        all_weeks = [int(i) for i in ROCS.keys()]
        # get two closest available numbers
        all_weeks.sort(key=lambda x: abs(x - week))
        weeks = sorted(all_weeks[:2])
        p0s = [p0(risk, diagnosis, w, ROCS) for w in weeks]
        return np.interp(week, weeks, p0s)

def p0(risk, diagnosis, week, ROCS):
    tpr, fpr = get_nearest_fpr_tpr(ROCS[str(week)], risk)
    if not diagnosis:
         return tpr
    else:
        return 1 - fpr
