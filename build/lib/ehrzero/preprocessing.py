import pandas as pd
import numpy as np
from random import randint
from math import floor
import os

#############################################
##  Functions for ICD-9 to CSV conversion  ##
#############################################

def files(path):  
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def optimize_memory(df):
    for col in df.columns:
        if df[col].dtype != object:

            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                #NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    

            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
    return df
        
def get_csv_dataset(DATAPATH, pattern, target = 'target', separate_x_y = True):
    data_dict = {'target': [], 'county': [], 'record': []}
    count = 0
    for i, filepath in enumerate(files(DATAPATH)):
        if pattern in filepath:
            with open(os.path.join(DATAPATH, filepath), 'r') as file:
                lines = file.readlines()
                for record_line in lines:
                        count += 1
                        data_dict['target'].append(int(filepath[1:4] == "POS"))
                        data_dict['county'].append(filepath[4:])
                        data_dict['record'].append(record_line)
    df = pd.DataFrame(data_dict)
    if not separate_x_y:
        return df
    else:
        return df.drop(target, 1), df[target]
    
def get_csv_dataset_integrated(DATAPATH):   
    data_dict = {'patient_id': [], 'gender': [], 'fips': [], 'record': [], 'target': []}
    with open(DATAPATH, 'r') as f:
        ao = f.readlines()
        for i, record in enumerate(ao):
            data = record.split()
            data_dict['patient_id'].append((data[1]))
            data_dict['gender'].append(data[0][0])
            data_dict['fips'].append(data[0][4:])
            data_dict['record'].append(" ".join(data[2:]))
            data_dict['target'].append(int(data[0][1:4] == "POS"))
    df = pd.DataFrame(data_dict)
    return df

def intc(i):
    try:
        return int(i)
    except:
        return 0 
    
def retrieve_raw_records(PATH, FIRST_N_WEEKS, full = False, integrated = False):
    if full:
        if integrated:
            test = get_csv_dataset_integrated(PATH)
        else:
            test = get_csv_dataset(PATH, 'M', separate_x_y = False)
    else:
        test = pd.read_csv('TEST_SET.csv')
    test['sequence'] = [[intc(i) for i in record[:-2].split(' ')][:FIRST_N_WEEKS] for record in test.record] #intc(i) to get integer

    X = pd.DataFrame(test.sequence.tolist(), columns=['SEQ_%i' % i for i in range(FIRST_N_WEEKS)])
    X['county'] = test.fips
    X['target'] = test.target
    X['gender'] = test.gender
    if integrated:
        X['patient_id'] = test.patient_id
    X.dropna(axis=0, inplace = True)
    return X


def optimal_cutoff(labels, preds):
    ####################################
    # Return the optimal binary cutoff value
    # given the ROC curve made by input labels and predictions
    # --------------------------------------------------------
    # The optimal cut off would be where tpr is high and fpr is low
    # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    ####################################
    fpr, tpr, cutoff = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(cutoff, index = i)})
    return float(roc.ix[(roc.tf-0).abs().argsort()[:1]]["thresholds"])

def augment_agg(X):
    mean = X.mean(axis = 1).reshape(X.shape[0],1)
    std = X.std(axis = 1).reshape(X.shape[0],1)
    rang = (X.max(axis = 1) - X.min(axis = 1)).reshape(X.shape[0],1)
    X = np.append(X,mean,1)
    X = np.append(X,std,1)
    X = np.append(X,rang,1)
    return X

def convert_to_dx(custom_id,seq,sep='|',delim=',',YR='2019'):
    '''
        Converts user input to DX format
        DX format: 000003878|  4|U|F|26099|2007^034.0 :  5:276|034.0 :...
        seq: input string
        sep: separator of sub-items within a record
        delim: separator for records
        Thus, for sep='|',delim=','
        a valid user input is:
        amy|M|01003,21|766.00,29|566.0,36|V0.20
        For sep=' ' and delim ':', the same input:
        amy M 01003:21 766.00:29 566.0:36 V0.20
        
        We can skip the FIPS (then 00000 is assumed),
        and we can skip the name/id, 
        then default SUBJECT is assumed
        returns: DX format input
    ''' 
    x=seq.split(delim)
    C=x[1:]
    
    hdr=x[0].split(sep)
    if len(hdr)==3:
        ID=hdr[0]
        G=hdr[1]
        FIPS=hdr[2]
    if len(hdr)==2:
        ID=hdr[0]
        G=hdr[1]
        FIPS='00000'
    if len(hdr)==1:
        G=hdr[0]
        ID = str(custom_id).rjust(9,"0")#str(randint(1,999999999)).rjust(9,"0")#'000000199'
        FIPS='00000'
    C_age_code = [(int(x.split(sep)[0]),
                 floor(int(x.split(sep)[0]) / 52 + 1),
                 x.split(sep)[1]) for x in C ]
    min_age = np.array([x[1] for x in C_age_code]).min()
    HDR = ID + "|  " + str(min_age) + '|U|' + G + "|" + FIPS + '|' + YR + '^'
    return (HDR+'|'.join([x[2]+':  '+str(x[1])+':'+str(x[0])  for x in C_age_code]) + "|").replace("\n", "") + '\r\n'

def files(path):  
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def reduce_memory(df):
    for col in df.columns:
        if df[col].dtype != object:
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                #NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
    return df
        
def get_csv_dataset(DATAPATH, pattern, target = 'target', separate_x_y = True):
    data_dict = {'target': [], 'county': [], 'record': []}
    count = 0
    for i, filepath in enumerate(files(DATAPATH)):
        if pattern in filepath:
            with open(os.path.join(DATAPATH, filepath), 'r') as file:
                lines = file.readlines()
                for record_line in lines:
                        count += 1
                        data_dict['target'].append(int(filepath[1:4] == "POS"))
                        data_dict['county'].append(filepath[4:])
                        data_dict['record'].append(record_line)
    df = pd.DataFrame(data_dict)
    if not separate_x_y:
        return df
    else:
        return df.drop(target, 1), df[target]
    
def get_csv_dataset_integrated(DATAPATH, WEEKS = 0):   
    data_dict = {'patient_id': [], 'gender': [], 'fips': [], 'record': [], 'target': []}
    with open(DATAPATH, 'r') as f:
        ao = f.readlines()
        for i, record in enumerate(ao):
            if i % 1000000 == 0 and i != 0:
                print(i)
            data = record.split()
            data_dict['patient_id'].append(("A" + data[1]))
            data_dict['gender'].append(data[0][0])
            data_dict['fips'].append(data[0][4:])
            if WEEKS:
                data_dict['record'].append(" ".join(data[2:(WEEKS + 2)]))
            else:
                data_dict['record'].append(" ".join(data[2:]))
            data_dict['target'].append(int(data[0][1:4] == "POS"))
    df = pd.DataFrame(data_dict)
    return df

def intc(i):
    try:
        return int(i)
    except:
        return 0 
    
def retrieve_raw_records(PATH, FIRST_N_WEEKS, full = False, all_rows = False, integrated = False):
    if full:
        data = get_csv_dataset_integrated(PATH)
        return data[["patient_id", "gender", "fips", "record", "target"]]
    else:
        if all_rows:
            if integrated:
                data = get_csv_dataset_integrated(PATH, FIRST_N_WEEKS)
            else:
                data = get_csv_dataset(PATH, 'M', separate_x_y = False)
        else:
            test = pd.read_csv('TEST_SET.csv')
            test['sequence'] = [[intc(i) for i in record[:-2].split(' ')][:FIRST_N_WEEKS] for record in test.record] 
            X = pd.DataFrame(test.sequence.tolist(), columns=['SEQ_%i' % i for i in range(FIRST_N_WEEKS)])
            X['county'] = test.fips
            X['target'] = test.target
            X['gender'] = test.gender
            if integrated:
                X['patient_id'] = test.patient_id
            return test
        #data.dropna(axis=0, inplace = True)
        return data.dropna(axis=0)[["patient_id", "gender", "fips", "record", "target"]]

def optimal_cutoff(labels, preds):
    ####################################
    # Return the optimal binary cutoff value
    # given the ROC curve made by input labels and predictions
    # --------------------------------------------------------
    # The optimal cut off would be where tpr is high and fpr is low
    # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    ####################################
    fpr, tpr, cutoff = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    #print("Area under the ROC curve : %f" % roc_auc)
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(cutoff, index = i)})
    return float(roc.ix[(roc.tf-0).abs().argsort()[:1]]["thresholds"])

def retrieve_diagnoses_before_x(x, records):
    aut_week = 0
    filtered_records = []
    # Get all the disease records that happened prior to (x+1)th week
    for record in records.split(":")[1:]:
        if "|" in record:
            vals = record.split("|")
            if vals[1].strip() in AUTISTIC:
                if not aut_week:
                    aut_week = int(vals[0].strip())
                else:
                    aut_week = min(aut_week, int(vals[0].strip()))
            elif int(vals[0].strip()) <= 150:
                filtered_records.append(record)
    if aut_week:
        filtered_records = [i for i in filtered_records if int(i.split("|")[0].strip()) < aut_week]
    return [i.split("|")[1].strip() for i in filtered_records]

def three_split(df, train_r=.6, proc_r=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_r * m)
    validate_end = int(proc_r * m) + train_end
    train = df.ix[perm[:train_end]]
    processing = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, processing, test

def get_data(conditions, path, gender, fips, jump):
    CONT = sp.county_cluster(fips, jump)
    dfs = []
    #KEYS = []
    train_sets = {}
    processing_sets = {}
    test_sets = {}
    for KEY in conditions:
        print(KEY)
        df = pd.read_csv(path % (gender, KEY))
        df = df[df.county.isin(CONT)]
        splits = three_split(df,
                            train_r = .4,
                            proc_r = .4)
        train_sets[KEY] = splits[0]
        processing_sets[KEY] = splits[1]
        test_sets[KEY] = splits[2]
    print(train_sets['Infectious_Disease'].shape)
    print(processing_sets['Infectious_Disease'].shape)
    print(test_sets['Infectious_Disease'].shape)
    return train_sets, processing_sets, test_sets

def build_pfsas(datasets, LLK):
    pfsas = []
    for key in datasets.keys():
        print(key)
        X = datasets[key].drop(['county'], 1)
        y = datasets[key].target
        Z=Z3Classifier(result_path = "RESULT_PATH", llk_path = LLK)
        #X['target'] = list(y)
        #print(">>", test.shape[0])
        Z.fit(X) # fits with BOTH feats and labels
        pfsas.append(Z)
    return pfsas

def obtain_lls(datasets, pfsas):
    dfs = []
    for key, Z in zip(datasets.keys(), pfsas):
        try:
            preds = {}
            test = datasets[key].drop(['county', 'target'], 1)
            target = datasets[key].target
            LL = Z.predict_loglike(test) # raw outpuit is zip
            preds[key + "_NEG"] = list(LL[0])
            preds[key + "_POS"] = list(LL[1])
            if len(preds[key + "_NEG"]) != test.shape[0] or len(preds[key + "_POS"]) != test.shape[0]:
                print("[ - - - ] Output size mismatch, retrying")
                continue
            df = pd.DataFrame(preds)
            df['patient_id'] = test['patient_id']
            df['target'] = list(target) #### ## #
            dfs.append(df)
            print(key)
        except:
            print('ERROR >> %s' % key)
            continue
            
    df = reduce(lambda left, right: pd.merge(left, right, on='patient_id',
                                                    how='outer').drop_duplicates(), dfs)
    print("SDDDDDDD")
    cols = list(df.columns)
    num = 0
    for i in range(len(cols)):
        if "target" in cols[i]:
            cols[i] = "target_%d" % num
            num += 1
    df.columns = cols
    print('KEOEK')
    targets = [df[var] for var in list(df.columns) if "target" in var]
    print("IKDPODK")
    full_target = reduce(lambda left, right : left.combine_first(right), targets)
    df.drop([i for i in list(df.columns) if "target" in i], 1, inplace = True)
    print('reduced')
    df['target'] = full_target
    if 'Infectious_Disease_NEG' in df.columns:
        df = df.dropna(subset=['Infectious_Disease_NEG'])
    return df

