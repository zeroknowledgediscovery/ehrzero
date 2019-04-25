import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from functools import reduce
import subprocess

def bash(command, mute = False):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if not mute:
        print(output)

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
                        data_dict['fips'].append(filepath[4:])
                        data_dict['record'].append(record_line)
    df = pd.DataFrame(data_dict)
    if not separate_x_y:
        return df
    else:
        return df.drop(target, 1), df[target]
    
    
def get_csv_dataset(DATAPATH):   
    data_dict = {'gender': [], 'fips': [], 'record': [], 'target': []}
    for filename in os.listdir(DATAPATH):
        with open(DATAPATH + "/" + filename, 'r') as f:
            ao = f.readlines()
            for record in ao:
                #data_dict['patient_id'].append(int(record[1]))
                data_dict['gender'].append(int(filename[0] == "M"))
                data_dict['target'].append(int(filename[1:4] == "POS"))
                data_dict['fips'].append(filename[4:])
                data_dict['record'].append(record)

    df = pd.DataFrame(data_dict)
    return df

def intc(i):
    try:
        return int(i)
    except:
        return 0 
    
def powerset(iterable, minlen = 0):
    """
        RETURN THE LIST OF ALL THE PROPER SUBSETS
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return [list(i) for i in chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1)) if len(i) > minlen]
    
def get_test_data(PATH, FIRST_N_WEEKS, full = False):
    if full:
        #test = get_csv_dataset(PATH, 'M', separate_x_y = False)
        test = get_csv_dataset(PATH)
    else:
        test = pd.read_csv('TEST_SET.csv')
    test['sequence'] = [[intc(i) for i in record[:-2].split(' ')][:FIRST_N_WEEKS] for record in test.record] #intc(i) to get integer
    X = pd.DataFrame(test.sequence.tolist(), columns=['SEQ_%i' % i for i in range(FIRST_N_WEEKS)])
    X['fips'] = test.fips
    X['target'] = test.target
    X['gender'] = test.gender
    #X['patient_id'] = test.patient_id
    X.dropna(axis=0, inplace = True)
    return X
    
def read_neighbor(sep=" "):
        '''
            Read neighbor fips file
        '''
        n_f = pd.DataFrame()
        with open(NEIGH_FILE, 'r') as f:
            for line in f:
                n_f = pd.concat([n_f,
                                 pd.DataFrame([tuple(line.strip().split(sep))])],
                                ignore_index=True)
        n_f.index = n_f[0]
        del n_f[0]
        n_f.index.name = 'fips'
        return n_f

def getNeighbors(fips, jump = 3):
    '''
        gets neighbors which might be more than
        one jump away
    '''
    neighbors = read_neighbor()
    a_x = list(set(np.array(neighbors.loc[fips].dropna()).astype(str)))
    while jump > 1:
        b_x = []
        for i in a_x:
            b_x = np.append(b_x,np.array(neighbors.loc[i].dropna()).astype(str))
        a_x = list(set(b_x))
        jump -= 1
    return a_x

def getFilenames(fips, jump,
                 gender='M',
                 cat='POS',):
    '''
       get filenames in specified format 
    '''
    fp = getNeighbors(fips, jump)
    return [gender+cat+i for i in fp]

NEIGH_FILE = 'NEIGHBOR_FIPS_EXT'

def county_cluster(FIPS = '45045', JUMP = 5):
    return [i[-5:] for i in getFilenames(FIPS, JUMP, gender='M', cat='NEG')]

def get_ranking(array, top = 4):
    array = np.array(array)
    rank = [i for i in np.sort(array) if i > 0]
    out = [int(i in rank[-top:]) for i in array]
    return pd.Series(out)

def combine_disease_groups(dfs, ID = "patient_id"):
                df = reduce(lambda left,right: pd.merge(left, right, on = ID,
                                                            how='outer').drop_duplicates(), dfs)
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

def format_disease_code(code):
    return code.ljust(6," ")