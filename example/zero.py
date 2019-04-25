#!/usr/bin/python3

import sys
import os
import pandas as pd
from ehrzero import ehrzero as ehr
import argparse
from argparse import RawTextHelpFormatter
import tempfile

import warnings
warnings.filterwarnings("ignore")

MODULEDIR=os.path.dirname(os.path.abspath(ehr.__file__))

# =============================================================
#     REGISTER PARAMETERS
# =============================================================
banner='\n\
▀▀█─█▀▀─█▀▄─▄▀▀▄\n\
▄▀──█▀▀─██▀─█──█\n\
▀▀▀─▀▀▀─▀─▀──▀▀─'
zed='copyright 2019 zed.uchicago.edu'
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Zero-Knowledge Risk Oracle\n'+banner+'\n'+zed,formatter_class=RawTextHelpFormatter)
parser._optionals.title="Program Options"

parser.add_argument('-data', metavar="", dest='DATA',
                    action="store", type=str,
                    default='',
                    help="Input medical history filepath / medical history string")
parser.add_argument('-outfile', metavar="", dest='OUTFILE',
                    action="store", type=str,
                    default='predictions.csv',help="output predictions path")
parser.add_argument('-n_weeks', nargs = "+", metavar="", dest='N_FIRST_WEEKS',
                    action="store", type=int,
                    default=[9999],help="mumber of weeks to consider")
parser.add_argument('-Verbose', metavar="", dest='VERBOSE',
                    action="store", type=str2bool,nargs='?',const=True,
                    default=True,help="set to False to suppress output")
parser.add_argument('-brief', metavar="", dest='BRIEF',
                    action="store", type=str2bool,nargs='?',const=True,
                    default=True,help="set to False to print dataframe")
parser.add_argument('-BriefToFile', metavar="", dest='BRIEF_TO_FILE',
                    action="store", type=str,nargs='?',
                    default="",help="filename to print brief output")

if len(sys.argv[1:])==0:
    parser.print_help()
    parser.exit()
args=parser.parse_args()
# =======================
## INFER FROM THE DATA ITSELF

PFSA_PATH = os.path.abspath("ehrzero/PFSA/%s")
MODEL_PATH = os.path.abspath("ehrzero/MODELS")
PROCDB = os.path.abspath('ehrzero/bin/procdb')
phenotypes = os.path.abspath('ehrzero_/PHENOTYPES/%s.dat')
DISEASE_GROUPS = ['Infectious_Disease','Cardiovascular',
                  'Development','Digestive','Endocrine',
                  'Hematologic','Immune','Integumentary',
                  'Metabolic', 'Musculoskeletal', 'Ophthalmological',
                  'Otic', 'Reproductive', 'Respiratory']

if not os.path.exists(args.DATA):
    tmpsrc = tempfile.NamedTemporaryFile()
    with open(tmpsrc.name, 'w') as file:
        file.write(args.DATA)
        SOURCE = tmpsrc.name
else:
    SOURCE = args.DATA 
outDf=ehr.predict_with_confidence(SOURCE,
                                  args.OUTFILE,
                                  separator = " ",
                                  delimiter = ":",
                                  verbose = args.VERBOSE,
                                  n_first_weeks = args.N_FIRST_WEEKS)

if(args.VERBOSE):
    if(args.BRIEF):
        header = " ".join(list(outDf.columns))
        if(args.BRIEF_TO_FILE != ""):
            with open(args.BRIEF_TO_FILE, "a+") as text_file:
                text_file.write(header+"\n")
        else:
            print(header)
        for result in outDf.iterrows():
            row = result[1]
            result_str = ""
            result_str+="{:s} ".format(row['patient_id'])
            result_str+="{:d} ".format(row['week'])
            result_str+="{:.6f} ".format(row['risk'])
            result_str+="{:.6f} ".format(row['relative_risk'])
            result_str += "{:2.2f}".format(100 * row['confidence'])
            if(args.BRIEF_TO_FILE != ""):
                with open(args.BRIEF_TO_FILE, "a+") as text_file:
                    text_file.write(result_str+"\n")
            else:
                print(result_str)
    else:
        for confid in ['confidence_%d' % nweeks for nweeks in args.N_FIRST_WEEKS]:
            outDf[confid] *= 100
        print(outDf)
