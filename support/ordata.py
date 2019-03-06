#!/usr/bin/env python3
#===============================================================================
#
#          FILE:  ordata.py
# 
#         USAGE:  ./ordata.py 
# 
#   DESCRIPTION:  create models file containing all models
#         Notes:  run this second 
#        AUTHOR:  Pete Schmitt (gemini), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.1
#       CREATED:  Mon Aug 14 13:35:40 EDT 2017
#      REVISION:  Fri Mar  9 13:49:58 EST 2018
#===============================================================================

import pandas as pd
import numpy as np
import glob
import ast
import os
import sys
G='g50'
MODELS = '/data/hibachi/models_all10-' + G + '.txt'
RESULTS = "/data/hibachi/results-" + G + "/???/or*"
files = glob.glob(RESULTS)
files = sorted(files)
theList = ['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9']
seed = 501
out = open(MODELS,'w')
out.write("Seed\tIndividual\tModel\n")
for f in files:
    print(f)
    base = os.path.basename(f)
    base = os.path.splitext(base)[0]
    df = pd.read_csv(f, sep='\t')
    rows = len(df)
    print('rows=', rows)
    
    for i in range(rows):
        if not all(x in df.Model[i] for x in theList): continue
        out.write(str(seed).zfill(3))
        out.write('\t')
        out.write(str(i).zfill(3))
        out.write('\t')
        out.write(df.Model[i])
        out.write('\n')
    seed += 1
out.close()
