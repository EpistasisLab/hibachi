#!/usr/bin/env python
#==============================================================================
#
#          FILE:  gendata.py
# 
#         USAGE:  gendata.py -C columns -R rows -s seed_value
# 
#   DESCRIPTION:  generate random data
#                 defaults:  -C 3 -R 1000 -s 100
# 
#       UPDATES:  
#
#        AUTHOR:  Pete Schmitt (debtfree), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.0
#       CREATED:  Wed Jul 18 10:30:07 EDT 2018
#      REVISION:  
#==============================================================================
import pandas as pd
import csv
import numpy as np
import pandas as pd
import argparse
import sys
import os
###############################################################################
def get_arguments():
    options = dict()

    parser = argparse.ArgumentParser(description = \
        "Create random data")

    parser.add_argument("-s", "--seed", type=int, 
            help="random seed to use (default=random value 1-1000)")
    parser.add_argument("-R", "--rows", type=int, 
            help="random data rows (default=1000)")
    parser.add_argument("-C", "--cols", type=int, 
            help="random data columns (default=3)")
    
    args = parser.parse_args()

    if(args.seed == None):
        options['seed'] = 100
    else:
        options['seed'] = args.seed

    if(args.rows == None):
        options['rows'] = 1000
    else:
        options['rows'] = args.rows

    if(args.cols == None):
        options['cols'] = 3
    else:
        options['cols'] = args.cols

    return options
###############################################################################
def get_random_data(rows, cols, seed=100):
    """ return randomly generated data is shape passed in """
    if seed != None: np.random.seed(seed)
    data = np.random.randint(0,3,size=(rows,cols))
#   x = data.transpose()
    return pd.DataFrame(data)
###############################################################################
def printf(format, *args):
    """ works just like the C/C++ printf function """
    sys.stdout.write(format % args)
    sys.stdout.flush()
###############################################################################
# Start here:
options = get_arguments()

rows = options['rows']
cols = options['cols']
seed = options['seed']

data = get_random_data(rows,cols,seed)
fname = "random_R" + str(rows) + "_C" + str(cols) + "_s" + str(seed) + '.tsv'
data.to_csv(fname, sep='\t', header=False, index=False)

printf("\nRows: %5d\nCols: %5d\nSeed: %5d\n", rows, cols, seed)
printf("\nFile %s written\n\n", fname)
