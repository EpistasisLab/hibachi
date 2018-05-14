#!/usr/bin/env python
#==============================================================================
#
#          FILE:  IO.py
# 
#         USAGE:  import IO (from hib.py)
# 
#   DESCRIPTION:  graphing and file i/o routines.  
# 
#       UPDATES:  170213: added subset() function
#                 170214: added getfolds() function
#                 170215: added record shuffle to getfolds() function
#                 170216: added addnoise() function
#                 170217: modified create_file() to name file uniquely
#                 170302: added plot_hist() to plot std
#                 170313: added get_arguments()
#                 170319: added addone()
#                 170329: added np.random.shuffle() to read_file_np() 
#                 170410: added option for case percentage
#                 170420: added option for output directory
#                 170706: added option for showing all fitnesses
#                 170710: added option to process given model
#                         added read_model and write_model
#                 180307: added oddsratio to evaluate options
#        AUTHOR:  Pete Schmitt (discovery (iMac)), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.13
#       CREATED:  02/06/2017 14:54:24 EST
#      REVISION:  Wed Mar  7 10:50:27 EST 2018
#==============================================================================
import pandas as pd
import csv
import numpy as np
import argparse
import sys
import os
###############################################################################
def printf(format, *args):
    """ works just like the C/C++ printf function """
    import sys
    sys.stdout.write(format % args)
    sys.stdout.flush()
###############################################################################
def get_arguments():
    options = dict()

    parser = argparse.ArgumentParser(description = \
        "Run hibachi evaluations on your data")

    parser.add_argument('-e', '--evaluation', type=str,
            help='name of evaluation [normal|folds|subsets|noise|oddsratio]' +
                 ' (default=normal) note: oddsration sets columns == 10')
    parser.add_argument('-f', '--file', type=str,
            help='name of training data file (REQ)' +
                 ' filename of random will create all data')
    parser.add_argument("-g", "--generations", type=int, 
            help="number of generations (default=40)")
    parser.add_argument("-i", "--information_gain", type=int, 
            help="information gain 2 way or 3 way (default=2)")
    parser.add_argument("-m", "--model_file", type=str, 
            help="model file to use to create Class from; otherwise \
                  analyze data for new model.  Other options available \
                  when using -m: [f,o,s,P]")
    parser.add_argument('-o', '--outdir', type=str,
            help='name of output directory (default = .)' +
            ' Note: the directory will be created if it does not exist')
    parser.add_argument("-p", "--population", type=int, 
            help="size of population (default=100)")
    parser.add_argument("-r", "--random_data_files", type=int, 
            help="number of random data to use instead of files (default=0)")
    parser.add_argument("-s", "--seed", type=int, 
            help="random seed to use (default=random value 1-1000)")
    parser.add_argument("-A", "--showallfitnesses", 
            help="show all fitnesses in a multi objective optimization",
            action='store_true')
    parser.add_argument("-C", "--columns", type=int, 
            help="random data columns (default=3) note: " +
                 "evaluation of oddsratio sets columns to 10")
    parser.add_argument("-F", "--fitness", 
            help="plot fitness results",action='store_true')
    parser.add_argument("-P", "--percent", type=int,
            help="percentage of case for case/control (default=25)")
    parser.add_argument("-R", "--rows", type=int, 
            help="random data rows (default=1000)")
    parser.add_argument("-S", "--statistics", 
            help="plot statistics",action='store_true')
    parser.add_argument("-T", "--trees", 
            help="plot best individual trees",action='store_true')

    args = parser.parse_args()

    if(args.file == None):
        printf("filename required\n")
        sys.exit()
    else:
        options['file'] = args.file
        options['basename'] = os.path.basename(args.file)
        options['dir_path'] = os.path.dirname(args.file)

    if(args.model_file != None):
        options['model_file'] = args.model_file
    else:
        options['model_file'] = ""

    if(args.outdir == None):
        options['outdir'] = "./"
    else:
        options['outdir'] = args.outdir + '/'

    if(args.seed == None):
        options['seed'] = -999
    else:
        options['seed'] = args.seed

    if(args.percent == None):
        options['percent'] = 25
    else:
        options['percent'] = args.percent
        
    if(args.population == None):
        options['population'] = 100
    else:
        options['population'] = args.population

    if(args.information_gain == None):
        options['information_gain'] = 2
    else:
        options['information_gain'] = args.information_gain

    if(args.random_data_files == None):
        options['random_data_files'] = 0
    else:
        options['random_data_files'] = args.random_data_files

    if(args.generations == None):
        options['generations'] = 40
    else:
        options['generations'] = args.generations

    if(args.evaluation == None):
        options['evaluation'] = 'normal'
    else:
        options['evaluation'] = args.evaluation
        if options['evaluation'] == 'oddsratio':
            args.columns = 10

    if(args.rows == None):
        options['rows'] = 1000
    else:
        options['rows'] = args.rows

    if(args.columns == None):
        options['columns'] = 3
    else:
        options['columns'] = args.columns

    if(args.showallfitnesses):
        options['showallfitnesses'] = True
    else:
        options['showallfitnesses'] = False

    if(args.statistics):
        options['statistics'] = True
    else:
        options['statistics'] = False

    if(args.trees):
        options['trees'] = True
    else:
        options['trees'] = False

    if(args.fitness):
        options['fitness'] = True
    else:
        options['fitness'] = False

    return options
###############################################################################
def get_random_data(rows, cols, seed=None):
    """ return randomly generated data is shape passed in """
    if seed != None: np.random.seed(seed)
    data = np.random.randint(0,3,size=(rows,cols))
    x = data.transpose()
    return data.tolist(), x.tolist()
###############################################################################
def create_file(x,result,outfile):
    d = np.array(x).transpose()    
    columns = [0]*len(x)
    # create columns names for variable number of columns.
    for i in range(len(x)):
        columns[i] = 'X' + str(i)
    
    df = pd.DataFrame(d, columns=columns)
    
    df['Class'] = result
    df.to_csv(outfile, sep='\t', index=False)
###############################################################################
def read_file(fname):
    """ return both data and x
        data = rows of instances
        x is data transposed to rows of features """
    data = np.genfromtxt(fname, dtype=np.int, delimiter='\t') 
    #np.random.shuffle(data) # give the data a good row shuffle
    x = data.transpose()
    return data.tolist(), x.tolist()
###############################################################################
def write_model(outfile, best):
    """ write top individual out to model file """
    f = open(outfile, 'w')
    f.write(str(best[0]))
    f.write('\n')
    f.close()
###############################################################################
def read_model(infile):
    f = open(infile, 'r')
    m = f.read()
    m = m.rstrip()
    f.close()
    return m
###############################################################################
def create_OR_table(best,fitness,seed,outdir,rowxcol,popstr,
                    genstr,evaluate,ig):
    """ write out odd_ratio and supporting data """
    fname = outdir + "or_sod_igsum-" + rowxcol + '-' 
    fname += 's' + str(seed).zfill(3) + '-'
    fname += popstr + '-' 
    fname += genstr + '-' 
    fname += evaluate + '-ig' + str(ig) + 'way.txt'
    f = open(fname, 'w')
    f.write("Individual\tFitness\tSOD\tigsum\tOR_list\tModel\n")
    for i in range(len(best)):
        f.write(str(i))
        f.write('\t')
        f.write(str(fitness[i][0]))
        f.write('\t')
        f.write(str(best[i].SOD))
        f.write('\t')
        f.write(str(best[i].igsum))
        f.write('\t')
        f.write(str(best[i].OR.tolist()))
        f.write('\t')
        f.write(str(best[i]))
        f.write('\n')

    f.close()
