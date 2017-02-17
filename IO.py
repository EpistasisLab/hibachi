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
#        AUTHOR:  Pete Schmitt (discovery (iMac)), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.1
#       CREATED:  02/06/2017 14:54:24 EST
#      REVISION:  Fri Feb 17 13:44:55 EST 2017
#==============================================================================
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from deap import gp
import pandas as pd
import csv
import numpy as np
###########################################################################
def plot_trees(best):
    """ create tree plots from best array """
    for i in range(len(best)):
        nodes, edges, labels = gp.graph(best[i])
        matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = graphviz_layout(g, prog="dot")
    
        f = plt.figure()
        nx.draw_networkx_nodes(g, pos, node_size=1500, 
                               font_size=8, node_color='lightblue')
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels, font_size=8)
        if (i < 10):
            plotfile = "tree_0" + str(i) + ".pdf"
        else:    
            plotfile = "tree_" + str(i) + ".pdf"
        plt.title(str(best[i]))
        f.savefig(plotfile)
###########################################################################
def plot_stats(df):
    matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
    ax = df.plot()
    fig = ax.get_figure()
    fig.savefig('stats.pdf')
###########################################################################
def plot_fitness(fit):
    fitdf = pd.DataFrame(fit, columns=['Fitness', 'GP Tree Size'])
    ax = fitdf.plot(x='GP Tree Size', y='Fitness', kind='scatter')
    fig = ax.get_figure()
    fig.savefig('fitness.pdf')
###########################################################################
def create_file(data,label,outfile):
    """ append label as column to data then write out file with header """
    for i in range(len(data)):
        data[i].append(label[i])
        
    # create header
    header = []
    for i in range(len(data[0])-1):
        header.append('X' + str(i))
    header.append('Class')
        
    # print data to results.tsv
    datadf = pd.DataFrame(data,columns=header) # convert to DF
    datadf.to_csv(outfile, sep='\t', index=False)
###########################################################################
def read_file(fname):
    """ return both data and x
        data = rows of instances
        x is data transposed to rows of features """
    with open(fname) as data:
        dataReader = csv.reader(data, delimiter='\t')
        data = list(list(int(elem) for elem in row) for row in dataReader)
    #transpose into x
    inst_length = len(data[0])
    x = []
    for i in range(inst_length):
        y = [col[i] for col in data]
        x.append(y)
    del y
    return data, x
###########################################################################
def read_file_np(fname):
    """ UNUSED: read data into numpy arrays """
    data = np.genfromtxt(fname, dtype=np.int, delimiter='\t') 
    x = data.transpose()
    return data, x
###########################################################################
def subsets(x,percent):
    """ take a subset of 25% of x """
    p = percent / 100
    xa = np.array(x)
    subsample_indices = np.random.choice(xa.shape[1], int(xa.shape[1] * p), 
                                         replace=False)
    return (xa[:, subsample_indices]).tolist()
###########################################################################
def getfolds(data,num):
    """ return num folds of size 1/num'th of x """
    folds = []
    fsize = end = int(len(data) / num)
    xa = np.array(data)
    np.random.shuffle(xa)
    xa = xa.transpose()
    start = 0
    for i in range(num):
        folds.append(xa[:,start:end])
        start += fsize
        end += fsize  
    return folds
###########################################################################
def addnoise(x,pcnt):
    """ add some percentage of noise to data """
    xa = np.array(x)
    val = pcnt/100
    rep = {}
    rep[0] = [1,2]
    rep[1] = [0,2]
    rep[2] = [0,1]

    for i in range(len(xa)):
        indices = np.random.choice(xa.shape[1], int(xa.shape[1] * val), 
                                   replace=False)
        for j in list(indices):
            xa[i][j] = np.random.choice(rep[xa[i][j]])

    return xa.tolist()
###########################################################################
def printf(format, *args):
    """ works just like the C/C++ printf function """
    import sys
    sys.stdout.write(format % args)
    sys.stdout.flush()
