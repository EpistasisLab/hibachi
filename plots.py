#!/usr/bin/env python
#==============================================================================
#
#          FILE:  plots.py
# 
#         USAGE:  import plots (from hib.py)
# 
#   DESCRIPTION:  graphing routines.  
# 
#       UPDATES:  
#        AUTHOR:  Pete Schmitt (discovery (iMac)), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.0
#       CREATED:  Tue Mar 21 13:12:46 EDT 2017
#      REVISION:  
#==============================================================================
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set(color_codes=True)
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from deap import gp
import numpy as np
import pandas as pd
###############################################################################
def plot_tree(best,rseed,outdir):
    """ create tree plots from best array """
    nodes, edges, labels = gp.graph(best)
    matplotlib.rcParams['figure.figsize'] = (15.0, 15.0)
    
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")
    
    f = plt.figure()
    nx.draw_networkx_nodes(g, pos, node_size=1500, 
                           font_size=7, node_color='lightblue')
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels, font_size=7)
    plotfile = outdir + "tree_" + str(rseed) + ".pdf"
    plt.title(str(best))
    f.savefig(plotfile)
###############################################################################
#def plot_trees(best):
#    """ create tree plots from best array """
#    for i in range(len(best)):
#        nodes, edges, labels = gp.graph(best[i])
#        matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
#
#        g = nx.Graph()
#        g.add_nodes_from(nodes)
#        g.add_edges_from(edges)
#        pos = graphviz_layout(g, prog="dot")
#    
#        f = plt.figure()
#        nx.draw_networkx_nodes(g, pos, node_size=1500, 
#                               font_size=7, node_color='lightblue')
#        nx.draw_networkx_edges(g, pos)
#        nx.draw_networkx_labels(g, pos, labels, font_size=7)
#        if (i < 10):
#            plotfile = "tree_0" + str(i) + ".pdf"
#        else:    
#            plotfile = "tree_" + str(i) + ".pdf"
#        plt.title(str(best[i]))
#        f.savefig(plotfile)
###############################################################################
def plot_stats(df,statfile):
    matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
    ax = df.plot()
    fig = ax.get_figure()
    fig.savefig(statfile)
###############################################################################
def plot_fitness(fit,fname):
    fitdf = pd.DataFrame(fit, columns=['Fitness', 'GP Tree Size'])
    ax = fitdf.plot(x='GP Tree Size', y='Fitness', kind='scatter')
    fig = ax.get_figure()
    fig.savefig(fname)
###############################################################################
def plot_bars(objects, evaluate, best0, best1, infile, rndnum):
    width = 0.35
    y_pos = np.arange(len(objects))
    f = plt.figure()
    plt.bar(y_pos, best0, width, color='b', align='center', 
            alpha=0.5, label='best[0]');
    plt.bar(y_pos+width, best1, width, color='g', align='center', 
            alpha=0.5, label='best[1]');
    plt.xticks(y_pos, objects);
    title = evaluate + ' - ' + infile[:7] + " - rseed: " + rndnum
    plotfile = evaluate + '-' + infile[:7] + "-rseed-" + rndnum + '.pdf'
    plt.title(title);
    plt.legend(loc='upper right')
    plt.ylim(0,1,.05)
    f.savefig(plotfile)
###############################################################################
def plot_hist(data, evaluate, infile, rndnum):
    f = plt.figure()
    count = len(data)
    plt.hist(data, 200, normed=1, alpha=0.75)
    xlab = "Standard Deviation (" + str(count) + ")"
    plt.xlabel(xlab);
    plt.ylabel('Count');
    title = ("std - " + evaluate + " - " + infile + ' - rseed: ' + str(rndnum))
    plotfile = ("std-"+evaluate+"-"+infile+'-rseed-'+str(rndnum)+'.pdf')
    plt.title(title);
    f.savefig(plotfile)
###############################################################################
