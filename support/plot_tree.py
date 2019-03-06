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
import seaborn as sns; sns.set(color_codes=True)
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from deap import gp
import numpy as np
import pandas as pd
###############################################################################
def plot_tree(best,rseed,outdir):
    """ create tree plots from best array """
    nodes, edges, labels = gp.graph(best)
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
    plotfile = outdir + "tree_" + str(rseed) + ".pdf"
    plt.title(str(best))
    f.savefig(plotfile)
