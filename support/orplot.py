#!/usr/bin/env python3
#===============================================================================
#
#          FILE:  orplot.py
# 
#         USAGE:  ./orplot.py 
# 
#   DESCRIPTION:  
# 
#        AUTHOR:  Pete Schmitt (discovery), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.0
#       CREATED:  08/07/2017 14:07:14 EDT
#      REVISION:  Wed Aug 16 13:54:41 EDT 2017
#===============================================================================

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import ast
import os
import sys
from textwrap import wrap

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)

files = glob.glob("/data/hibachi/results-g10/???/or*")
files = sorted(files)
#fixed = np.array([4,2,1.5,1.4,1.3,1.2,1.18,1.16,1.14,1.12])
fixed = np.array([1.5, 1.3, 1.2, 1.15, 1.1, 1.09, 1.08, 1.07, 1.06, 1.05])
theList = ['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9']
n = 1
for f in files:
    print(f)
    pltdir = '/data/hibachi/plots-g10/' + str(n).zfill(3) + '/'
    if not os.path.exists(pltdir):
        os.makedirs(pltdir)
    base = os.path.basename(f)
    base = os.path.splitext(base)[0]
    df = pd.read_csv(f, sep='\t')
    rows = len(df)
    print('rows=', rows)
    
    for i in range(rows):
        if not all(x in df.Model[i] for x in theList): continue
        fit = df.Fitness[i]
        sod = df.SOD[i]
        model = df.Model[i]
        model = wrap(model,90)
        mdl = ""
        for j in range(len(model)):
            mdl += model[j] + '\n'
        orlist = ast.literal_eval(df.OR_list[i])
        pf = plt.figure()
        ax = pf.add_subplot(1,1,1)
        ax.plot(orlist, 'r')
        ax.plot(fixed, 'b')
        label1 = "Fitness: " + str(fit)
        label1 += " --- Sum of Diffs: " + str(sod)
        plt.xlabel(label1)
        title = "Individual: " + str(i) + " -- Random Seed: " + str(n) 
        plt.title(title)
        red_patch = mpatches.Patch(color='red', label='Odds Ratio')
        blue_patch = mpatches.Patch(color='blue', label='Fixed Constant')
        ax.legend(handles=[red_patch, blue_patch])
        ax.set_ylim(ymin=0)
        ax.annotate(mdl, xy=(.1,.5), xytext=(.1,.5))
        plt.grid(True)
#       ax.annotate(label2, xy=(3,2.5), xytext=(3,2.5))
        plotfile = pltdir + base + "-" + str(i).zfill(3) + ".png"
        pf.savefig(plotfile)
        plt.close(pf)
    n += 1
