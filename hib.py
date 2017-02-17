#!/usr/bin/env python
#===============================================================================
#
#          FILE:  hib.py
# 
#         USAGE:  ./hib.py <tab_delimited_input_file>
# 
#   DESCRIPTION:  Data simulation software that creates data sets with 
#                 particular characteristics
#       OPTIONS:  input_file [folds|subsets|noise]
#  REQUIREMENTS:  python >= 3.5, deap, scikit-mdr, pygraphviz
#          BUGS:  ---
#         Notes:  test with 10-fold CV
#        AUTHOR:  Pete Schmitt (discovery (iMac)), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.0
#       CREATED:  02/06/2017 14:54:24 EST
#      REVISION:  Wed Feb 15 15:21:00 EST 2017
#===============================================================================
from deap import algorithms, base, creator, tools, gp
from mdr.utils import three_way_information_gain as three_way_ig
import IO
import itertools
import numpy as np
import operator as op
import operators as ops
import os
import pandas as pd
import random
import sys
###############################################################################
if (sys.version_info[0] < 3):
    print("hibachi requires Python version 3.5 or newer")
    sys.exit(1)

try:
    infile = sys.argv[1]
except:
    print('no file argument')
    sys.exit(2)

try:
    evaluate = sys.argv[2]
except:
    print('running default normal evaluation')
    evaluate = 'normal'

labels = list()
# Read the data and put it in a list of lists.
# x is transposed view of data
data, x = IO.read_file(infile)
inst_length = len(x)
###############################################################################
# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, inst_length), 
                            bool, "X")
# boolean operators 
pset.addPrimitive(op.and_, [bool, bool], bool)
pset.addPrimitive(op.or_, [bool, bool], bool)
pset.addPrimitive(op.not_, [bool], bool)
# basic operators 
pset.addPrimitive(op.add, [float,float], float)
pset.addPrimitive(op.sub, [float,float], float)
pset.addPrimitive(op.mul, [float,float], float)
pset.addPrimitive(ops.safediv, [float,float], float)
pset.addPrimitive(ops.modulus, [float,float], float)
pset.addPrimitive(ops.plus_mod_two, [float,float], float)
# logic operators 
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

pset.addPrimitive(op.lt, [float, float], bool)
pset.addPrimitive(op.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)
pset.addPrimitive(ops.xor, [float,float], float)
# bitwise operators 
pset.addPrimitive(ops.bitand, [float,float], float)
pset.addPrimitive(ops.bitor, [float,float], float)
pset.addPrimitive(ops.bitxor, [float,float], float)
# unary operators 
pset.addPrimitive(op.abs, [float], float)
pset.addPrimitive(ops.factorial, [float], float)
pset.addPrimitive(ops.log10ofA, [float], float)
pset.addPrimitive(ops.log2ofA, [float], float)
pset.addPrimitive(ops.logEofA, [float], float)
# large operators 
pset.addPrimitive(ops.power, [float,float], float)
pset.addPrimitive(ops.logAofB, [float,float], float)
pset.addPrimitive(ops.permute, [float,float], float)
pset.addPrimitive(ops.choose, [float,float], float)
# misc operators 
#pset.addPrimitive(ops.minimum, [float,float], float)
#pset.addPrimitive(ops.maximum, [float,float], float)
pset.addPrimitive(min, [float,float], float)
pset.addPrimitive(max, [float,float], float)
pset.addPrimitive(ops.left, [float,float], float)
pset.addPrimitive(ops.right, [float,float], float)
# terminals 
randval = "rand" + str(random.random())[2:]  # so it can rerun from ipython
pset.addEphemeralConstant(randval, lambda: random.random() * 100, float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)
# creator 
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
# toolbox 
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
toolbox.register("individual",
                 tools.initIterate,creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
##############################################################################
def evalData(individual, training_data):
    """ evaluate the individual """
    result = list()
    igsums = np.array([])
    x = training_data
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # Create class possibility.  
    # If class has a unique length of 1, toss it.
    result = [int(func(*inst[:inst_length])) for inst in data]
    if (len(np.unique(result)) == 1):
        return -sys.maxsize, sys.maxsize
    
    if evaluate == 'folds':
        rangeval = numfolds = 10
        folds = IO.getfolds(data, numfolds)

    elif evaluate == 'subsets':
        rangeval = 10
        percent = 25

    elif evaluate == 'noise':
        rangeval = 10
        percent = 10

    else:  # normal 
        rangeval = 1

    for m in range(rangeval):
        igsum = 0 
        if evaluate == 'folds': 
            xsub = list(folds[m])
        elif evaluate == 'subsets': 
            xsub = IO.subsets(x,percent)
        elif evaluate == 'noise': 
            xsub = IO.addnoise(x,percent)
        else:  # normal
            xsub = x
    
        # Calculate information gain between data columns and result
        # and return mean of these calculations
        for i in range(inst_length):
            for j in range(i+1,inst_length):
                for k in range(j+1,inst_length):
                    igsum += three_way_ig(xsub[i], xsub[j], xsub[k], result)
                    
        igsums = np.append(igsums,igsum)
        
    igsum_avg = np.mean(igsums)
    labels.append((igsum_avg, result)) # save all results
    
    if len(individual) <= 1:
        return -sys.maxsize, sys.maxsize
    else:
        if evaluate == 'normal':
            return igsum, len(individual)
        else:
            return igsum_avg, len(individual)
##############################################################################
toolbox.register("evaluate", evalData, training_data=x)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
##############################################################################
def pareto_eq(ind1, ind2):
    """Determines whether two individuals are equal on the Pareto front
       Parameters (ripped from tpot's base.py)
        ----------
        ind1: DEAP individual from the GP population
         First individual to compare
        ind2: DEAP individual from the GP population
         Second individual to compare
        Returns
        ----------
        individuals_equal: bool
         Boolean indicating whether the two individuals are equal on
         the Pareto front
    """
    return np.all(ind1.fitness.values == ind2.fitness.values)
##############################################################################
def hibachi():
    """ set up stats and population size,
        then start the process """
    MU, LAMBDA = 500, 500
    #random.seed(64)
    pop = toolbox.population(n=MU)
#   hof = tools.ParetoFront()
    hof = tools.ParetoFront(similar=pareto_eq)
    stats = tools.Statistics(lambda ind: max(ind.fitness.values[0],0))
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaMuPlusLambda(pop,toolbox,mu=MU,lambda_=LAMBDA, 
                          cxpb=0.7, mutpb=0.3, ngen=40, stats=stats, 
                          verbose=True, halloffame=hof)
    
    return pop, stats, hof, log
##############################################################################
# run the program #
###################
pop, stats, hof, logbook = hibachi()
best = list()
fitness = list()

for ind in hof:
    best.append(ind)
    fitness.append(ind.fitness.values)

for i in range(len(hof)):
    print("Best", i, "=", best[i])
    print("Fitness", i, '=', fitness[i])

record = stats.compile(pop)
print("statistics:")
print(record)

df = pd.DataFrame(logbook)
del df['gen']
del df['nevals']

print("writing data with Class to results.tsv")
infile = os.path.splitext(infile)[0]
outfile = "results-" + infile + "-" + evaluate + ".tsv"
labels.sort(key=op.itemgetter(0),reverse=True)     # sort by igsum (score)
IO.create_file(data,labels[0][1],outfile)       # use first individual

print('saving stats to stats.pdf')
IO.plot_stats(df)

print('saving tree plots to tree_##.pdf')
IO.plot_trees(best)

print('saving fitness plot to fitness.pdf')
IO.plot_fitness(fitness)
