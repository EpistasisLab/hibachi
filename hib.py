#!/usr/bin/env python
#===============================================================================
#
#          FILE:  hib.py
# 
#         USAGE:  ./hib.py <file.tsv> evaluate population generations
# 
#   DESCRIPTION:  Data simulation software that creates data sets with 
#                 particular characteristics
#       OPTIONS:  input_file [folds|subsets|noise]
#  REQUIREMENTS:  python >= 3.5, deap, scikit-mdr, pygraphviz
#          BUGS:  ---
#       UPDATES:  170224: try/except in evalData()
#                 170228: files.sort() to order files
#                 170313: modified to use IO.get_arguments()
#        AUTHOR:  Pete Schmitt (discovery (iMac)), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.0
#       CREATED:  02/06/2017 14:54:24 EST
#      REVISION:  Mon Mar 13 12:20:52 EDT 2017
#===============================================================================
from deap import algorithms, base, creator, tools, gp
from mdr.utils import three_way_information_gain as three_way_ig
from mdr.utils import two_way_information_gain as two_way_ig
import IO
import itertools
import glob
import numpy as np
import operator as op
import operators as ops
import os
import pandas as pd
import random
import sys
import time

###############################################################################
if (sys.version_info[0] < 3):
    print("hibachi requires Python version 3.5 or later")
    sys.exit(1)

labels = []
all_igsums = []
rseed = random.randint(1,1000)
start = time.time()

options = IO.get_arguments()
infile = options['file']
evaluate = options['evaluation']
population = options['population']
generations = options['generations']
ig = options['information_gain']
#
# Read the data and put it in a list of lists.
# x is transposed view of data
# select file randomly
#
if infile == 'random':
    files = glob.glob('data/in*')
    num_files = len(files)
    random_file = random.randint(0,num_files-1)
    infile = files[random_file]
    print('random file selected:',infile)
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
    try:
        result = [int(func(*inst[:inst_length])) for inst in data]
    except:
        return -sys.maxsize, sys.maxsize

    if (len(np.unique(result)) == 1):
        return -sys.maxsize, sys.maxsize
    
    if evaluate == 'folds':
        rangeval = 10
        numfolds = 10
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
        if(ig == 2):
            for i in range(inst_length):
                for j in range(i+1,inst_length):
                    igsum += two_way_ig(xsub[i], xsub[j], result)
        elif(ig == 3):
            for i in range(inst_length):
                for j in range(i+1,inst_length):
                    for k in range(j+1,inst_length):
                        igsum += three_way_ig(xsub[i], xsub[j], xsub[k], result)
                    
        igsums = np.append(igsums,igsum)
        
    igsum_avg = np.mean(igsums)
    labels.append((igsum_avg, result)) # save all results
    all_igsums.append(igsums)
    
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
def hibachi(pop,gen,rseed):
    """ set up stats and population size,
        then start the process """
    MU, LAMBDA = pop, pop
    NGEN = gen 
    random.seed(rseed)
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront(similar=pareto_eq)
    stats = tools.Statistics(lambda ind: max(ind.fitness.values[0],0))
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaMuPlusLambda(pop,toolbox,mu=MU,lambda_=LAMBDA, 
                          cxpb=0.7, mutpb=0.3, ngen=NGEN, stats=stats, 
                          verbose=True, halloffame=hof)
    
    return pop, stats, hof, log
##############################################################################
# run the program #
###################
print('input data:  ' + infile)
print('population:  ' + str(population))
print('generations: ' + str(generations))
print('evaluation:  ' + str(evaluate))
print('ign 2/3way:  ' + str(ig))

pop, stats, hof, logbook = hibachi(population,generations,rseed)
best = []
fitness = []
for ind in hof:
    best.append(ind)
    fitness.append(ind.fitness.values)

for i in range(len(hof)):
    print("Best", i, "=", best[i])
    print("Fitness", i, '=', fitness[i])

record = stats.compile(pop)
print("statistics:")
print(record)

tottime = time.time() - start
if tottime > 3600:
    IO.printf("\nRuntime: %.2f hours\n", tottime/3600)
elif tottime > 60:
    IO.printf("\nRuntime: %.2f minutes\n", tottime/60)
else:
    IO.printf("\nRuntime: %.2f seconds\n", tottime)
df = pd.DataFrame(logbook)
del df['gen']
del df['nevals']
#
# sys.exit(0)
#
file = os.path.splitext(os.path.basename(infile))[0]
outfile = "results-" + file + "-" + evaluate + "-" + str(rseed) + ".tsv"
print("writing data with Class to", outfile)
labels.sort(key=op.itemgetter(0),reverse=True)     # sort by igsum (score)
IO.create_file(data,labels[0][1],outfile)       # use first individual
#
file = os.path.splitext(os.path.basename(infile))[0]
statfile = "stats-" + file + "-" + evaluate + "-" + str(rseed) + ".pdf"
print('saving stats to', statfile)
IO.plot_stats(df,statfile)
#
#print('saving tree plots to tree_##.pdf')
#IO.plot_trees(best)
#
outfile = "fitness-" + file + "-" + evaluate + "-" + str(rseed) + ".pdf"
print('saving fitness plot to', outfile)
IO.plot_fitness(fitness,outfile)
#
# test results against other data
#
#all_igsums.append(np.array([0,0,0,0,0,0,0,0,0,0]))
files = glob.glob('data/in*')
files.sort()
D = [0] * len(files)
X = [0] * len(files)
#Dr,Xr = IO.read_file('data/rotated01-3.tsv') # rotated from input3.tsv
print()
#print('input file: rotated01-3.tsv')
#print("best[0]", evalData(best[0],Xr), end="\t")
#print('best[1]', evalData(best[1],Xr))
#
#  Test remaining data files with top 2 best individuals
#
print('number of files:',len(files))
for i in range(len(files)):
    if files[i] == infile: continue
    D[i],X[i] = IO.read_file(files[i]) #  new data file
    print()
    print('input file:', files[i])
    print('best[0]', evalData(best[0],X[i]))
    print('best[1]', evalData(best[1],X[i]))

print()
print('NORMAL evaluation (original input file):')
print()
#all_igsums.append(np.array([0,0,0,0,0,0,0,0,0,0]))
evaluate = 'normal'
print('input file:', infile)
print('best[0]', evalData(best[0],x))
print('best[1]', evalData(best[1],x))
print()

print('tree-0', best[0])
print('tree-1', best[1])
#
# save for manual processing
#
#ind_str = 'some individual string'
#individual = creator.Individual.from_string(ind_str, pset)
#func = toolbox.compile(expr=individual)
#
#  Plot standard deviations
#
std_igsums = np.array([])
for i in range(len(all_igsums)):
    std_igsums = np.append(std_igsums, np.std(all_igsums[i]))
infile = os.path.splitext(os.path.basename(infile))[0]
IO.plot_hist(std_igsums,evaluate,infile,rseed)
