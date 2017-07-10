#!/usr/bin/env python3
#===============================================================================
#
#          FILE:  hib.py
# 
#         USAGE:  ./hib.py [options]
# 
#   DESCRIPTION:  Data simulation software that creates data sets with 
#                 particular characteristics
#
#       OPTIONS:  ./hib.py -h for all options
#
#  REQUIREMENTS:  python >= 3.5, deap, scikit-mdr, pygraphviz
#          BUGS:  Damn ticks!!
#       UPDATES:  170224: try/except in evalData()
#                 170228: files.sort() to order files
#                 170313: modified to use IO.get_arguments()
#                 170319: modified to use evals for evaluations
#                 170320: modified to add 1 to data elements before processing
#                 170323: added options for plotting
#                 170410: added call to evals.reclass_result() in evalData()
#                 170417: reworked post processing of new random data tests
#                 170422: added ability for output directory selection
#                         directory is created if it doesn't exist
#                 170510: using more protected operators from operators.py
#                 170621: import information gains from local util.py to
#                         avoid unnecessary matplotlib import
#                 170626: added equal and not_equal operators
#                 170706: added option to show all fitnesses
#                 170710: added option to process given model
#                         writes out best model to model file
#       AUTHORS:  Pete Schmitt (discovery), pschmitt@upenn.edu
#                 Randy Olson, olsonran@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.2.0
#       CREATED:  02/06/2017 14:54:24 EST
#      REVISION:  Mon Jul 10 15:07:09 EDT 2017
#===============================================================================
from deap import algorithms, base, creator, tools, gp
from utils import three_way_information_gain as three_way_ig
from utils import two_way_information_gain as two_way_ig
import IO
import evals
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
    print("Python version 3.5 or later is HIGHLY recommended")
    print("speed, accuracy and reproducibility.")


labels = []
all_igsums = []
#results = []
start = time.time()

options = IO.get_arguments()
infile = options['file']
evaluate = options['evaluation']
population = options['population']
generations = options['generations']
rdf_count = options['random_data_files']
ig = options['information_gain']
rows = options['rows']
cols = options['columns']
Stats = options['statistics']
Trees = options['trees']
Fitness = options['fitness']
prcnt = options['percent']
outdir = options['outdir']
showall = options['showallfitnesses']
model_file = options['model_file']
if Fitness or Trees or Stats:
    import plots
#
# set up random seed
#
if(options['seed'] == -999):
    rseed = random.randint(1,1000)
else:
    rseed = options['seed']
random.seed(rseed)
np.random.seed(rseed)
#
# Read/create the data and put it in a list of lists.
# data is normal view of columns as features
# x is transposed view of data
#
if infile == 'random':
    data, x = IO.get_random_data(rows,cols,rseed)
else:
    data, x = IO.read_file(infile)
    rows = len(data)
    cols = len(x)

inst_length = len(x)
###############################################################################
# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, inst_length), 
                            float, "X")
# basic operators 
pset.addPrimitive(ops.addition, [float,float], float)
pset.addPrimitive(ops.subtract, [float,float], float)
pset.addPrimitive(ops.multiply, [float,float], float)
pset.addPrimitive(ops.safediv, [float,float], float)
pset.addPrimitive(ops.modulus, [float,float], float)
pset.addPrimitive(ops.plus_mod_two, [float,float], float)
# logic operators 
pset.addPrimitive(ops.equal, [float, float], float)
pset.addPrimitive(ops.not_equal, [float, float], float)
pset.addPrimitive(ops.gt, [float, float], float)
pset.addPrimitive(ops.lt, [float, float], float)
pset.addPrimitive(ops.AND, [float, float], float)
pset.addPrimitive(ops.OR, [float, float], float)
pset.addPrimitive(ops.xor, [float,float], float)
# bitwise operators 
pset.addPrimitive(ops.bitand, [float,float], float)
pset.addPrimitive(ops.bitor, [float,float], float)
pset.addPrimitive(ops.bitxor, [float,float], float)
# unary operators 
pset.addPrimitive(op.abs, [float], float)
pset.addPrimitive(ops.NOT, [float], float)
pset.addPrimitive(ops.factorial, [float], float)
pset.addPrimitive(ops.left, [float,float], float)
pset.addPrimitive(ops.right, [float,float], float)
# large operators 
pset.addPrimitive(ops.power, [float,float], float)
pset.addPrimitive(ops.logAofB, [float,float], float)
pset.addPrimitive(ops.permute, [float,float], float)
pset.addPrimitive(ops.choose, [float,float], float)
# misc operators 
pset.addPrimitive(min, [float,float], float)
pset.addPrimitive(max, [float,float], float)
# terminals 
randval = "rand" + str(random.random())[2:]  # so it can rerun from ipython
pset.addEphemeralConstant(randval, lambda: random.random() * 100, float)
pset.addTerminal(0.0, float)
pset.addTerminal(1.0, float)
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
def evalData(individual, xdata, xtranspose):
    """ evaluate the individual """
    result = []
    igsums = np.array([])
    x = xdata
    data = xtranspose

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # Create class possibility.  
    # If class has a unique length of 1, toss it.
    try:
        result = [(func(*inst[:inst_length])) for inst in data]
    except:
        return -sys.maxsize, sys.maxsize

    if (len(np.unique(result)) == 1):
        return -sys.maxsize, sys.maxsize
    
     
    if evaluate == 'normal':
        rangeval = 1

    elif evaluate == 'folds':
        rangeval = numfolds = 10  # must be equal
        folds = evals.getfolds(x, numfolds)

    elif evaluate == 'subsets':
        rangeval = 10
        percent = 25

    elif evaluate == 'noise':
        rangeval = 10
        percent = 10

    result = evals.reclass_result(x, result, prcnt)
#   results.append(result)

    for m in range(rangeval):
        igsum = 0 
        if evaluate == 'folds': 
            xsub = list(folds[m])

        elif evaluate == 'subsets': 
            xsub = evals.subsets(x,percent)

        elif evaluate == 'noise': 
            xsub = evals.addnoise(x,percent)

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
toolbox.register("evaluate", evalData, xdata = x, xtranspose=data)
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
def hibachi(pop,gen,rseed,showall):
    """ set up stats and population size,
        then start the process """
    MU, LAMBDA = pop, pop
    NGEN = gen 
    np.random.seed(rseed)
    random.seed(rseed)
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront(similar=pareto_eq)
    if showall:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
    else:
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
# run the program
##############################################################################
print('input data:  ' + infile)
print('data shape:  ' + str(rows) + ' X ' + str(cols))
print('random seed: ' + str(rseed))
print('prcnt cases: ' + str(prcnt) + '%')
print('output dir:  ' + outdir)
if(model_file == ""):
    print('population:  ' + str(population))
    print('generations: ' + str(generations))
    print('evaluation:  ' + str(evaluate))
    print('ign 2/3way:  ' + str(ig))
print("")
# 
# If model file, ONLY process the model
#
if(model_file != ""):
    individual = IO.read_model(model_file)
    func = toolbox.compile(expr=individual)
    result = [(func(*inst[:inst_length])) for inst in data]
    nresult = evals.reclass_result(x, result, prcnt)
    outfile = outdir + 'results_using_model_from-' + os.path.basename(model_file) 
    print('Write result to',outfile)
    IO.create_file(x,nresult,outfile)
    sys.exit(0)
#
# Start evaluation here
#
pop, stats, hof, logbook = hibachi(population,generations,rseed,showall)
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
if(infile == 'random'):
    file1 = 'random0'
else:
    file1 = os.path.splitext(os.path.basename(infile))[0]
#
# make output directory if it doesn't exist
#
if not os.path.exists(outdir):
    os.makedirs(outdir)

outfile = outdir + "results-" + file1 + "-" + str(rseed) + '-' 
outfile += evaluate + "-" + str(ig) + "way.txt" 
print("writing data with Class to", outfile)
labels.sort(key=op.itemgetter(0),reverse=True)     # sort by igsum (score)
IO.create_file(x,labels[0][1],outfile)       # use first individual
#
# write top model out to file
#
moutfile = outdir + "model-" + file1 + "-" + str(rseed) + '-' 
moutfile += evaluate + "-" + str(ig) + "way.txt" 
print("writing model to", moutfile)
IO.write_model(moutfile, best)
#
# test results against other data
#
#if rdf_count == 0:
#    files = glob.glob('data/in*')
#    files.sort()
#
#  Test remaining data files with best individual
#
save_seed = rseed
if(infile == 'random' or rdf_count > 0):
    print('number of random data to generate:',rdf_count)
    for i in range(rdf_count):
        rseed += 1
        D, X = IO.get_random_data(rows,cols,rseed)
        nfile = 'random' + str(i+1)
        print(nfile)
        individual = best[0]
        func = toolbox.compile(expr=individual)
        result = [(func(*inst[:inst_length])) for inst in D]
        nresult = evals.reclass_result(X, result, prcnt)
        outfile = outdir + 'model_from-' + file1 
        outfile += '-using-' + nfile + '-' + str(rseed) + '-' 
        outfile += str(evaluate) + '-' + str(ig) + "way.txt" 
        print(outfile)
        IO.create_file(X,nresult,outfile)
#else:
#    print('number of files:',len(files))
#    for i in range(len(files)):
#        rseed += 1
#        if files[i] == infile: continue
#        nfile = os.path.splitext(os.path.basename(files[i]))[0]
#        print(infile)
#        print()
#        D, X = IO.read_file(files[i]) #  new data file
#        print('input file:', files[i])
#        individual = best[0]
#        func = toolbox.compile(expr=individual)
#        result = [(func(*inst[:inst_length])) for inst in D]
#        nresult = evals.reclass_result(X, result, prcnt)
#        outfile = outdir + 'model_from-' + file1 + '-using-' + nfile + '-'
#        outfile += str(rseed) + '-' + nfile + '-'
#        outfile += str(evaluate) + '-' + str(ig) + "way.txt" 
#        print(outfile)
#        IO.create_file(X,nresult,outfile)
#
# plot data if selected
#
file = os.path.splitext(os.path.basename(infile))[0]
if Stats == True:
    statfile = outdir + "stats-" + file + "-" + evaluate 
    statfile += "-" + str(rseed) + ".pdf"
    print('saving stats to', statfile)
    plots.plot_stats(df,statfile)

if Trees == True:
    print('saving tree plot to ' + outdir + 'tree_' + str(save_seed) + '.pdf')
    plots.plot_tree(best[0],save_seed,outdir)

if Fitness == True:
    outfile = outdir
    outfile += "fitness-" + file + "-" + evaluate + "-" + str(rseed) + ".pdf"
    print('saving fitness plot to', outfile)
    plots.plot_fitness(fitness,outfile)
