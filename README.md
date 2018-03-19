# hibachi
Data simulation software that creates data sets with particular characteristics
```
usage: hib.py [-h] [-e EVALUATION] [-f FILE] [-g GENERATIONS]
              [-i INFORMATION_GAIN] [-m MODEL_FILE] [-o OUTDIR]
              [-p POPULATION] [-r RANDOM_DATA_FILES] [-s SEED] [-A]
              [-C COLUMNS] [-F] [-P PERCENT] [-R ROWS] [-S] [-T]

Run hibachi evaluations on your data

optional arguments:
  -h, --help            show this help message and exit
  -e EVALUATION, --evaluation EVALUATION
                        name of evaluation
                        [normal|folds|subsets|noise|oddsratio]
                        (default=normal) note: oddsration sets columns == 10
  -f FILE, --file FILE  name of training data file (REQ) filename of random
                        will create all data
  -g GENERATIONS, --generations GENERATIONS
                        number of generations (default=40)
  -i INFORMATION_GAIN, --information_gain INFORMATION_GAIN
                        information gain 2 way or 3 way (default=2)
  -m MODEL_FILE, --model_file MODEL_FILE
                        model file to use to create Class from; otherwise
                        analyze data for new model. Other options available
                        when using -m: [f,o,s,P,T]
  -o OUTDIR, --outdir OUTDIR
                        name of output directory (default = .) Note: the
                        directory will be created if it does not exist
  -p POPULATION, --population POPULATION
                        size of population (default=100)
  -r RANDOM_DATA_FILES, --random_data_files RANDOM_DATA_FILES
                        number of random data to use instead of files
                        (default=0)
  -s SEED, --seed SEED  random seed to use (default=random value 1-1000)
  -A, --showallfitnesses
                        show all fitnesses in a multi objective optimization
  -C COLUMNS, --columns COLUMNS
                        random data columns (default=3) note: evaluation of
                        oddsratio sets columns to 10
  -F, --fitness         plot fitness results
  -P PERCENT, --percent PERCENT
                        percentage of case for case/control (default=25)
  -R ROWS, --rows ROWS  random data rows (default=1000)
  -S, --statistics      plot statistics
  -T, --trees           plot best individual trees
```
  Prerequisites:
```
  graphviz libraries
  Python 3.4+ and packages
    argparse
    collections
    csv
    deap
    glob
    itertools
    math
    matplotlib
    networkx
    numpy 
    operator
    os
    pandas
    pygraphviz
    random
    scipy
    sklearn
    sys
    time
```
