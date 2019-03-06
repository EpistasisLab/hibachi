#!/bin/bash
#===============================================================================
#
#          FILE:  create_models.sh
# 
#         USAGE:  ./create_models.sh 
# 
#   DESCRIPTION:  create models files based on seed and individual
#         NOTES:  run this third 
#        AUTHOR:  Pete Schmitt (gemini), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.0
#       CREATED:  08/14/2017 14:41:16 EDT
#      REVISION:  Fri Mar  9 14:47:53 EST 2018
#===============================================================================
G='g50'
while
    read line
do
    model=`echo $line | cut -c9-`
    seed=`echo $line | awk '{print $1}'`
    indi=`echo $line | awk '{print $2}'`
    file="/data/hibachi/models-$G/model-seed-$seed-ind-$indi.txt"
    echo $model > $file
    echo $file
done < /data/hibachi/models_all10-$G.txt
