#!/bin/bash
#===============================================================================
#
#          FILE:  all.sh
# 
#         USAGE:  ./all.sh 
# 
#   DESCRIPTION:  grab all models that include all 10 variables
#         NOTES:  run first 
#        AUTHOR:  Pete Schmitt (gemini), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  1.1
#       CREATED:  08/12/2017 10:05:12 AM EDT
#      REVISION:  Fri Mar  9 12:54:35 EST 2018
#===============================================================================
G=g100
cd /data/hibachi/results-$G
for i in ???
do 
    cd $i
    echo $i
    grep Best OR* | grep X0 | grep X1 | grep X2 | grep X3 | grep X4 | grep X5 | grep X6 | grep X7 | grep X8 | grep X9 
    grep Best OR* | grep X0 | grep X1 | grep X2 | grep X3 | grep X4 | grep X5 | grep X6 | grep X7 | grep X8 | grep X9 > /data/hibachi/all10-$G/all$i.txt
    cd .. 
done
