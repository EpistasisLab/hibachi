#!/bin/bash
#===============================================================================
#
#          FILE:  all.sh
# 
#         USAGE:  ./all.sh 
# 
#   DESCRIPTION:  grab all models that include all 10 variables
#         NOTES:  run fourth 
#        AUTHOR:  Pete Schmitt (gemini), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1
#       CREATED:  Mon Mar 12 11:43:20 EDT 2018
#      REVISION:  
#===============================================================================
G=g50
cd /data/hibachi/results-$G
echo "SEED,IND,ENTROPY,SOD,COMPLEX"
for i in ???
do 
    SEED=$i
    cd $i
    FILE=`echo or_sod*`
    while
        read line
    do
        M=`echo "$line" | cut -f6`
        F=`echo "$line" | cut -f2`
        I=`echo "$line" | cut -f1`
        SOD=`echo "$line" | cut -f3`
        IGSUM=`echo "$line" | cut -f4`
        LINE="$SEED" 
        model=`echo $M | grep X0 | grep X1 | grep X2 | grep X3 | grep X4 | grep X5 | grep X6 | grep X7 | grep X8 | grep X9`
        if test "$model" = ""
        then
            continue
        else
            a=`echo $model | sed 's/(/ /g'`
            b=`echo $a | sed 's/)/ /g'`
            c=`echo $b | sed 's/,/ /g'`
            complex=`echo $c | wc -w`
            echo "$SEED,$I,$F,$SOD,${complex}"
        fi
    done < $FILE
    cd .. 
done
