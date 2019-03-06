#!/bin/bash
#===============================================================================
#
#          FILE:  runmodels.sh
#
#         USAGE:  ./runmodels.sh
#
#   DESCRIPTION:
#
#       OPTIONS:  ---
#  REQUIREMENTS:  ---
#          BUGS:  ---
#         NOTES:  ---
#        AUTHOR:  Pete Schmitt (gemini), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  1.0
#       CREATED:  08/14/2017 03:09:40 PM EDT
#      REVISION:  ---
#===============================================================================
cd ..
for i in /data/hibachi/models/*
do
    echo $i
    seed=`echo $i | cut -f3 -d-`
    echo $seed
    python3 hib.py -f random -s $seed -C 10 -m $i -o /data/hibachi/data
done
