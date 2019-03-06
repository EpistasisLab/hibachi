#!/bin/bash
#===============================================================================
#
#          FILE:  empty.sh
# 
#         USAGE:  ./empty.sh 
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
#       CREATED:  08/17/2017 08:43:20 AM EDT
#      REVISION:  ---
#===============================================================================

cd /data/hibachi/plots-g50
for DIR in *
do
    if [ "$(ls -A $DIR)" ]; then
        :
    #echo "Take action $DIR is not Empty"
    else
        echo "$DIR is Empty"
        rmdir -v $DIR
    fi
done
