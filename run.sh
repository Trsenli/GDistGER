
#!/bin/bash
#
# Script Name: run.sh
# Description: The run shell script of DistGER.
# Author: lzl
# Date: 2023/03/20
#
# Usage:
#  ./your_script_name.sh [option1] [option2] ...
#
# Options:
#  -h, --help    Displays this help message
#  -v, --version Displays version information
#
# Example Usage:
#  ./your_script_name.sh -v
#
# Dependencies:
#  List any dependencies your script requires to run
#
# Notes:
#  Any additional notes or information about your script
#

bin=./bin/huge_walk
graph=../dataset/binary/$1.data
node_num=4
other_option=" -o ./out/walk.txt --make-undirected \
    -eoutput ./out/$1_emb.txt -size 128 -iter 1 -threads 10 -window 10 -negative 5 -batch-size 21 -min-count 0 -sample 1e-3 -alpha 0.01 -debug 2 "

if [ $1 = "wiki" ];then
    mpiexec -n $node_num $bin -g $graph -v 7115 -w 7115 --min_L 20 --min_R 5 $other_option
elif [ $1 = "ytb" ];then
    mpiexec -n $node_num $bin -g $graph -v 1138499 -w 1138499 --min_L 20 --min_R 10 $other_option
fi;



