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
sync_size_cal=$(echo "2^$2" | bc)
# sync_size_cal=32768
bin=./bin/huge_walk
graph=../dataset/binary/$1.data
train_graph=../dataset/train_bin/$1_train.data
node_num=1
other_option=" -o ./out/walk.txt --make-undirected \
    -eoutput ./out/$1_emb.txt -size 300 -iter 3 -threads 10 -window 10 -negative 5 -batch-size 21 -min-count 0 -sample 1e-3 -alpha 0.01 -debug 2 -sync-size $sync_size_cal"

# nvprof="nvprof --metrics sm_efficiency,achieved_occupancy --profile-from-start off "
# nvprof="nvprof "

if [ $1 = "wiki" ]; then
	# mpiexec -n $node_num $bin -g $graph -v 7115 -w 7115 --min_L 20 --min_R 5 $other_option
	$nvprof $bin -g $graph -v 7115 -w 7115 --min_L 20 --min_R 5 $other_option
elif [ $1 = "ytb" ]; then
	# mpiexec -n $node_num $bin -g $graph -v 1138499 -w 1138499 --min_L 20 --min_R 10 $other_option
	$nvprof $bin -g $train_graph -v 1138499 -w 1138499 --min_L 20 --min_R 1 $other_option
elif [ $1 = "soc" ]; then
	# mpiexec -n $node_num $bin -g $graph -v 1632803  -w 1632803  --min_L 20 --min_R 5 $other_option
	$nvprof $bin -g $graph -v 1632803 -w 1632803 --min_L 20 --min_R 1 $other_option
elif [ $1 = "LJ" ]; then
	# mpiexec -n $node_num $bin -g $graph -v 2238731 -w 2238731 --min_L 20 --min_R 5 $other_option
	$nvprof $bin -g $graph -v 2238731 -w 2238731 --min_L 20 --min_R 1 $other_option
elif [ $1 = "com" ]; then
	# mpiexec -n $node_num $bin -g $graph -v 3072441  -w 3072441  --min_L 20 --min_R 5 $other_option
	$nvprof $bin -g $graph -v 3072441 -w 3072441 --min_L 20 --min_R 1 $other_option
elif [ $1 = "twt" ]; then
	mpiexec -n $node_num $bin -g $graph -v 41652230 -w 41652230 --min_L 20 --min_R 5 $other_option
fi

echo $sync_size_cal
