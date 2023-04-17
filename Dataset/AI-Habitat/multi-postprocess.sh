#!/bin/bash
{
source ~/.bashrc
conda activate ss


max=40

dataset='hm3d-4view-rotation'
# dataset='hm3d-3view-smalltrans'

flag='--mix_reverb --indirect_ratio=0.2 --downsample'


for (( i=0; i < $max; i++))
do {
    echo "Process \"$i\" started";
    python postprocess.py --dataset=$dataset $flag --split=$i --total=$max & pid=$! 
} done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";
}