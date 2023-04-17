#!/bin/bash
{
source ~/.bashrc
conda activate ss


cudas=(0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4)
max=25

dataset='hm3d'


------ Rotation ------ #
num_scene=2500
num_source=3
num_camera=500
num_angle=40
num_view=4
outdir='hm3d-4view-rotation'
flag='--rotation-only --high-resol'

# ------ Rotation with small translation ------ #
# num_scene=2000
# num_source=2
# num_camera=400
# num_angle=40
# num_view=3
# outdir='hm3d-3view-smalltrans'
# flag='--rotation-only --add-small-translation --high-resol'






for (( i=0; i < $max; i++))
do {
    echo "Process \"$i\" started";
    CUDA_VISIBLE_DEVICES=${cudas[$i]} nice -n 0 python generate_audiosfm.py --dataset=$dataset --output-dir=$outdir --num-per-scene=$num_scene --num-source=$num_source --num-camera=$num_camera --num-view=$num_view --num-angle=$num_angle $flag --split=$i --total=$max & pid=$! 
} done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";

}