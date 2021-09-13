#!/bin/bash

task=$1
model=$2
run_type=$3
model_suffix=dialogbert/dapt/${model}

cmd="scripts/run_${task}.sh /home/data/zhuqi/pre-trained-models/${model_suffix} save/${run_type}/${model_suffix}"
if [[ ${run_type} == "probe" ]]; then
    cmd+=" --fix_encoder"
fi
echo $cmd
eval $cmd
