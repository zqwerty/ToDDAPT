#!/bin/bash

task=$1
model=$2
model_suffix=dialogbert/dapt/${model}

cmd="scripts/run_${task}_fs.sh /home/data/zhuqi/pre-trained-models/${model_suffix} save/finetune/${model_suffix}"
echo $cmd
eval $cmd
