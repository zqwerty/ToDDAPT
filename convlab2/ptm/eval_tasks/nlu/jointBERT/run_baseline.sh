#!/usr/bin/env bash
devices="6"
seeds=(42 726 2019)
train_ratios=(0.01 0.1 0.25 1.0)

pretrained_weights[0]="/home/data/zhuqi/pre-trained-models/bert-base-uncased"
output_dir[0]="/home/data/zhuqi/pre-trained-models/dialogbert/eval_nlu/multiwoz23/all/bert"
model_type[0]="bert"

pretrained_weights[1]="/home/data/zhuqi/pre-trained-models/tod-bert/ToD-BERT-mlm"
output_dir[1]="/home/data/zhuqi/pre-trained-models/dialogbert/eval_nlu/multiwoz23/all/tod-bert-mlm"
model_type[1]="tod-bert"

pretrained_weights[2]="/home/data/zhuqi/pre-trained-models/tod-bert/ToD-BERT-jnt"
output_dir[2]="/home/data/zhuqi/pre-trained-models/dialogbert/eval_nlu/multiwoz23/all/tod-bert-jnt"
model_type[2]="tod-bert"

echo "run on devices $devices"
for ((i=0;i<${#pretrained_weights[@]};i++));
do
    echo "${pretrained_weights[i]}"
    CUDA_VISIBLE_DEVICES=$devices python train.py \
        --seeds ${seeds[*]} \
        --train_ratios ${train_ratios[*]} \
        --do_train \
        --do_eval \
        --pretrained_weights ${pretrained_weights[i]} \
        --basemodel ${model_type[i]} \
        --data_dir "multiwoz/multiwoz23/all_data" \
        --output_dir ${output_dir[i]}
done