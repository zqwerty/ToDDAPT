#!/usr/bin/env bash
set -e

devices="7"
pretrained_model_path="/home/data/zhuqi/pre-trained-models/dialogbert/dapt/1.0data_0506_lr1e-4_steps40000_bz32ac8_block256_warmup0/"
model_des="domain adaptive pretraining + task adaptive pretraining"

dataset_name[0]="hwu"
lr[0]=5e-5
steps[0]=1000
dataset_name[1]="banking"
lr[1]=5e-5
steps[1]=1000
dataset_name[2]="oos"
lr[2]=1e-4
steps[2]=1000
dataset_name[3]="multiwoz"
lr[3]=5e-5
steps[3]=10000
dataset_name[4]="restaurant8k"
lr[4]=2e-5
steps[4]=1000
dataset_name[5]="top"
lr[5]=5e-5
steps[5]=2000
dataset_name[6]="m2m"
lr[6]=5e-5
steps[6]=500
dataset_name[7]="dstc2"
lr[7]=1e-4
steps[7]=1000

bz=32
accum=8
block=256
warmup=$((steps/100*0))
date=0524


for ((i=4;i<8;i++));
do
    dataset="dataset_config_tapt_${dataset_name[i]}.json"
    output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/tapt/dapt_${dataset_name[i]}_${date}_lr${lr[i]}_steps${steps[i]}_bz${bz}ac${accum}_block${block}_warmup${warmup}"
    python run_pretrain.py \
        --cuda_visible_devices=$devices \
        --model_name_or_path=$pretrained_model_path \
        --dapt \
        --train_ratio=1.0 \
        --model_des=$model_des \
        --dataset_config=$dataset \
        --block_size=$block \
        --output_dir=$output_dir \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --evaluate_during_training \
        --per_gpu_train_batch_size=$bz \
        --per_gpu_eval_batch_size=$bz \
        --gradient_accumulation_steps=$accum \
        --learning_rate=${lr[i]} \
        --weight_decay=0.01 \
        --adam_epsilon=1e-6 \
        --max_steps=${steps[i]} \
        --warmup_steps=$warmup \
        --logging_steps=100 \
        --save_total_limit=1 \
        --seed=42
done
