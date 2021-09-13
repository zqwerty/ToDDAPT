#!/usr/bin/env bash
set -e

devices="3"
pretrained_model_path="/home/data/zhuqi/pre-trained-models/bert-base-uncased"
model_des="task adaptive pretraining"
lr=2e-5
steps=1000
bz=32
accum=8
block=256
warmup=$((steps/100*0))
date=0520
dataset_name="m2m"
dataset="dataset_config_tapt_${dataset_name}.json"

for steps in 500
do
    for lr in 1e-4 5e-5 2e-5
    do
        output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/tapt/${dataset_name}_${date}_lr${lr}_steps${steps}_bz${bz}ac${accum}_block${block}_warmup${warmup}"
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
        --learning_rate=$lr \
        --weight_decay=0.01 \
        --adam_epsilon=1e-6 \
        --max_steps=$steps \
        --warmup_steps=$warmup \
        --logging_steps=100 \
        --save_total_limit=1 \
        --seed=42
    done
done
