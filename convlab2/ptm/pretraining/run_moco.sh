#!/usr/bin/env bash
devices="2"
pretrained_model_path="/home/guyuxian/pretrain-models/dialog-bert/mlm_sgd_3k_batch64_lr1e-4_block256_1031"
output_dir="results/moco/keep_60k-11.30-wm40k-lr2e-4-m0.99-bs32-t0.07-mlm_init-linear"
dataset="dataset_config.json"
model_des="moco"

python3 run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --moco \
    --model_des=$model_des \
    --dataset_config=$dataset \
    --block_size=256 \
    --output_dir=$output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=32 \
    --gradient_accumulation_steps=8 \
    --learning_rate=2e-4 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --max_steps=80000 \
    --warmup_steps=40000 \
    --logging_steps=100 \
    --save_steps=10000 \
    --save_total_limit=100000 \
    --seed=42 \
    --moco_K 4096 \
    --moco_m 0.99 \
    --moco_T 0.07 \
    --moco_keep_value \
    --sche_type linear \
