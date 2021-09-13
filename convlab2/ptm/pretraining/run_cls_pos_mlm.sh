#!/usr/bin/env bash
devices="2"
pretrained_model_path="/home/guyuxian/pretrain-models/bert-base-uncased/"
output_dir="/home/guyuxian/ConvLab2-Pretraining/convlab2/ptm/pretraining/results/cls_mlm/cls_pos_mlm_12k_batch64_lr1e-4_block256_11.5_test"
dataset="dataset_config.json"
model_des="use cls and masked embeddings to predict the masked words. add positional embeddings to cls token"

python3 run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --cls_pos_mlm \
    --model_des=$model_des \
    --dataset_config=$dataset \
    --block_size=256 \
    --output_dir=$output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --max_steps=12000 \
    --warmup_steps=1200 \
    --logging_steps=1000 \
    --save_steps=1000 \
    --save_total_limit=1 \
    --seed=42 \
