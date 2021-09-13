#!/usr/bin/env bash
devices="4"
pretrained_model_path="/home/data/zhuqi/pre-trained-models/bert-base-uncased"
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/mlm/last_turn_mlm_12k_batch64_lr1e-4_block256_warmup10_1019_bert"
dataset="dataset_config.json"
model_des="last turn mlm. batch=32*2, block=256, cut_turn=29, step 12k, lr 1e-4, warm up 10%"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --last_turn_mlm \
    --mlm_wwm \
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
    --logging_steps=500 \
    --save_steps=500 \
    --save_total_limit=5 \
    --seed=42
