#!/usr/bin/env bash
devices="7"
pretrained_model_path="/home/data/zhuqi/pre-trained-models/bert-base-uncased"
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/ssl/mlm_tf_allidf_12k_batch64_lr1e-4_block256_1026"
dataset="dataset_config.json"
model_des="mlm + tf_idf regression. mask 15% token, predict tf_idf and masked words. use head transform+linear for tf_idf regression"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --ssl \
    --ssl_tf_idf \
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
    --save_total_limit=3 \
    --seed=42
