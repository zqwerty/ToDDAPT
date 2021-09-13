#!/usr/bin/env bash
devices="1"
pretrained_model_path="/home/data/zhuqi/pre-trained-models/bert-base-uncased"
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/bio_mlm/mlm_masked_bio_fulldial_12k_batch64_lr1e-4_block512_1117"
dataset="dataset_config.json"
model_des="mlm + masked bio. use MLP classifier for bio tagging; mask 15% token, predict BIO."

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --mlm \
    --bio \
    --model_des=$model_des \
    --dataset_config=$dataset \
    --block_size=512 \
    --output_dir=$output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --max_steps=12000 \
    --warmup_steps=1200 \
    --logging_steps=1000 \
    --save_steps=1000 \
    --save_total_limit=1 \
    --seed=42
