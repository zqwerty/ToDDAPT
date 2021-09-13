#!/usr/bin/env bash
devices="5"
pretrained_model_path="/home/data/zhuqi/pre-trained-models/bert-base-uncased"
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/augdial/schemalinking_batch48_accum2_1217"
dataset="dataset_config_test.json"
model_des="cls_similarity. sgd dataset"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --augdial \
    --pos_aug_num=0 \
    --neg_aug_num=0 \
    --use_label \
    --model_des=$model_des \
    --dataset_config=$dataset \
    --block_size=256 \
    --output_dir=$output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size=48 \
    --per_gpu_eval_batch_size=48 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --max_steps=3000 \
    --warmup_steps=300 \
    --logging_steps=100 \
    --save_steps=500 \
    --save_total_limit=5 \
    --seed=42
