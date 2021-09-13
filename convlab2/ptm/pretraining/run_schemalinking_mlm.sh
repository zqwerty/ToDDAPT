#!/usr/bin/env bash
devices="0"
pretrained_model_path="/home/data/zhuqi/pre-trained-models/bert-base-uncased"
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/schemalinking_mlm/cls_utt_slot_mlm_sgd_3k_batch64_lr1e-4_block256_1031"
dataset="dataset_config_test.json"
model_des="schema linking: predict slots of a sentence and a dialog([USR],[SYS],[CLS]), mask out all value. only use schema dataset"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --mlm \
    --schema_linking \
    --span_mask_probability=0 \
    --neg_samples=0 \
    --model_des=$model_des \
    --dataset_config=$dataset \
    --block_size=256 \
    --output_dir=$output_dir \
    --overwrite_output_dir \
    --do_train \
    --evaluate_during_training \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --max_steps=3000 \
    --warmup_steps=300 \
    --logging_steps=500 \
    --save_steps=500 \
    --save_total_limit=1 \
    --seed=42
