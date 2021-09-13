#!/usr/bin/env bash
devices="5"
pretrained_model_path="/home/data/zhuqi/pre-trained-models/bert-base-uncased"

# all data mlm, 1 epoch
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/augdial/mlm_alldata_12k_batch32_accum2_lr1e-4_block256_1218"
dataset="dataset_config.json"
model_des="mlm on all 10 dataset (not use multiwoz), 1 epoch"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --mlm \
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
    --save_total_limit=1 \
    --seed=42

# all data mlm, 10 epoch
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/augdial/mlm_alldata_120k_batch32_accum2_lr1e-4_block256_1218"
dataset="dataset_config.json"
model_des="mlm on all 10 dataset (not use multiwoz), 10 epoch"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --mlm \
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
    --max_steps=120000 \
    --warmup_steps=12000 \
    --logging_steps=500 \
    --save_steps=500 \
    --save_total_limit=1 \
    --seed=42

# sgd data mlm, 1 epoch
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/augdial/mlm_sgd_3k_batch32_accum2_lr1e-4_block256_1218"
dataset="dataset_config_test.json"
model_des="mlm on sgd dataset, 1 epoch"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --mlm \
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
    --max_steps=3000 \
    --warmup_steps=300 \
    --logging_steps=500 \
    --save_steps=500 \
    --save_total_limit=1 \
    --seed=42

# sgd data mlm, 10 epoch
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/augdial/mlm_sgd_30k_batch32_accum2_lr1e-4_block256_1218"
dataset="dataset_config_test.json"
model_des="mlm on sgd dataset, 10 epoch"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --mlm \
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
    --max_steps=30000 \
    --warmup_steps=3000 \
    --logging_steps=500 \
    --save_steps=500 \
    --save_total_limit=1 \
    --seed=42

# sgd data schemalinking, 1 epoch
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/augdial/schemalinking_sgd_3k_batch32_accum2_lr1e-4_block256_1218"
dataset="dataset_config_test.json"
model_des="mlm, bio-slot, slot, intent, domain prediction. sgd dataset. 1 epoch"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --schema_linking \
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
    --max_steps=3000 \
    --warmup_steps=300 \
    --logging_steps=500 \
    --save_steps=500 \
    --save_total_limit=1 \
    --seed=42

# sgd data schemalinking, 10 epoch
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/augdial/schemalinking_sgd_30k_batch32_accum2_lr1e-4_block256_1218"
dataset="dataset_config_test.json"
model_des="mlm, bio-slot, slot, intent, domain prediction. sgd dataset. 10 epoch"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --schema_linking \
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
    --max_steps=30000 \
    --warmup_steps=3000 \
    --logging_steps=500 \
    --save_steps=500 \
    --save_total_limit=1 \
    --seed=42
