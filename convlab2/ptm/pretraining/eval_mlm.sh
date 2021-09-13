#!/usr/bin/env bash
set -e
devices="1"
MODEL_DIR="/home/data/zhuqi/pre-trained-models/dialogbert"
dataset="dataset_config_eval_mlm.json"
model_des="eval mlm acc"
bz=64

pretrained_model_path[0]="$MODEL_DIR/dapt/1.0data_0506_lr1e-4_steps40000_bz32ac8_block256_warmup0/checkpoint-39000"
pretrained_model_path[1]="$MODEL_DIR/dapt/0.25data_0517_lr5e-5_steps10000_bz32ac8_block256_warmup0"
pretrained_model_path[2]="$MODEL_DIR/dapt/0.05data_0518_lr2e-5_steps5000_bz32ac8_block256_warmup0/checkpoint-4000"
pretrained_model_path[3]="$MODEL_DIR/dapt/0.01data_0518_lr2e-5_steps1000_bz32ac8_block256_warmup0"
pretrained_model_path[4]="$MODEL_DIR/tapt/hwu_0520_lr5e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-700"
pretrained_model_path[5]="$MODEL_DIR/tapt/banking_0520_lr5e-5_steps1000_bz32ac8_block256_warmup0"
pretrained_model_path[6]="$MODEL_DIR/tapt/oos_0520_lr1e-4_steps1000_bz32ac8_block256_warmup0"
pretrained_model_path[7]="$MODEL_DIR/tapt/restaurant8k_0520_lr2e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-900"
pretrained_model_path[8]="$MODEL_DIR/tapt/top_0520_lr5e-5_steps2000_bz32ac8_block256_warmup0/checkpoint-1800"
pretrained_model_path[9]="$MODEL_DIR/tapt/multiwoz_0519_lr5e-5_steps10000_bz32ac8_block256_warmup0"
pretrained_model_path[10]="$MODEL_DIR/tapt/m2m_0520_lr5e-5_steps500_bz32ac8_block256_warmup0/checkpoint-400"
pretrained_model_path[11]="$MODEL_DIR/tapt/dstc2_0520_lr1e-4_steps1000_bz32ac8_block256_warmup0/checkpoint-600"
pretrained_model_path[12]="$MODEL_DIR/tapt/dapt_hwu_0524_lr5e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-300"
pretrained_model_path[13]="$MODEL_DIR/tapt/dapt_banking_0524_lr5e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-600"
pretrained_model_path[14]="$MODEL_DIR/tapt/dapt_oos_0524_lr1e-4_steps1000_bz32ac8_block256_warmup0/checkpoint-1000"
pretrained_model_path[15]="$MODEL_DIR/tapt/dapt_restaurant8k_0524_lr2e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-300"
pretrained_model_path[16]="$MODEL_DIR/tapt/dapt_top_0524_lr5e-5_steps2000_bz32ac8_block256_warmup0/checkpoint-1900"
pretrained_model_path[17]="$MODEL_DIR/tapt/dapt_multiwoz_0524_lr5e-5_steps10000_bz32ac8_block256_warmup0/checkpoint-4100"
pretrained_model_path[18]="$MODEL_DIR/tapt/dapt_m2m_0524_lr5e-5_steps500_bz32ac8_block256_warmup0/checkpoint-200"
pretrained_model_path[19]="$MODEL_DIR/tapt/dapt_dstc2_0524_lr1e-4_steps1000_bz32ac8_block256_warmup0/checkpoint-400"

echo "run on devices $devices"
for ((i=0;i<${#pretrained_model_path[@]};i++));
do
    python run_pretrain.py \
        --cuda_visible_devices=$devices \
        --model_name_or_path=${pretrained_model_path[i]} \
        --dapt \
        --model_des=$model_des \
        --dataset_config=$dataset \
        --block_size=512 \
        --output_dir=$MODEL_DIR \
        --overwrite_output_dir \
        --do_eval \
        --per_gpu_eval_batch_size=$bz \
        --logging_dir=${pretrained_model_path[i]} \
        --seed=42
done