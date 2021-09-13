#!/usr/bin/env bash
devices="0"
seeds=(42)
train_ratios=(1.0)

OUTPUT_DIR_PREFIX="/home/data/zhuqi/pre-trained-models/dialogbert/eval_nlu/multiwoz23/all_dialogbert"
MODEL_DIR_PREFIX="/home/data/zhuqi/pre-trained-models/dialogbert"

#pretrained_weights[0]="${MODEL_DIR_PREFIX}/dapt/1.0data_0506_lr1e-4_steps40000_bz32ac8_block256_warmup0/checkpoint-39000/"
#output_dir[0]="${OUTPUT_DIR_PREFIX}/1.0data_0506_lr1e-4_steps40000_bz32ac8_block256_warmup0"

pretrained_weights[0]="${MODEL_DIR_PREFIX}/dapt/0.25data_0517_lr5e-5_steps10000_bz32ac8_block256_warmup0/"
output_dir[0]="${OUTPUT_DIR_PREFIX}/0.25data_0517_lr5e-5_steps10000_bz32ac8_block256_warmup0"
#
#pretrained_weights[0]="${MODEL_DIR_PREFIX}/dapt/0.05data_0518_lr2e-5_steps5000_bz32ac8_block256_warmup0/checkpoint-4000/"
#output_dir[0]="${OUTPUT_DIR_PREFIX}/0.05data_0518_lr2e-5_steps5000_bz32ac8_block256_warmup0"
#
#pretrained_weights[0]="${MODEL_DIR_PREFIX}/dapt/0.01data_0518_lr2e-5_steps1000_bz32ac8_block256_warmup0/"
#output_dir[0]="${OUTPUT_DIR_PREFIX}/0.01data_0518_lr2e-5_steps1000_bz32ac8_block256_warmup0"
#
#pretrained_weights[0]="${MODEL_DIR_PREFIX}/tapt/multiwoz_0519_lr5e-5_steps10000_bz32ac8_block256_warmup0/"
#output_dir[0]="${OUTPUT_DIR_PREFIX}/multiwoz_0519_lr5e-5_steps10000_bz32ac8_block256_warmup0"
#
#pretrained_weights[0]="${MODEL_DIR_PREFIX}/tapt/dapt_multiwoz_0524_lr5e-5_steps10000_bz32ac8_block256_warmup0/checkpoint-4100/"
#output_dir[0]="${OUTPUT_DIR_PREFIX}/dapt_multiwoz_0524_lr5e-5_steps10000_bz32ac8_block256_warmup0"

echo "run on devices $devices"
for ((i=0;i<${#pretrained_weights[@]};i++));
do
    echo "${pretrained_weights[i]}"
    CUDA_VISIBLE_DEVICES=$devices python train.py \
        --seeds ${seeds[*]} \
        --train_ratios ${train_ratios[*]} \
        --do_train \
        --do_eval \
        --pretrained_weights ${pretrained_weights[i]} \
        --basemodel dialogbert \
        --data_dir "multiwoz/multiwoz23/all_data" \
        --output_dir ${output_dir[i]}
done