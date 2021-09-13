#!/usr/bin/env bash
devices="6"
pretrained_model_path="/home/data/zhuqi/pre-trained-models/dialogbert/schemalinking_mlm/schemalinking_mlm_des2span_nograd_10k_lr5e-6_0914_mlm_wwm_120k_0831_bert"
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/schemalinking_mlm/test/schemalinking_mlm_des2span_nograd_10k_lr5e-6_0914_mlm_wwm_120k_0831_bert"
log_dir="/home/data/zhuqi/pre-trained-models/dialogbert/schemalinking_mlm/test/schemalinking_mlm_des2span_nograd_10k_lr5e-6_0914_mlm_wwm_120k_0831_bert"
dataset="dataset_config_test.json"
model_des="eval write embeddings"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --mlm \
    --mlm_wwm \
    --schema_linking \
    --span_mask_probability=0. \
    --neg_samples=0 \
    --model_des=$model_des \
    --dataset_config=$dataset \
    --block_size=512 \
    --output_dir=$output_dir \
    --overwrite_output_dir \
    --do_eval \
    --per_gpu_eval_batch_size=1 \
    --logging_dir=$log_dir \
    --seed=42
