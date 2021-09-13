#!/usr/bin/env bash
devices="2"
pretrained_model_path="/home/data/zhuqi/pre-trained-models/bert-base-uncased"
#pretrained_model_path="/home/data/zhuqi/pre-trained-models/dialogbert/schemalinking/mlm_bioslot_sid_bertinit_sgd_3k_batch64_lr1e-4_block256_1205"
output_dir="/home/data/zhuqi/pre-trained-models/dialogbert/augdial/ori_mlm_tokenslot_sid_clscl_noclip_notkeepv_simclr_pos2_t0.1_block128_batch48_accum2_1216"
dataset="dataset_config_test.json"
model_des="cls_similarity. sgd dataset"

python run_pretrain.py \
    --cuda_visible_devices=$devices \
    --model_name_or_path=$pretrained_model_path \
    --augdial \
    --pos_aug_num=2 \
    --neg_aug_num=0 \
    --use_label \
    --cls_contrastive \
    --temperature=0.1 \
    --cls_contrastive_type=2 \
    --model_des=$model_des \
    --dataset_config=$dataset \
    --block_size=128 \
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
