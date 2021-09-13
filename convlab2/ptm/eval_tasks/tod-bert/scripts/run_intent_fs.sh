#!/bin/bash

model=dialogbert
bert_dir=$1
output_dir=$2
bsz=32

for ratio in 0.01 0.1; do
    python finetune.py \
        --my_model=IntentRecognitionModel \
        --datasets="oos_intent" \
        --task_name="intent" \
        --earlystop="acc" \
        --output_dir=${output_dir}/Intent/OOS-Ratio/R${ratio} \
        --do_train \
        --task=nlu \
        --example_type=turn \
        --model_type=${model} \
        --model_name_or_path=${bert_dir} \
        --batch_size=16 \
        --usr_token=[USR] --sys_token=[SYS] \
        --epoch=500 --eval_by_step=100 --warmup_steps=100 \
        --train_data_ratio=${ratio} \
        --nb_runs 3
done
