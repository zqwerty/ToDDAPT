#!/bin/bash

# fine-tune for intent recognition

#./scripts/run_intent.sh /home/data/zhuqi/pre-trained-models/dialogbert/augdial/uselabel_pos0_neg0_batch32_accum2_1221 save/dialogbert/augdial/uselabel_pos0_neg0_batch32_accum2_1221 pooled

model=dialogbert
bert_dir=$1
output_dir=$2
extra=$3
bsz=32

python finetune.py \
    --my_model IntentRecognitionModel \
    --datasets oos_intent \
    --task_name intent \
    --earlystop acc \
    --output_dir ${output_dir}/Intent/OOS \
    --do_train \
    --task nlu \
    --example_type turn \
    --model_type ${model} \
    --model_name_or_path ${bert_dir} \
    --batch_size ${bsz} \
    --usr_token [USR] --sys_token [SYS] \
    --eval_by_step 500 --warmup_steps 250 \
    ${extra}
