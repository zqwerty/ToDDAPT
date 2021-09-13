#!/bin/bash

# fine-tune for response selection

#./scripts/run_rs.sh /home/data/zhuqi/pre-trained-models/dialogbert/augdial/uselabel_pos0_neg0_batch32_accum2_1221 save/dialogbert/augdial/uselabel_pos0_neg0_batch32_accum2_1221

model=dialogbert
bert_dir=$1
output_dir=$2
bsz=25

python finetune.py \
    --my_model=ResponseSelectionModel \
    --do_train \
    --dataset='["universal_act_sim_joint"]' \
    --task=nlg --task_name=rs \
    --example_type=turn \
    --model_type=${model} \
    --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir}/RS/SIM_JOINT/BSZ$bsz/ \
    --batch_size=${bsz} --eval_batch_size=100 \
    --max_seq_length=256 \
    --fix_rand_seed

python finetune.py \
    --my_model=ResponseSelectionModel \
    --do_train \
    --task=nlg \
    --task_name=rs \
    --example_type=turn \
    --model_type=${model} \
    --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir}/RS/MWOZ/BSZ$bsz/ \
    --batch_size=${bsz} --eval_batch_size=100 \
    --usr_token=[USR] --sys_token=[SYS] \
    --fix_rand_seed \
    --eval_by_step=1000 \
    --max_seq_length=256

python finetune.py \
    --my_model=ResponseSelectionModel \
    --do_train \
    --dataset='["universal_act_dstc2"]' \
    --task=nlg --task_name=rs \
    --example_type=turn \
    --model_type=${model} \
    --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir}/RS/DSTC2/BSZ$bsz/ \
    --batch_size=${bsz} --eval_batch_size=100 \
    --max_seq_length=256 \
    --fix_rand_seed
