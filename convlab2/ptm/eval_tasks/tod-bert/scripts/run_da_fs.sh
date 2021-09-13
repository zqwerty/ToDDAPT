#!/bin/bash

# fine-tune for dialog act prediction

#./scripts/run_da.sh /home/data/zhuqi/pre-trained-models/dialogbert/augdial/uselabel_pos0_neg0_batch32_accum2_1221 save/dialogbert/augdial/uselabel_pos0_neg0_batch32_accum2_1221

model=dialogbert
bert_dir=$1
output_dir=$2
extra=$3
bsz=8

# DA
for ratio in 0.01; do
    python finetune.py \
        --my_model=DialogActPredictionModel \
        --do_train --datasets="multiwoz" \
        --task=dm --task_name=sysact --example_type=turn \
        --model_type=${model} \
        --model_name_or_path=${bert_dir} \
        --output_dir=${output_dir}/DA/MWOZ-Ratio/R${ratio} \
        --batch_size=8 \
        --eval_batch_size=4 \
        --learning_rate=5e-5 \
        --eval_by_step=200 \
        --usr_token=[USR] --sys_token=[SYS] \
        --earlystop=f1_weighted \
        --train_data_ratio=${ratio} \
        --nb_runs 3

    python finetune.py \
        --my_model=DialogActPredictionModel \
        --do_train \
        --datasets="universal_act_dstc2" \
        --task=dm --task_name=sysact --example_type=turn \
        --model_type=${model} --model_name_or_path=${bert_dir} \
        --output_dir=${output_dir}/DA/DSTC2-Ratio/R${ratio} \
        --batch_size=8 \
        --eval_batch_size=4 \
        --learning_rate=5e-5 \
        --eval_by_step=100 \
        --usr_token=[USR] --sys_token=[SYS] \
        --earlystop=f1_weighted \
        --train_data_ratio=${ratio} \
        --nb_runs 3

    python finetune.py \
        --my_model=DialogActPredictionModel \
        --do_train \
        --datasets="universal_act_sim_joint" \
        --task=dm --task_name=sysact --example_type=turn \
        --model_type=${model} --model_name_or_path=${bert_dir} \
        --output_dir=${output_dir}/DA/SIM_JOINT-Ratio/R${ratio} \
        --batch_size=8 \
        --eval_batch_size=4 \
        --learning_rate=5e-5 \
        --eval_by_step=100 \
        --usr_token=[USR] --sys_token=[SYS] \
        --earlystop=f1_weighted \
        --train_data_ratio=${ratio} \
        --nb_runs 3
done

for ratio in 0.1; do
    python finetune.py \
        --my_model=DialogActPredictionModel \
        --do_train --datasets="multiwoz" \
        --task=dm --task_name=sysact --example_type=turn \
        --model_type=${model} \
        --model_name_or_path=${bert_dir} \
        --output_dir=${output_dir}/DA/MWOZ-Ratio/R${ratio} \
        --batch_size=8 \
        --eval_batch_size=4 \
        --learning_rate=5e-5 \
        --eval_by_step=500 \
        --usr_token=[USR] --sys_token=[SYS] \
        --earlystop=f1_weighted \
        --train_data_ratio=${ratio} \
        --nb_runs 3

    python finetune.py \
        --my_model=DialogActPredictionModel \
        --do_train \
        --datasets="universal_act_dstc2" \
        --task=dm --task_name=sysact --example_type=turn \
        --model_type=${model} --model_name_or_path=${bert_dir} \
        --output_dir=${output_dir}/DA/DSTC2-Ratio/R${ratio} \
        --batch_size=8 \
        --eval_batch_size=4 \
        --learning_rate=5e-5 \
        --eval_by_step=500 \
        --usr_token=[USR] --sys_token=[SYS] \
        --earlystop=f1_weighted \
        --train_data_ratio=${ratio} \
        --nb_runs 3

    python finetune.py \
        --my_model=DialogActPredictionModel \
        --do_train \
        --datasets="universal_act_sim_joint" \
        --task=dm --task_name=sysact --example_type=turn \
        --model_type=${model} --model_name_or_path=${bert_dir} \
        --output_dir=${output_dir}/DA/SIM_JOINT-Ratio/R${ratio} \
        --batch_size=8 \
        --eval_batch_size=4 \
        --learning_rate=5e-5 \
        --eval_by_step=500 \
        --usr_token=[USR] --sys_token=[SYS] \
        --earlystop=f1_weighted \
        --train_data_ratio=${ratio} \
        --nb_runs 3
done