#!/usr/bin/env bash
devices="4"

TASK="multiwoz21"

# change following two lines
model_name[0]="mlm_batch48_accum2_1217"
pretrained_weights[0]="/home/data/zhuqi/pre-trained-models/dialogbert/augdial/mlm_batch48_accum2_1217"

for ((i=0;i<${#pretrained_weights[@]};i++));
do
    echo "run on devices ${devices}"
    echo "${pretrained_weights[i]}"
    OUT_DIR="outputs/epoch50/${model_name[i]}"
    echo ${OUT_DIR}
    mkdir -p ${OUT_DIR}

    DATA_DIR="../data_utils/dialoglue/multiwoz/MULTIWOZ2.1"

    for step in train test; do
        args_add=""
        if [ "$step" = "train" ]; then
        args_add="--do_train --predict_type=dummy"
        elif [ "$step" = "dev" ] || [ "$step" = "test" ]; then
        args_add="--do_eval --predict_type=${step}"
        fi

        CUDA_VISIBLE_DEVICES=${devices} python3 run_dst.py \
            --task_name=${TASK} \
            --data_dir=${DATA_DIR} \
            --dataset_config=dataset_config/${TASK}.json \
            --model_type="dialogbert" \
            --model_name_or_path=${pretrained_weights[i]} \
            --do_lower_case \
            --learning_rate=1e-4 \
            --num_train_epochs=50 \
            --max_seq_length=180 \
            --per_gpu_train_batch_size=48 \
            --per_gpu_eval_batch_size=1 \
            --output_dir=${OUT_DIR} \
            --save_epochs=10 \
            --logging_steps=10 \
            --warmup_proportion=0.1 \
            --adam_epsilon=1e-6 \
            --label_value_repetitions \
            --swap_utterances \
            --append_history \
            --use_history_labels \
            --delexicalize_sys_utts \
            --class_aux_feats_inform \
            --class_aux_feats_ds \
            --seed 42 \
            ${args_add} \
            2>&1 | tee ${OUT_DIR}/${step}.log

        if [ "$step" = "dev" ] || [ "$step" = "test" ]; then
            python3 metric_bert_dst.py \
                ${TASK} \
            dataset_config/${TASK}.json \
                "${OUT_DIR}/pred_res.${step}*json" \
                2>&1 | tee ${OUT_DIR}/eval_pred_${step}.log
        fi
    done
done
