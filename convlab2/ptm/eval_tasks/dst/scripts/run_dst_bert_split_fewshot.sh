TRANSFORMER=bert
TRANSFORMER_PATH=${HOME}/pretrain-models/bert-base-uncased/
OUTPUT_DIR=test25/
# RATIO=1
EVAL_RATIO=1
DATASET=multiwoz25
EVAL_CONFIG=eval_configs/full.json
GRAD_ACC=1
TRN_BS=8 # train batch size
LR=2e-5 # learning
WU=1
WV=5
WS=5

for ratio in 0.05 0.25
do
    MODEL_NAME=origin_bert_1018_sche_balance_split_few_all_seed # the name we give the model
    for seed in 23333 233333 2333333
    do
        CMD="train.py"
        CMD+=" --transformer ${TRANSFORMER}"
        CMD+=" --transformer_path ${TRANSFORMER_PATH}"
        CMD+=" --output_dir ${OUTPUT_DIR}"
        CMD+=" --do_train"
        CMD+=" --do_eval"
        # CMD+=" --save_all"
        CMD+=" --model_name ${MODEL_NAME}"
        CMD+=" --ratio ${ratio}"
        CMD+=" --dataset ${DATASET}"
        CMD+=" --eval_config ${EVAL_CONFIG}"
        CMD+=" --grad_acc ${GRAD_ACC}"
        CMD+=" --split"
        CMD+=" --per_gpu_train_batch_size ${TRN_BS}"
        CMD+=" --eval_ratio ${EVAL_RATIO}"
        CMD+=" --balance"
        CMD+=" --sche"
        CMD+=" --lr ${LR}"
        CMD+=" --w_update ${WU}"
        CMD+=" --w_value ${WV}"
        CMD+=" --w_span ${WS}"
        CMD+=" --seed ${seed}"
        CMD="python3 ${CMD}"

        echo ${CMD}
        ${CMD}
    done
done
 