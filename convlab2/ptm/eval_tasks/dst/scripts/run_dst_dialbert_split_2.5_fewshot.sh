TRANSFORMER=dialog-bert # model type. must be "bert" or "dialog-bert"
TRANSFORMER_PATH=/home/guyuxian/pretrain-models/dialog-bert/sop_with_multiwoz_test_1004_bert/checkpoint-100000/ # path to pre-trained model
OUTPUT_DIR=test25/ # the directory to save a model
# RATIO=1 # the ratio we split the train set for training
EVAL_RATIO=1 # the ratio we split the test/valid set
DATASET=multiwoz25 # dataset name, "multiwoz21" or "multiwoz25"
EVAL_CONFIG=eval_configs/full.json # the metrics we need for evaluation
GRAD_ACC=1 # gradient accumulation step
TRN_BS=8 # train batch size
LR=2e-5 # learning
WU=1
WV=5
WS=5

for ratio in 0.05 0.25
do
    MODEL_NAME=sop_with_multiwoz_test_1004_bert_ckpt100000_1014_sche_balance_split_few_all_seed # the name we give the model
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
