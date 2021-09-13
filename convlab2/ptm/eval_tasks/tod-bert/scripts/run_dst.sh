BERT_DIR=/home/data/zhuqi/pre-trained-models/dialogbert/mlm/mlm_wwm_120k_0831_bert
OUTPUT_DIR=results/DST/MWOZ/mlm_wwm_120k_0831_bert/
DATA_PATH=data/
EVAL_BY_STEP=4000
TRAIN_BATCH_SIZE=6
EVAL_BATCH_SIZE=6
MODEL_TYPE=dialogbert

python3 finetune.py \
    --my_model=DialogBertForDST \
    --model_type=${MODEL_TYPE} \
    --dataset='["multiwoz"]' \
    --task_name="dst" \
    --earlystop="joint_acc" \
    --output_dir=${OUTPUT_DIR} \
    --do_train \
    --task=dst \
    --example_type=turn \
    --model_name_or_path=${BERT_DIR} \
    --batch_size=${TRAIN_BATCH_SIZE} \
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    --usr_token=[USR] \
    --sys_token=[SYS] \
    --eval_by_step=${EVAL_BY_STEP} \
    --data_path=${DATA_PATH}