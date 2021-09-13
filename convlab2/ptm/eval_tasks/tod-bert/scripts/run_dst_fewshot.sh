BERT_DIR=${HOME}/pretrain-models/dialog-bert/output_multiturn_fulldialog_7.19
OUTPUT_DIR=results/DST/MWOZ/bert-multiturn_fulldialog
DATA_PATH=data/
MODEL_TYPE=dialogbert

for ratio in 0.01 0.05 
do
python3 train.py \
    --my_model=DialogBertForDST \
    --model_type=${MODEL_TYPE} \
    --dataset='["multiwoz"]' \
    --task_name="dst" \
    --earlystop="joint_acc" \
    --output_dir=${OUTPUT_DIR}-${ratio} \
    --do_train \
    --task=dst \
    --example_type=turn \
    --model_name_or_path=${BERT_DIR} \
    --batch_size=8 \
    --eval_batch_size=8 \
    --usr_token=[USR] \
    --sys_token=[SYS] \
    --eval_by_step=200 \
    --train_data_ratio=${ratio} \
    --data_path=${DATA_PATH}
done
for ratio in 0.1 0.25
do
python3 train.py \
    --my_model=DialogBertForDST \
    --model_type=${MODEL_TYPE} \
    --dataset='["multiwoz"]' \
    --task_name="dst" \
    --earlystop="joint_acc" \
    --output_dir=${OUTPUT_DIR}-${ratio} \
    --do_train \
    --task=dst \
    --example_type=turn \
    --model_name_or_path=${BERT_DIR} \
    --batch_size=8 \
    --eval_batch_size=8 \
    --usr_token=[USR] \
    --sys_token=[SYS] \
    --eval_by_step=500 \
    --train_data_ratio=${ratio} \
    --data_path=${DATA_PATH}
done