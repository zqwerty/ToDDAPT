TRANSFORMER=dialog-bert # model type. must be "bert" or "dialog-bert"
TRANSFORMER_PATH=${HOME}/pretrain-models/dialog-bert/mlm_wwm_120k_0831_bert # path to pre-trained model
OUTPUT_DIR=test25/ # the directory to save a model
RATIO=1 # the ratio we split the train set for training
EVAL_RATIO=0.02 # the ratio we split the test/valid set
DATASET=multiwoz25 # dataset name, "multiwoz21" or "multiwoz25"
MODEL_NAME=0831_dial_0919_joint # the name we give the model
EVAL_CONFIG=eval_configs/full.json # the metrics we need for evaluation
GRAD_ACC=1 # gradient accumulation step
TRN_BS=8 # train batch size

CMD="train.py"
CMD+=" --transformer ${TRANSFORMER}"
CMD+=" --transformer_path ${TRANSFORMER_PATH}"
CMD+=" --output_dir ${OUTPUT_DIR}"
# CMD+=" --do_train"
CMD+=" --do_eval"
CMD+=" --save_all"
CMD+=" --model_name ${MODEL_NAME}"
CMD+=" --ratio ${RATIO}"
CMD+=" --dataset ${DATASET}"
CMD+=" --eval_config ${EVAL_CONFIG}"
CMD+=" --grad_acc ${GRAD_ACC}"
CMD+=" --per_gpu_train_batch_size ${TRN_BS}"
CMD+=" --eval_ratio ${EVAL_RATIO}"

CMD="python3 ${CMD}"

echo ${CMD}

${CMD}
