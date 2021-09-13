export TRAIN_FILE=all_10turn_train.txt
export TEST_FILE=all_10turn_dev.txt

CUDA_VISIBLE_DEVICES=5,6 python3 run_dialog_mlm.py \
    --output_dir=output_all_10turn_7.6 \
    --model_name_or_path=/home/libing/Convlab2-Pretraining/convlab2/ptm/model/dialog_mlm/dialogbert_base_uncased \
    --model_type=bert \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --mlm \
    --line_by_line True \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --block_size=512 \
    --evaluate_during_training \
    --logging_dir=tensorboard_7.6 \
    --per_gpu_train_batch_size=6 \
    --per_gpu_eval_batch_size=6 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --adam_epsilon=1e-6 \
    --weight_decay=0.01 \
    --max_steps=450000 \
    --warmup_steps=45000 \
    --logging_steps=5000 \
    --save_steps=5000 \
    --save_total_limit=40
