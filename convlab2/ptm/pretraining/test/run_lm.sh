#!/usr/bin/env bash
export TRAIN_FILE=wikitext-2-raw/wiki.train.raw
export TEST_FILE=wikitext-2-raw/wiki.test.raw

#CUDA_VISIBLE_DEVICES="7" python run_language_modeling.py \
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node=4 --master_port 20209 run_language_modeling.py \
    --output_dir=output \
    --model_type=bert \
    --model_name_or_path=/home/data/zhuqi/pre-trained-models/bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --block_size=512 \
    --max_steps=500 \
    --overwrite_output_dir