#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python run.py \
        --train_data_path data_utils/dialoglue/hwu/train.csv \
        --val_data_path data_utils/dialoglue/hwu/val.csv \
        --test_data_path data_utils/dialoglue/hwu/test.csv \
        --token_vocab_path /home/data/zhuqi/pre-trained-models/bert-base-uncased/vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path /home/data/zhuqi/pre-trained-models/bert-base-uncased --task intent --do_lowercase --max_seq_length 50 --mlm_pre --mlm_during --dump_outputs \
        --output_dir output
