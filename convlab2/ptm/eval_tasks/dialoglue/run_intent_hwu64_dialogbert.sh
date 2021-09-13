
export MODEL_TYPE=dialogbert
export BERT_MODEL_DIR=/home/data/zhuqi/pre-trained-models/dialogbert/mlm/mlm_wwm_12k_batch64_lr1e-4_block256_warmup10_1020_bert
export BERT_VOCAB_PATH=$BERT_MODEL_DIR/vocab.txt

CUDA_VISIBLE_DEVICES=2 python3 run.py \
        --train_data_path data_utils/dialoglue/hwu/train.csv \
        --val_data_path data_utils/dialoglue/hwu/val.csv \
        --test_data_path data_utils/dialoglue/hwu/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 50 --dump_outputs \
        --model_type $MODEL_TYPE