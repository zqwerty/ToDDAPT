export MODEL_TYPE=bert
#export BERT_MODEL_DIR=/home/data/zhuqi/pre-trained-models/dialogbert/mlm/mlm_wwm_120k_0831_bert
export BERT_MODEL_DIR=/home/libing/pretrained_models/bert-base-uncased
export BERT_VOCAB_PATH=$BERT_MODEL_DIR/vocab.txt

CUDA_VISIBLE_DEVICES=3 python3 run.py \
        --train_data_path data_utils/dialoglue/restaurant8k/train.json \
        --val_data_path data_utils/dialoglue/restaurant8k/val.json \
        --test_data_path data_utils/dialoglue/restaurant8k/test.json \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task slot --do_lowercase --max_seq_length 50 --dump_outputs \
        --model_type $MODEL_TYPE