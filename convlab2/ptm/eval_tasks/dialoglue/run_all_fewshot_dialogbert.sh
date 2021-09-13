set -e

MODEL_TYPE=dialogbert
BERT_MODEL_DIR=/home/data/zhuqi/pre-trained-models/dialogbert/augdial/ori_mlm_tokenslot_sid_clscl_clipaug_simclr_pos2_t0.1_1214
BERT_VOCAB_PATH=$BERT_MODEL_DIR/vocab.txt
OUTPUT_DIR_PREFIX=/home/data/zhuqi/pre-trained-models/dialogbert/eval_dialoglue
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/hwu/train_10.csv \
        --val_data_path data_utils/dialoglue/hwu/val.csv \
        --test_data_path data_utils/dialoglue/hwu/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 50 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 5

CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/clinc/train_10.csv \
        --val_data_path data_utils/dialoglue/clinc/val.csv \
        --test_data_path data_utils/dialoglue/clinc/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 50 --dump_outputs \
        --model_type=$MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 5

CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train_10.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 5

CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/restaurant8k/train_10.json \
        --val_data_path data_utils/dialoglue/restaurant8k/val.json \
        --test_data_path data_utils/dialoglue/restaurant8k/test.json \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task slot --do_lowercase  --max_seq_length 50 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 5

CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/dstc8_sgd/train_10.json \
        --val_data_path data_utils/dialoglue/dstc8_sgd/val.json \
        --test_data_path data_utils/dialoglue/dstc8_sgd/test.json \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task slot --do_lowercase --max_seq_length 50 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 5

CUDA_VISIBLE_DEVICES=$GPU python run.py \
        --train_data_path data_utils/dialoglue/top/train_10.txt \
        --val_data_path data_utils/dialoglue/top/eval.txt \
        --test_data_path data_utils/dialoglue/top/test.txt \
        --token_vocab_path $BERT_VOCAB_PATH \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task top --do_lowercase --max_seq_length 50 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 5