set -e

MODEL_TYPE=dialogbert
OUTPUT_DIR_PREFIX=/home/data/zhuqi/pre-trained-models/dialogbert/eval_dialoglue
GPU=1

BERT_MODEL_DIR=/home/data/zhuqi/pre-trained-models/dialogbert/dapt/1.0data_0506_lr1e-4_steps40000_bz32ac8_block256_warmup0/checkpoint-39000
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train_10.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 3

BERT_MODEL_DIR=/home/data/zhuqi/pre-trained-models/dialogbert/dapt/0.25data_0517_lr5e-5_steps10000_bz32ac8_block256_warmup0/checkpoint-10000
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train_10.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 3

BERT_MODEL_DIR=/home/data/zhuqi/pre-trained-models/dialogbert/dapt/0.05data_0518_lr2e-5_steps5000_bz32ac8_block256_warmup0/checkpoint-4000
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train_10.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 3

BERT_MODEL_DIR=/home/data/zhuqi/pre-trained-models/dialogbert/dapt/0.01data_0518_lr2e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-1000
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train_10.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 3

BERT_MODEL_DIR=/home/data/zhuqi/pre-trained-models/dialogbert/tapt/banking_0520_lr5e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-1000
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train_10.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 3

BERT_MODEL_DIR=/home/data/zhuqi/pre-trained-models/dialogbert/tapt/dapt_banking_0524_lr5e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-600
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX
CUDA_VISIBLE_DEVICES=$GPU python3 run.py \
        --train_data_path data_utils/dialoglue/banking/train_10.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path $BERT_MODEL_DIR/vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path $BERT_MODEL_DIR --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
        --model_type $MODEL_TYPE \
        --output_dir_prefix $OUTPUT_DIR_PREFIX \
        --repeat 3