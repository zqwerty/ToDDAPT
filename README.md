## ToDDAPT

Codes for paper: "When does Further Pre-training MLM Help? An Empirical Study on Task-Oriented Dialog Pre-training".

This project is developed based on [ConvLab-2 (commit 786d08)](https://github.com/thu-coai/ConvLab-2/tree/786d089a21f05bf9fbc64366e1eae4af332ef702).

Run `pip install -e .` and `pip install -r dialogbert_requirements.txt`.

Note that you should replace pre-trained models' paths and output paths in the codes.

### Dataset preprocess

In `data_ptm`, run `preprocess.py` for each dataset.

In `convlab2/ptm/pretraining/prepare_data`, run `preprocess_data.py`.

### DAPT

In `convlab2/ptm/pretraining/run_dapt.sh`, set hyperparameters and run.

Best checkpoint for each DAPT data ratio:

1. dialogbert/dapt/1.0data_0506_lr1e-4_steps40000_bz32ac8_block256_warmup0/checkpoint-39000/
2. dialogbert/dapt/0.25data_0517_lr5e-5_steps10000_bz32ac8_block256_warmup0/
3. dialogbert/dapt/0.05data_0518_lr2e-5_steps5000_bz32ac8_block256_warmup0/checkpoint-4000/
4. dialogbert/dapt/0.01data_0518_lr2e-5_steps1000_bz32ac8_block256_warmup0/

### TAPT

In `convlab2/ptm/pretraining/run_tapt.sh`, set hyperparameters and run.

Best checkpoint for each TAPT data ratio:

1. dialogbert/tapt/multiwoz_0519_lr5e-5_steps10000_bz32ac8_block256_warmup0/
2. dialogbert/tapt/hwu_0520_lr5e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-700
3. dialogbert/tapt/banking_0520_lr5e-5_steps1000_bz32ac8_block256_warmup0
4. dialogbert/tapt/oos_0520_lr1e-4_steps1000_bz32ac8_block256_warmup0
5. dialogbert/tapt/restaurant8k_0520_lr2e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-900
6. dialogbert/tapt/top_0520_lr5e-5_steps2000_bz32ac8_block256_warmup0/checkpoint-1800
7. dialogbert/tapt/m2m_0520_lr5e-5_steps500_bz32ac8_block256_warmup0/checkpoint-400

### DAPT+TAPT

In `convlab2/ptm/pretraining/run_dapt_tapt.sh`, set hyperparameters and run.

Best checkpoint for each DAPT+TAPT data ratio:

1. dialogbert/tapt/dapt_hwu_0524_lr5e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-300/
2. dialogbert/tapt/dapt_banking_0524_lr5e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-600/
3. dialogbert/tapt/dapt_restaurant8k_0524_lr2e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-300
4. dialogbert/tapt/dapt_top_0524_lr5e-5_steps2000_bz32ac8_block256_warmup0/checkpoint-1900/
5. dialogbert/tapt/dapt_multiwoz_0524_lr5e-5_steps10000_bz32ac8_block256_warmup0/checkpoint-4100
6. dialogbert/tapt/dapt_oos_0524_lr1e-4_steps1000_bz32ac8_block256_warmup0/checkpoint-1000/
7. dialogbert/tapt/dapt_m2m_0524_lr5e-5_steps500_bz32ac8_block256_warmup0/checkpoint-200/

### MLM test

In `convlab2/ptm/pretraining/eval_mlm.sh`, set hyperparameters and run.

### Fine-tune

#### BERTNLU

In `convlab2/ptm/eval_tasks/nlu/run.sh`, set pre-trained model path and run.

#### HWU, BANKING, RESTAURANT8K, TOP

In `convlab2/ptm/eval_tasks/dialoglue/run_[hwu|banking|rest8k|top].sh`, set pre-trained model path and run.

Bert: `convlab2/ptm/eval_tasks/dialoglue/run_all_[fewshot|fulldata]_bert.sh`

#### Trippy

In `convlab2/ptm/eval_tasks/dialoglue/trippy/run_[baseline|trippy].sh`, set pre-trained model path and run.

#### TOD-DST

In `convlab2/ptm/eval_tasks/dst/scripts/`, set pre-trained model path and run.

#### OOS, DAP

https://github.com/jasonwu0731/ToD-BERT, replace the pre-trained model path

### RSA

In `convlab2/ptm/eval_tasks/RSA/rsa_analysis.py`, select model path and run.

In `convlab2/ptm/eval_tasks/RSA/RSA.ipynb`, compute rsa similarity.