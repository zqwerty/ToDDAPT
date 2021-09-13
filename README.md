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

### TAPT

In `convlab2/ptm/pretraining/run_tapt.sh`, set hyperparameters and run.

### DAPT+TAPT

In `convlab2/ptm/pretraining/run_dapt_tapt.sh`, set hyperparameters and run.

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