import json
import os
from convlab2.ptm.pretraining.model import DialogBertTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import torch
import numpy as np
from numpy import tri
import random
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RSADataset(Dataset):
    def __init__(self, data, max_length, tokenizer, sample_token_num):
        """load full dialogue"""
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.data = []
        self.total_token = 0
        skip_tokens = ['[CLS]', '[SEP]', '[USR]', '[SYS]']
        skip_ids = tokenizer.convert_tokens_to_ids(skip_tokens)
        for d in data:
            assert len(d['dialogue']) % 2 == 1
            encoded_inputs = self.tokenizer.prepare_input_seq(d['dialogue'], max_length=self.max_length, return_tensors='pt')
            input_ids = encoded_inputs['input_ids']
            mask = (input_ids!=skip_ids[0]) & (input_ids!=skip_ids[1]) & (input_ids!=skip_ids[2]) & (input_ids!=skip_ids[3])
            self.total_token += mask.sum().item()
            encoded_inputs['rsa_mask'] = mask
            self.data.append(encoded_inputs)
        self.token_sample_prob = sample_token_num/self.total_token + 0.01  # ensure sample enough tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class RSADatasetBERT(Dataset):
    def __init__(self, data, max_length, tokenizer: DialogBertTokenizer, sample_token_num):
        """load full dialogue"""
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.data = []
        self.total_token = 0
        skip_tokens = ['[CLS]', '[SEP]', '[USR]', '[SYS]']
        skip_ids = tokenizer.convert_tokens_to_ids(skip_tokens)
        for d in data:
            assert len(d['dialogue']) % 2 == 1
            encoded_inputs = self.tokenizer.prepare_input_seq(d['dialogue'], max_length=self.max_length, return_tensors='pt')
            input_ids = encoded_inputs['input_ids']
            usr_sys_mask = (input_ids!=skip_ids[2]) & (input_ids!=skip_ids[3])
            input_ids = input_ids[usr_sys_mask].unsqueeze(0)
            mask = (input_ids!=skip_ids[0]) & (input_ids!=skip_ids[1])
            self.total_token += mask.sum().item()
            encoded_inputs['input_ids'] = input_ids
            encoded_inputs['rsa_mask'] = mask
            self.data.append(encoded_inputs)
        self.token_sample_prob = sample_token_num/self.total_token + 0.01  # ensure sample enough tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def rsa_collate_fn(batch_data):
    batch_size = len(batch_data)
    max_seq_len = max([x['input_ids'].size(1) for x in batch_data])
    input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    rsa_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    for i in range(batch_size):
        sen_len = batch_data[i]['input_ids'].size(1)
        input_ids[i, :sen_len] = batch_data[i]['input_ids']
        rsa_mask[i, :sen_len] = batch_data[i]['rsa_mask']
    output_data = {
        "input_ids": input_ids,
        "attention_mask": input_ids > 0,
        "rsa_mask": rsa_mask
    }
    return output_data


def move_input_to_device(inputs, device):
    if isinstance(inputs, list):
        return [move_input_to_device(v, device) for v in inputs]
    elif isinstance(inputs, dict):
        return {k: move_input_to_device(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    else:
        return inputs


if __name__ == '__main__':
    data_dir = '/home/zhuqi/research/Platform/ConvLab2-Pretraining/convlab2/ptm/pretraining/prepare_data/full_dialog_mlm'
    datasets_names = [
        # 'hwu',
        # 'banking',
        # 'restaurant8k',
        # 'top',
        'multiwoz21',
        # 'oos',
        # 'm2m'
    ]
    tokenizer = DialogBertTokenizer.from_pretrained(
        '/home/data/zhuqi/pre-trained-models/dialogbert/dapt/1.0data_0506_lr1e-4_steps40000_bz32ac8_block256_warmup0')
    pretrained_models_paths = [
        # '/home/data/zhuqi/pre-trained-models/bert-base-uncased',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/dapt/1.0data_0506_lr1e-4_steps40000_bz32ac8_block256_warmup0/checkpoint-39000',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/dapt/0.25data_0517_lr5e-5_steps10000_bz32ac8_block256_warmup0',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/dapt/0.05data_0518_lr2e-5_steps5000_bz32ac8_block256_warmup0/checkpoint-4000',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/dapt/0.01data_0518_lr2e-5_steps1000_bz32ac8_block256_warmup0',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/hwu_0520_lr5e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-700',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/banking_0520_lr5e-5_steps1000_bz32ac8_block256_warmup0',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/oos_0520_lr1e-4_steps1000_bz32ac8_block256_warmup0',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/multiwoz_0519_lr5e-5_steps10000_bz32ac8_block256_warmup0',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/restaurant8k_0520_lr2e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-900',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/top_0520_lr5e-5_steps2000_bz32ac8_block256_warmup0/checkpoint-1800',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/m2m_0520_lr5e-5_steps500_bz32ac8_block256_warmup0/checkpoint-400',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/dstc2_0520_lr1e-4_steps1000_bz32ac8_block256_warmup0/checkpoint-600',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/dapt_hwu_0524_lr5e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-300',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/dapt_banking_0524_lr5e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-600',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/dapt_restaurant8k_0524_lr2e-5_steps1000_bz32ac8_block256_warmup0/checkpoint-300',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/dapt_top_0524_lr5e-5_steps2000_bz32ac8_block256_warmup0/checkpoint-1900',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/dapt_multiwoz_0524_lr5e-5_steps10000_bz32ac8_block256_warmup0/checkpoint-4100',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/dapt_oos_0524_lr1e-4_steps1000_bz32ac8_block256_warmup0/checkpoint-1000',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/dapt_m2m_0524_lr5e-5_steps500_bz32ac8_block256_warmup0/checkpoint-200',
        # '/home/data/zhuqi/pre-trained-models/dialogbert/tapt/dapt_dstc2_0524_lr1e-4_steps1000_bz32ac8_block256_warmup0/checkpoint-400'
    ]
    ft_tasks = [
        # 'hwu',
        # 'banking',
        # 'restaurant8k',
        # 'top',
        # 'oos',
        # 'bertnlu',
        # 'trippy',
        'tod-dst',
        'dap_mwoz',
        # 'dap_gsim'
    ]
    for ft_task in ft_tasks:
        for root, dirs, files in os.walk('/home/data/zhuqi/pre-trained-models/dialogbert/ft/'+ft_task):
            for dir_name in dirs:
                if dir_name.endswith('data'):
                    pretrained_models_paths.append(os.path.join(root, dir_name))
    print(pretrained_models_paths)
    device = 'cuda:3'
    max_num_token = 5000
    max_seq_len = 512
    tril_mask = tri(max_num_token, dtype=bool)
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rsa_test_output')
    datasets, dataloaders = {}, {}
    datasets_bert, dataloaders_bert = {}, {}
    dialogbert_config = '/home/data/zhuqi/pre-trained-models/dialogbert/dapt/1.0data_0506_lr1e-4_steps40000_bz32ac8_block256_warmup0/config.json'
    bert_config = '/home/data/zhuqi/pre-trained-models/bert-base-uncased/config.json'
    for dataset_name in datasets_names:
        data = json.load(open(os.path.join(data_dir, 'rsa_{}_test_data.json'.format(dataset_name))))
        dataset = RSADataset(data, max_seq_len, tokenizer, max_num_token)
        dataloader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=32,
            collate_fn=rsa_collate_fn,
            num_workers=4
        )
        dataset_bert = RSADatasetBERT(data, max_seq_len, tokenizer, max_num_token)
        dataloader_bert = DataLoader(
            dataset_bert,
            sampler=SequentialSampler(dataset_bert),
            batch_size=32,
            collate_fn=rsa_collate_fn,
            num_workers=4
        )
        print('dataset', dataset_name)
        print('total tokens: {}, token_sample_prob: {}'.format(dataset.total_token, dataset.token_sample_prob))
        datasets[dataset_name] = dataset
        dataloaders[dataset_name] = dataloader
        datasets_bert[dataset_name] = dataset_bert
        dataloaders_bert[dataset_name] = dataloader_bert
    for model_path in tqdm(pretrained_models_paths, desc='model'):
        if '/ft/' in model_path:
            model_name = '-'.join(model_path.split('/')[-3:])
        else:
            model_name = model_path.split('/')[-1] if 'checkpoint' not in model_path.split('/')[-1] else model_path.split('/')[-2]
        if '/ft/' in model_path and any([x in model_path for x in ['hwu', 'banking', 'restaurant8k', 'top']]):
            state_dict = torch.load(model_path+'/model.pt')
            config = bert_config if 'bert' in model_path.split('/')[-1] else dialogbert_config
            new_state_dict = {}
            for key in state_dict:
                if 'bert_model' in key:
                    new_state_dict[key.replace('bert_model', 'bert')] = state_dict[key]
            del state_dict
            model = BertModel.from_pretrained(None, config=config, state_dict=new_state_dict).to(device)
        elif '/ft/' in model_path and any([x in model_path for x in ['tod-dst', 'oos', 'dap_mwoz', 'dap_gsim']]):
            state_dict = torch.load(model_path + '/pytorch_model.bin')
            config = bert_config if 'bert' in model_path.split('/')[-1] else dialogbert_config
            new_state_dict = {}
            for key in state_dict:
                if 'utterance_encoder' in key:
                    new_state_dict[key.replace('utterance_encoder', 'bert')] = state_dict[key]
            del state_dict
            model = BertModel.from_pretrained(None, config=config, state_dict=new_state_dict).to(device)
        else:
            model = BertModel.from_pretrained(model_path).to(device)
        model.encoder.output_hidden_states = True
        for dataset_name in tqdm(datasets_names, desc='dataset'):
            show = False
            if 'bert' in model_path.split('/')[-1]:
                dataset = datasets_bert[dataset_name]
                dataloader = dataloaders_bert[dataset_name]
            else:
                dataset = datasets[dataset_name]
                dataloader = dataloaders[dataset_name]
            set_seed(42)
            token_reprs = [[] for i in range(model.config.num_hidden_layers)]
            for batch_data in dataloader:
                batch_data = move_input_to_device(batch_data, device)
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = model(input_ids=batch_data['input_ids'],
                                                                          attention_mask=batch_data['attention_mask'])
                    # hidden_states: Tuple of :obj:`torch.FloatTensor` (embeddings + output of each layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`
                    probability_matrix = torch.full((batch_data['rsa_mask'].sum().item(),), dataset.token_sample_prob)
                    sample_indices = torch.bernoulli(probability_matrix).bool()
                    if show:
                        print(batch_data['rsa_mask'].size(), len(sample_indices))
                        print(sample_indices)
                        print(tokenizer.convert_ids_to_tokens(batch_data['input_ids'][batch_data['rsa_mask']][sample_indices]))
                        show=False
                    for i in range(model.config.num_hidden_layers):
                        token_repr = hidden_states[i + 1][batch_data['rsa_mask']][sample_indices]
                        token_reprs[i].append(token_repr.cpu().numpy())
            for i in range(model.config.num_hidden_layers):
                token_reprs[i] = np.concatenate(token_reprs[i], axis=0)[:5000]
                assert len(token_reprs[i]) == 5000
                cos_sim = cosine_similarity(token_reprs[i], token_reprs[i])
                flatten_sim_score = cos_sim[~tril_mask]
                save_path = os.path.join(save_dir, '{}_layer{}_{}'.format(dataset_name, i, model_name))
                np.save(save_path, flatten_sim_score)
