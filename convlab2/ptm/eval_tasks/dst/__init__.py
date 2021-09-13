from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')

def get_transformer_settings(name):
    if name in ['bert', 'tod-bert']:
        from transformers import BertTokenizer, BertModel, BertConfig
        return BertTokenizer, BertModel, BertConfig
    elif name == 'dialog-bert':
        from convlab2.ptm.pretraining.model import DialogBertTokenizer, DialogBertModel, DialogBertConfig
        return DialogBertTokenizer, DialogBertModel, DialogBertConfig
    else:
        raise Exception(f'unrecognized transformer name: {name}')

import random
import numpy
import torch
