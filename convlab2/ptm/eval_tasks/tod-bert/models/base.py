import torch
from torch import nn

from convlab2.ptm.pretraining.model import DialogBertModel


# base model for evaluation models using dialog-bert
class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args['model_type'] == 'dialogbert'
        self.utterance_encoder = DialogBertModel.from_pretrained(args["model_name_or_path"])
        if args["fix_encoder"]:
            print("[Info] Fixing Encoder...")
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False
        self.args = args

    def get_turn_embeddings_num(self):
        return self.utterance_encoder.get_turn_embeddings().weight.size(0)

    def encode_context(self, context):
        with torch.set_grad_enabled(not self.args['fix_encoder']):
            outputs = self.utterance_encoder(**context)
            sentence_rep = self.args['sentence_rep']
            if sentence_rep == 'sequence0':
                return outputs[0][:, 0, :]
            elif sentence_rep == 'sequence1':
                return outputs[0][:, 1, :]
            else:
                assert sentence_rep == 'pooled'
                return outputs[1]
