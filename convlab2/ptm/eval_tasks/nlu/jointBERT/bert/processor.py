import torch
from transformers import BertTokenizer
import re


class BertProcessor:
    def __init__(self, pretrained_weights, intent_dim, reverse_order=False):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.intent_dim = intent_dim
        self.reverse_order = reverse_order

    def prepare_input_fn(self, role, tokens, tags, context, context_size, cut_sen_len):
        tokenized_tokens, tokenized_tags = [], []
        # reverse order [U_t, S_t-1, U_t-1, ..., ]
        # normal order [..., U_t-1, S_t-1, U_t]
        if self.reverse_order:
            tokenized_tokens += ['[CLS]'] + tokens + ['[SEP]']
            tokenized_tags += ['X'] + tags + ['X']
            for i, context_tokens in enumerate(context[::-1]):
                if i >= context_size:
                    break
                context_tokens = self.tokenizer.tokenize(context_tokens)
                tokenized_tokens += context_tokens + ['[SEP]']
                tokenized_tags += ['X'] * (len(context_tokens) + 1)
        else:
            for i, context_tokens in enumerate(context[::-1]):
                if i >= context_size:
                    break
                context_tokens = self.tokenizer.tokenize(context_tokens)
                tokenized_tokens = context_tokens + ['[SEP]'] + tokenized_tokens
                tokenized_tags = ['X'] * (len(context_tokens) + 1) + tokenized_tags
            tokenized_tokens = ['[CLS]'] + tokenized_tokens + tokens + ['[SEP]']
            tokenized_tags = ['X'] + tokenized_tags + tags + ['X']

        assert len(tokenized_tokens) == len(tokenized_tags)
        return tokenized_tokens[:cut_sen_len], tokenized_tags[:cut_sen_len]

    def batch_fn(self, batch_data):
        # d = ('user'/'sys', tokens, tags, intents, dialog_act, context, ori_tokens, bert_token2ori_token
        # tokenized_tokens, tokenized_tags, tags_id, intents_id)
        batch_size = len(batch_data)
        max_seq_len = max([len(x[-4]) for x in batch_data])
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        intent_tensor = torch.zeros((batch_size, self.intent_dim), dtype=torch.float)
        for i in range(batch_size):
            tokens = batch_data[i][-4]
            tags = batch_data[i][-2]
            intents = batch_data[i][-1]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            sen_len = len(tokens)
            word_seq_tensor[i, :sen_len] = torch.LongTensor([indexed_tokens])
            word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)
            tag_seq_tensor[i, :sen_len] = torch.LongTensor(tags)
            tag_mask_tensor[i, :sen_len] = torch.LongTensor([0 if x==-1 else 1 for x in tags]) # -1:'X'
            for j in intents:
                intent_tensor[i, j] = 1.
        return word_seq_tensor, word_mask_tensor, tag_seq_tensor, tag_mask_tensor, intent_tensor

    def recover_dialogue_act(self, dataloader, slot_logits, intent_logits, tag_mask_tensor, ori_tokens, bert_token2ori_token, tokenized_tokens):
        # ori_batch: role, tokens, tags, intents, dialog_act, context, ori_tokens, bert_token2ori_token, tokenized_tokens, tokenized_tags, tags_id, intents_id
        # pad_batch: word_seq_tensor, word_mask_tensor, tag_seq_tensor, tag_mask_tensor, intent_tensor
        intents = []
        for j in range(dataloader.intent_dim):
            if intent_logits[j] > 0:
                intent, slot, value = re.split('[+*]', dataloader.id2intent[j])
                intents.append([intent, slot, value])
        tags = []
        # print('ori_tokens', ori_tokens)
        # print('tokenized_tokens', tokenized_tokens)
        max_len = max(map(int, bert_token2ori_token.keys())) + 1
        if self.reverse_order:
            for j in range(1, max_len+1):
                if tag_mask_tensor[j] == 1:
                    _, tag_id = torch.max(slot_logits[j], dim=-1)
                    tags.append(dataloader.id2tag[tag_id.item()])
                else:
                    tags.append('X')
            assert len(tags) == max_len, print(tokenized_tokens, tags)
        else:
            for j in range(len(tokenized_tokens)-max_len-1, len(tokenized_tokens)-1):
                if tag_mask_tensor[j] == 1:
                    _, tag_id = torch.max(slot_logits[j], dim=-1)
                    tags.append(dataloader.id2tag[tag_id.item()])
                else:
                    tags.append('X')
            assert len(tags) == max_len, print(tokenized_tokens, tags)

        ori_tags = ['S'] * len(ori_tokens)
        for i, tag in enumerate(tags):
            if tag != 'X':
                ori_idx = bert_token2ori_token[str(i)]
                ori_tags[ori_idx] = tag

        # print('ori_tokens', ori_tokens)
        # print('tokenized_tokens', tokenized_tokens)
        # print(max_len)
        # print(bert_token2ori_token)
        # print('tags', tags)
        # print('tag_mask_tensor', tag_mask_tensor)
        # print('ori_tags', ori_tags)
        # print(list(zip(ori_tokens, ori_tags)))
        # print()

        triples = []
        i = 0
        while i < len(ori_tags):
            tag = ori_tags[i]
            if tag.startswith('B'):
                intent, slot = tag[2:].split('+')
                value = ori_tokens[i]
                j = i + 1
                while j < len(ori_tags):
                    if ori_tags[j].startswith('I') and ori_tags[j][2:] == tag[2:]:
                        value += ori_tokens[j]
                        i += 1
                        j += 1
                    elif ori_tags[j] == 'S':
                        assert not ori_tokens[j].strip(), print(j, [ori_tokens[j]], ori_tags[j])
                        value += ori_tokens[j]
                        i += 1
                        j += 1
                    else:
                        break
                triples.append([intent, slot, value.strip()])
            i += 1
        # print(triples)
        # print('='*50)
        # intents += tag2triples(merge_tokens, merge_tags)
        intents += triples
        return intents

    def recover_intents_tags(self, dataloader, slot_logits, intent_logits, tag_mask_tensor, intent_tensor, tag_seq_tensor):
        # ori_batch: role, tokens, tags, intents, dialog_act, context, ori_tokens, bert_token2ori_token, tokenized_tokens, tokenized_tags, tags_id, intents_id
        # pad_batch: word_seq_tensor, word_mask_tensor, tag_seq_tensor, tag_mask_tensor, intent_tensor
        intents = []
        gold_intents = []
        for j in range(dataloader.intent_dim):
            if intent_logits[j] > 0:
                intent, slot, value = re.split('[+*]', dataloader.id2intent[j])
                intents.append([intent, slot, value])
            if intent_tensor[j] > 0:
                intent, slot, value = re.split('[+*]', dataloader.id2intent[j])
                gold_intents.append([intent, slot, value])

        tags = []
        gold_tags = []
        max_seq_len = slot_logits.size(0)
        for j in range(max_seq_len):
            if tag_mask_tensor[j] == 1:
                _, tag_id = torch.max(slot_logits[j], dim=-1)
                tags.append(dataloader.id2tag[tag_id.item()])
                gold_tags.append(dataloader.id2tag[tag_seq_tensor[j].item()])

        return intents, gold_intents, tags, gold_tags
