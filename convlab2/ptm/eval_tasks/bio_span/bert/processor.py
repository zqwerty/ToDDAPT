import torch
from transformers import BertTokenizer
import re
from typing import List


class BertProcessor:
    def __init__(self, pretrained_weights, labels_map):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.labels_map = labels_map
        self.tag_O_id = self.labels_map['label_to_id']['O']

    def batch_fn(self, batch_data):
        # d = ('user'/'sys', tokens, tags, intents, dialog_act, context, ori_tokens, bert_token2ori_token
        # tokenized_tokens, tokenized_tags, tags_id, intents_id)
        batch_size = len(batch_data)
        max_seq_len = max([len(x['input_ids']) for x in batch_data])
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_seq_tensor = torch.ones((batch_size, max_seq_len), dtype=torch.long) * self.tag_O_id
        dial_ids = []
        turn_ids = []
        golden_update = []
        all_cate_labels = []
        all_noncate_labels = []

        for i in range(batch_size):
            all_cate_labels.append(batch_data[i]['categorical'])
            all_noncate_labels.append(batch_data[i]['non-categorical'])
            golden_update.append(batch_data[i]['golden_state_update'])
            dial_ids.append(batch_data[i]['dial_id'])
            turn_ids.append(batch_data[i]['turn_id'])
            tokens = batch_data[i]['input_ids']
            tags = batch_data[i]['bio_labels']
            sen_len = len(tokens)
            word_seq_tensor[i, :sen_len] = torch.LongTensor(tokens)
            word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)
            tag_seq_tensor[i, :sen_len] = torch.LongTensor(tags)
            tag_mask_tensor[i, :sen_len] = torch.LongTensor([0 if x==-1 else 1 for x in tags]) # -1:'X'
        # word_mask_tensor, position_id_tensor = None, turn_id_tensor = None, role_id_tensor = None,
        # tag_seq_tensor = None, tag_mask_tensor = None
        all_cate_labels = torch.LongTensor(all_cate_labels)  # (bs, num_slot)
        all_cate_labels = torch.split(all_cate_labels, 1, dim=-1)  # [bs, 1]
        all_cate_labels = [l.squeeze(-1) for l in all_cate_labels]
        all_noncate_labels = torch.LongTensor(all_noncate_labels)
        all_noncate_labels = torch.split(all_noncate_labels, 1, dim=-1)  # [bs, 1]
        all_noncate_labels = [l.squeeze(-1) for l in all_noncate_labels]
        return [{'word_seq_tensor': word_seq_tensor, 'word_mask_tensor': word_mask_tensor,
                'tag_seq_tensor': tag_seq_tensor, 'tag_mask_tensor': tag_mask_tensor, 'cate_slot_labels': all_cate_labels,
                 'noncate_slot_labels': all_noncate_labels}, dial_ids, turn_ids, golden_update]


    def recover_dialogue_act(self, tag_ids:List[int], input_ids:List[int]):
        # ori_batch: role, tokens, tags, intents, dialog_act, context, ori_tokens, bert_token2ori_token, tokenized_tokens, tokenized_tags, tags_id, intents_id
        # pad_batch: word_seq_tensor, word_mask_tensor, tag_seq_tensor, tag_mask_tensor, intent_tensor
        tags = [self.labels_map['id_to_label'][_id] for _id in tag_ids]

        pred_state = {}
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith('B'):
                _domain, _slot = tag[2:].split('-')
                value_ids = [input_ids[i]]
                j = i + 1
                while j < len(tags):
                    if tags[j].startswith('I') and tags[j][2:] == tag[2:]:
                        value_ids.append(input_ids[j])
                        i += 1
                        j += 1
                    else:  # 'O'
                        decoded_value = self.tokenizer.decode(value_ids)
                        if _domain not in pred_state:
                            pred_state[_domain] = {}
                        if _slot not in pred_state[_domain]:
                            pred_state[_domain][_slot] = decoded_value
                        # else:
                        #     # print('found multiple tag prediction for domain {} slot {}, value {}'.format(_domain, _slot, [pred_state[_domain][_slot], decoded_value]))
                        #     continue
                        break
            i += 1

        return pred_state
