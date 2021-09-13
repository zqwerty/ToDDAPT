# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """


import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, CosineEmbeddingLoss
import torch.nn.functional as F

from .activations import gelu, gelu_new, swish
from .configuration_dialog_bert import DialogBertConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import PreTrainedModel, prune_linear_layer
from .modeling_bert import BertModel


logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://cdn.huggingface.co/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://cdn.huggingface.co/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://cdn.huggingface.co/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://cdn.huggingface.co/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://cdn.huggingface.co/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://cdn.huggingface.co/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://cdn.huggingface.co/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://cdn.huggingface.co/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://cdn.huggingface.co/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://cdn.huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://cdn.huggingface.co/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://cdn.huggingface.co/bert-base-german-dbmdz-cased-pytorch_model.bin",
    "bert-base-german-dbmdz-uncased": "https://cdn.huggingface.co/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    "bert-base-japanese": "https://cdn.huggingface.co/cl-tohoku/bert-base-japanese/pytorch_model.bin",
    "bert-base-japanese-whole-word-masking": "https://cdn.huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/pytorch_model.bin",
    "bert-base-japanese-char": "https://cdn.huggingface.co/cl-tohoku/bert-base-japanese-char/pytorch_model.bin",
    "bert-base-japanese-char-whole-word-masking": "https://cdn.huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/pytorch_model.bin",
    "bert-base-finnish-cased-v1": "https://cdn.huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    "bert-base-finnish-uncased-v1": "https://cdn.huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
    "bert-base-dutch-cased": "https://cdn.huggingface.co/wietsedv/bert-base-dutch-cased/pytorch_model.bin",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


BertLayerNorm = torch.nn.LayerNorm


class DialogBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.turn_embeddings = nn.Embedding(config.max_turn_embeddings, config.hidden_size)
        # self.role_embeddings = nn.Embedding(config.max_role_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, turn_ids, position_ids, role_ids, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # if position_ids is None:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        # if turn_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # turn_embeddings = self.turn_embeddings(turn_ids)
        # role_embeddings = self.role_embeddings(role_ids)

        # embeddings = inputs_embeds + position_embeddings + turn_embeddings + role_embeddings
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# class DialogBertEmbeddings(nn.Module):
#     """Construct the embeddings from word, position and token_type embeddings.
#     """
#
#     def __init__(self, config):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
#         self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
#         # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
#         self.turn_embeddings = nn.Embedding(config.max_turn_embeddings, config.hidden_size)
#         self.role_embeddings = nn.Embedding(config.max_role_embeddings, config.hidden_size)
#
#         # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
#         # any TensorFlow checkpoint file
#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#     def forward(self, input_ids, turn_ids, position_ids, role_ids, inputs_embeds=None):
#         # if input_ids is not None:
#         #     input_shape = input_ids.size()
#         # else:
#         #     input_shape = inputs_embeds.size()[:-1]
#
#         # seq_length = input_shape[1]
#         # device = input_ids.device if input_ids is not None else inputs_embeds.device
#         # if position_ids is None:
#         #     position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
#         #     position_ids = position_ids.unsqueeze(0).expand(input_shape)
#         # if turn_ids is None:
#         #     token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
#
#         if inputs_embeds is None:
#             inputs_embeds = self.word_embeddings(input_ids)
#         position_embeddings = self.position_embeddings(position_ids)
#         turn_embeddings = self.turn_embeddings(turn_ids)
#         role_embeddings = self.role_embeddings(role_ids)
#
#         embeddings = inputs_embeds + position_embeddings + turn_embeddings + role_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class DialogBertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DialogBertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class DialogBertModel(DialogBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = DialogBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_turn_embeddings(self):
        return self.embeddings.turn_embeddings

    def resize_turn_embeddings(self, new_num_tokens=None):
        old_embeddings = self.embeddings.turn_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.turn_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        turn_ids=None,
        position_ids=None,
        role_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertModel, BertTokenizer
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, turn_ids=turn_ids, role_ids=role_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class DialogBertTokenHead(nn.Module):
    def __init__(self, config, num_classes, use_transform_layer=True):
        super().__init__()
        self.config = config
        self.use_transform_layer = use_transform_layer
        if self.use_transform_layer:
            self.transform = BertPredictionHeadTransform(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.decoder = nn.Linear(config.hidden_size, num_classes)

        self.apply(self._init_weights)

    def forward(self, sequence_output, seq_mask=None):
        if seq_mask is not None:
            active_tokens = seq_mask.view(-1) == 1
            sequence_output = sequence_output.view(-1, sequence_output.size(-1))[active_tokens]
        if self.use_transform_layer:
            sequence_output = self.transform(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.decoder(sequence_output)
        return logits

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # nn.init.xavier_uniform_(module.weight)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DialogBertCLSSIMHead(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.decoder = nn.Linear(config.hidden_size * 4, num_classes)

        self.apply(self._init_weights)

    def forward(self, a_cls, b_cls):
        concat_input = torch.cat((a_cls, b_cls, torch.abs(a_cls-b_cls), a_cls * b_cls), dim=-1)
        logits = self.decoder(concat_input)
        return logits

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # nn.init.xavier_uniform_(module.weight)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DialogBertMLMCLSHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(2 * config.hidden_size, config.vocab_size + 9)

    def forward(self, pooled_output, sequence_output):
        sequence_output = self.transform(sequence_output)
        cls_mlm_hidden = torch.cat((pooled_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1), sequence_output), dim=-1)
        cls_mlm_logits = self.decoder(cls_mlm_hidden)
        return cls_mlm_logits


def MeanPooling(input, mask):
    """

    :param input: (batch_size, max_seq_len, hidden_size)
    :param mask: (batch_size, max_seq_len)
    :return:
    """
    masked_seq = input * mask.unsqueeze(-1)  # (batch_size, max_seq_len, hidden_state)
    return torch.sum(masked_seq, dim=1) / torch.sum(mask, dim=1, keepdim=True)  # (batch_size, hidden_state)


@add_start_docstrings("""DialogBert Model for pre-training. """, BERT_START_DOCSTRING)
class DialogBertForPretraining(DialogBertPreTrainedModel):
    def __init__(self, config, training_args, model_args, data_args, metadataloader):
        super().__init__(config)
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.metadataloader = metadataloader

        # self.bert = DialogBertModel(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

        if self.data_args.schema_linking:
            self.intent_head = DialogBertTokenHead(config, num_classes=metadataloader.processors['schema_linking'].intent_dim['schema'])
            self.domain_head = DialogBertTokenHead(config, num_classes=metadataloader.processors['schema_linking'].domain_dim['schema'])
            self.slot_head = DialogBertTokenHead(config, num_classes=metadataloader.processors['schema_linking'].slot_dim['schema'])
            self.bio_head = DialogBertTokenHead(config, num_classes=metadataloader.processors['schema_linking'].slot_dim['schema']*2+1)

        if self.data_args.bio:
            self.bio_head = DialogBertTokenHead(config, num_classes=3)

        if self.data_args.ssl:
            # self.bio_head = DialogBertTokenHead(config, num_classes=3)
            # self.bio_head = DialogBertBIOHead(config)
            self.tf_idf_head = DialogBertTokenHead(config, num_classes=1)

        if self.data_args.cls_mlm:
            self.cls_mlm_head = DialogBertMLMCLSHead(config)

        if self.data_args.augdial or self.data_args.augdial_ssl:
            if self.data_args.use_label:
                self.intent_head = DialogBertTokenHead(config, num_classes=metadataloader.processors['augdial'].intent_dim['schema'])
                self.domain_head = DialogBertTokenHead(config, num_classes=metadataloader.processors['augdial'].domain_dim['schema'])
                self.slot_head = DialogBertTokenHead(config, num_classes=metadataloader.processors['augdial'].slot_dim['schema'])
                self.tokenslot_head = DialogBertTokenHead(config, num_classes=metadataloader.processors['augdial'].slot_dim['schema']*2+1)
            if self.data_args.cls_contrastive:
                if self.data_args.cls_contrastive_type == 5:
                    self.cls_sim_head1 = DialogBertCLSSIMHead(config, num_classes=2)
                    self.cls_sim_head2 = DialogBertCLSSIMHead(config, num_classes=2)
                    self.cls_contrastive_head1 = DialogBertTokenHead(config, num_classes=config.hidden_size, use_transform_layer=False)
                    self.cls_contrastive_head2 = DialogBertTokenHead(config, num_classes=config.hidden_size, use_transform_layer=False)
                else:
                    self.cls_sim_head = DialogBertCLSSIMHead(config, num_classes=2)
                    self.cls_contrastive_head = DialogBertTokenHead(config, num_classes=config.hidden_size, use_transform_layer=False)

        # if self.data_args.dapt:
        #     # dapt train dataset
        #     self.slot_dim = metadataloader.processors['dapt'].slot_dim
        #     self.intent_dim = metadataloader.processors['dapt'].intent_dim
        #     self.domain_dim = metadataloader.processors['dapt'].domain_dim
        #     # self.slot_datasets = [k for k, v in self.slot_dim.items() if v > 1]
        #     # self.intent_datasets = [k for k, v in self.intent_dim.items() if v > 1]
        #     # self.domain_datasets = [k for k, v in self.domain_dim.items() if v > 1]
        #     # self.slot_datasets = ['schema', 'taskmaster']
        #     self.slot_datasets = ['schema']
        #     self.intent_datasets = ['schema']
        #     # self.domain_datasets = ['schema', 'taskmaster']
        #     self.domain_datasets = ['schema']
        #     if self.data_args.biotagging:
        #         self.tokenslot_heads = nn.ModuleList()
        #         for dataset in self.slot_datasets:
        #             self.tokenslot_heads.append(DialogBertTokenHead(config, num_classes=self.slot_dim[dataset] * 2 + 1))
        #
        #     if self.data_args.sencls or self.data_args.dialcls:
        #         self.slot_heads = nn.ModuleList()
        #         self.intent_heads = nn.ModuleList()
        #         self.domain_heads = nn.ModuleList()
        #         for dataset in self.slot_datasets:
        #             self.slot_heads.append(DialogBertTokenHead(config, num_classes=self.slot_dim[dataset]))
        #         for dataset in self.intent_datasets:
        #             self.intent_heads.append(DialogBertTokenHead(config, num_classes=self.intent_dim[dataset]))
        #         for dataset in self.domain_datasets:
        #             self.domain_heads.append(DialogBertTokenHead(config, num_classes=self.domain_dim[dataset]))


    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(self, task, dataset,
        attention_mask=None, input_ids=None, turn_ids=None, position_ids=None, role_ids=None,
        head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
        masked_lm_labels=None, # MLM task
        bio_tag_ids=None, # BIO task
        token_tag_ids=None,
        intent_tag_ids=None, domain_tag_ids=None, slot_tag_ids=None, sen_cls_mask=None,
        cls_intent_tag_ids=None, cls_domain_tag_ids=None, cls_slot_tag_ids=None,
        resp_labels=None, # response selection task
        spans=None, neg_spans=None,
        evaluate=False,
        tf_idf=None, token_mask=None,
        **kwargs
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
                Next token prediction loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Examples::

            from transformers import BertTokenizer, BertForMaskedLM
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, masked_lm_labels=input_ids)

            loss, prediction_scores = outputs[:2]

        """

        if task == 'augdial':
            # {batches: ori_dial, pos_dial1, ... neg_dial1, ...}
            outputs = {}
            loss = 0
            cls_outputs = ()  # (1 + pos_aug_num + neg_aug_num + pick1utt_num, [batch, hidden])
            for i, batch in enumerate(kwargs["batches"]):
                if 0 < i and self.data_args.nograd4aug:
                    # no grad for augmented sample
                    with torch.no_grad():
                        bert_outputs = self.bert(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            turn_ids=batch["turn_ids"],
                            position_ids=batch["position_ids"],
                            role_ids=batch["role_ids"]
                        )
                else:
                    bert_outputs = self.bert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        turn_ids=batch["turn_ids"],
                        position_ids=batch["position_ids"],
                        role_ids=batch["role_ids"]
                    )
                sequence_output = bert_outputs[0]
                pooled_output = bert_outputs[1]
                if self.data_args.cls_contrastive:
                    cls_outputs += (pooled_output, )

                if 0 < i and self.data_args.nograd4aug:
                    continue

                if "sen_cls_mask" in batch:
                    sen_cls_mask = batch["sen_cls_mask"]

                if "masked_lm_labels" in batch:
                    prediction_scores = self.cls(sequence_output)  # (batch_size, seq_len, vocab_size)
                    outputs["prediction_scores_aug{}".format(i)] = prediction_scores
                    loss_fct = CrossEntropyLoss()  # ignore -100 index
                    masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                              batch["masked_lm_labels"].view(-1))
                    outputs["aug{}_mlm_loss".format(i)] = masked_lm_loss

                if "token_tag_ids" in batch:
                    tokenslot_logits = self.tokenslot_head(sequence_output)  # (batch_size, seq_len, N)

                    outputs["tokenslot_logits"] = tokenslot_logits

                    loss_fct = CrossEntropyLoss()  # ignore -100 index
                    tokenslot_loss = loss_fct(tokenslot_logits.view(-1, tokenslot_logits.size(-1)),
                                              batch["token_tag_ids"].view(-1))
                    outputs["aug{}_tokenslot_loss".format(i)] = tokenslot_loss

                if "intent_tag_ids" in batch:
                    intent_logits = self.intent_head(sequence_output, sen_cls_mask)
                    outputs["intent_logits"] = intent_logits

                    loss_fct = BCEWithLogitsLoss()
                    active_tokens = sen_cls_mask.view(-1) == 1
                    active_intent_labels = batch["intent_tag_ids"].view(-1, self.metadataloader.processors[task].intent_dim[dataset])[active_tokens]
                    intent_loss = loss_fct(intent_logits, active_intent_labels)
                    outputs["aug{}_intent_loss".format(i)] = intent_loss

                if "domain_tag_ids" in batch:
                    domain_logits = self.domain_head(sequence_output, sen_cls_mask)
                    outputs["domain_logits"] = domain_logits

                    loss_fct = BCEWithLogitsLoss()
                    active_tokens = sen_cls_mask.view(-1) == 1
                    active_domain_labels = batch["domain_tag_ids"].view(-1, self.metadataloader.processors[task].domain_dim[dataset])[active_tokens]
                    domain_loss = loss_fct(domain_logits, active_domain_labels)
                    outputs["aug{}_domain_loss".format(i)] = domain_loss

                if "slot_tag_ids" in batch:
                    slot_logits = self.slot_head(sequence_output, sen_cls_mask)
                    outputs["slot_logits"] = slot_logits

                    loss_fct = BCEWithLogitsLoss()
                    active_tokens = sen_cls_mask.view(-1) == 1
                    active_slot_labels = batch["slot_tag_ids"].view(-1, self.metadataloader.processors[task].slot_dim[dataset])[active_tokens]
                    slot_loss = loss_fct(slot_logits, active_slot_labels)
                    outputs["aug{}_slot_loss".format(i)] = slot_loss

                if "cls_intent_tag_ids" in batch:
                    cls_intent_logits = self.intent_head(pooled_output)
                    outputs["cls_intent_logits"] = cls_intent_logits

                    loss_fct = BCEWithLogitsLoss()
                    cls_intent_loss = loss_fct(cls_intent_logits, batch["cls_intent_tag_ids"])
                    outputs["aug{}_cls_intent_loss".format(i)] = cls_intent_loss

                if "cls_domain_tag_ids" in batch:
                    cls_domain_logits = self.domain_head(pooled_output)
                    outputs["cls_domain_logits"] = cls_domain_logits

                    loss_fct = BCEWithLogitsLoss()
                    cls_domain_loss = loss_fct(cls_domain_logits, batch["cls_domain_tag_ids"])
                    outputs["aug{}_cls_domain_loss".format(i)] = cls_domain_loss

                if "cls_slot_tag_ids" in batch:
                    cls_slot_logits = self.slot_head(pooled_output)
                    outputs["cls_slot_logits"] = cls_slot_logits

                    loss_fct = BCEWithLogitsLoss()
                    cls_slot_loss = loss_fct(cls_slot_logits, batch["cls_slot_tag_ids"])
                    outputs["aug{}_cls_slot_loss".format(i)] = cls_slot_loss

                aug_dial_loss = outputs.get("aug{}_mlm_loss".format(i), 0) + \
                                outputs.get("aug{}_tokenslot_loss".format(i), 0) + \
                                outputs.get("aug{}_intent_loss".format(i), 0) + \
                                outputs.get("aug{}_domain_loss".format(i), 0) + \
                                outputs.get("aug{}_slot_loss".format(i), 0) + \
                                outputs.get("aug{}_cls_intent_loss".format(i), 0) + \
                                outputs.get("aug{}_cls_domain_loss".format(i), 0) + \
                                outputs.get("aug{}_cls_slot_loss".format(i), 0)

                # if aug_dial_loss != 0:
                outputs["aug{}_loss".format(i)] = aug_dial_loss
                loss += outputs["aug{}_loss".format(i)]

            if self.data_args.cls_contrastive:
                batch_size = cls_outputs[0].size(0)
                if self.data_args.cls_contrastive_type == 0:
                    # classification way, sim_dist [2*batch, 2]
                    # ori dial x, pos aug x+, shifted pos aug x- from another dial
                    # f(g(x), g(x+)) -> 1, f(g(x), g(x-)) -> 0
                    # g: MLP, f: softmax(linear(a,b,|a-b|,a*b))
                    assert self.data_args.pick1utt_num + self.data_args.pos_aug_num == 1
                    assert self.data_args.neg_aug_num == 0
                    ori_cls_repr = self.cls_contrastive_head(cls_outputs[0])
                    pos_cls_repr = self.cls_contrastive_head(cls_outputs[1])
                    shifted_pos_cls_repr = torch.roll(pos_cls_repr, shifts=1, dims=0)
                    pos_sim_class = torch.zeros(batch_size, dtype=torch.long, device=ori_cls_repr.device)
                    pos_sim_logit = self.cls_sim_head(ori_cls_repr, pos_cls_repr)  # [batch, 2]
                    neg_sim_class = torch.ones(batch_size, dtype=torch.long, device=ori_cls_repr.device)
                    neg_sim_logit = self.cls_sim_head(ori_cls_repr, shifted_pos_cls_repr)  # [batch, 2]
                    sim_logit = torch.cat((pos_sim_logit, neg_sim_logit), dim=0)
                    sim_class = torch.cat((pos_sim_class, neg_sim_class), dim=0)

                    loss_fct = CrossEntropyLoss()
                    sim_loss = loss_fct(sim_logit, sim_class)
                    loss += sim_loss

                    with torch.no_grad():
                        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                        sim_dist = F.softmax(sim_logit, dim=-1)
                        acc = torch.sum(torch.max(sim_logit, dim=-1).indices == sim_class).item() / sim_class.size(0)
                        cos_sim = torch.cat((cos(cls_outputs[0], cls_outputs[1]).unsqueeze(-1),
                                             cos(cls_outputs[0], torch.roll(cls_outputs[1], shifts=1, dims=0)).unsqueeze(-1)),
                                            dim=-1)
                    outputs["aug{}_sim_loss".format(len(cls_outputs)-1)] = sim_loss
                    outputs["aug{}_sim_dist".format(len(cls_outputs) - 1)] = sim_dist
                    outputs["aug{}_sim_acc".format(len(cls_outputs) - 1)] = acc
                    outputs["aug{}_cossim".format(len(cls_outputs) - 1)] = cos_sim
                    outputs["acc_loss"] = acc

                elif self.data_args.cls_contrastive_type == 1:
                    # select pos one from 3 samples: (pos_aug x_p, neg_aug x_n, shift_ori_dial x_s)
                    # ori_dial x, x': (pos_aug x_p, neg_aug x_n, shift_ori_dial x_s)
                    # softmax(cos(g(x), g(x')))
                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

                    assert self.data_args.pos_aug_num == 1
                    assert self.data_args.neg_aug_num == 1
                    ori_cls_repr = self.cls_contrastive_head(cls_outputs[0])
                    pos_aug_repr = self.cls_contrastive_head(cls_outputs[1])
                    neg_aug_repr = self.cls_contrastive_head(cls_outputs[2])
                    shifted_ori_cls_repr = torch.roll(ori_cls_repr, shifts=1, dims=0)

                    # cossim_ori_xx: [batch, 1]
                    cossim_ori_pos = cos(ori_cls_repr, pos_aug_repr).unsqueeze_(-1)
                    cossim_ori_neg = cos(ori_cls_repr, neg_aug_repr).unsqueeze_(-1)
                    cossim_ori_shifted_ori = cos(ori_cls_repr, shifted_ori_cls_repr).unsqueeze_(-1)
                    # cossim: [batch, 3]
                    cossim = torch.cat((cossim_ori_pos, cossim_ori_neg, cossim_ori_shifted_ori), -1)
                    sim_class = torch.zeros(cossim.size(0), dtype=torch.long, device=cossim.device)
                    loss_fct = CrossEntropyLoss()
                    sim_loss = loss_fct(cossim, sim_class)

                    loss += sim_loss
                    outputs["aug{}_sim_loss".format(0)] = sim_loss
                    outputs["aug{}_sim_dist".format(0)] = F.softmax(cossim, dim=-1)
                    outputs["aug{}_cossim".format(0)] = cossim

                    acc = torch.sum(torch.max(cossim, dim=-1).indices == sim_class).item() / sim_class.size(0)
                    outputs["aug{}_sim_acc".format(0)] = acc
                    outputs["acc_loss"] = acc

                elif self.data_args.cls_contrastive_type == 2:
                    # SimCLR way, sim mat: [2*batch, 2*batch]. no negative.
                    # ori dial x1, pos aug x2 or pos_aug x1, x2
                    # g: MLP
                    # softmax(cos(g(x1), g(x2))/temperature)
                    assert self.data_args.neg_aug_num == 0
                    # [batch, hidden]
                    if self.data_args.pos_aug_num == 2:
                        # pos_aug x1, x2
                        pos1_h = cls_outputs[1]
                        pos2_h = cls_outputs[2]
                    else:
                        # ori_dial x1, pos_aug x2 (can set no grad for aug for larger batch size)
                        pos1_h = cls_outputs[0]
                        pos2_h = cls_outputs[1]
                    # g-transform
                    pos1_z = self.cls_contrastive_head(pos1_h)
                    pos2_z = self.cls_contrastive_head(pos2_h)
                    # l2 normalized
                    pos1_z = pos1_z / pos1_z.norm(dim=1)[:, None]
                    pos2_z = pos2_z / pos2_z.norm(dim=1)[:, None]
                    # [batch*2, hidden]
                    all_z = torch.cat((pos1_z, pos2_z), dim=0)
                    # [batch*2, batch*2], diagonal elements are self-similarity that should be masked
                    sim_mat = torch.mm(all_z, all_z.transpose(0, 1))
                    sim_mat_scaled = sim_mat / self.data_args.temperature
                    sim_mat_scaled[range(sim_mat_scaled.size(0)), range(sim_mat_scaled.size(0))] = float('-inf')

                    # [batch*2] label: (batch+i)%(2*batch) for i in range(2*batch)
                    sim_label = [(x + batch_size) % (2 * batch_size) for x in range(sim_mat.size(0))]
                    sim_label = torch.tensor(sim_label, dtype=torch.long, device=sim_mat.device)

                    loss_fct = CrossEntropyLoss()
                    sim_loss = loss_fct(sim_mat_scaled, sim_label)

                    loss += sim_loss

                    with torch.no_grad():
                        sim_dist = F.softmax(sim_mat_scaled, dim=-1)
                        acc = torch.sum(torch.max(sim_mat_scaled, dim=-1).indices == sim_label).item() / sim_label.size(0)
                        all_h = torch.cat((pos1_h / pos1_h.norm(dim=1)[:, None], pos2_h / pos2_h.norm(dim=1)[:, None]), dim=0)
                        cls_sim_mat = torch.mm(all_h, all_h.transpose(0, 1))
                        cls_sim_mat[range(cls_sim_mat.size(0)), range(cls_sim_mat.size(0))] = float('-inf')

                    outputs["aug{}_sim_loss".format(0)] = sim_loss
                    outputs["aug{}_sim_dist".format(0)] = sim_dist
                    outputs["aug{}_cossim".format(0)] = cls_sim_mat
                    outputs["aug{}_sim_acc".format(0)] = acc
                    outputs["acc_loss"] = acc

                elif self.data_args.cls_contrastive_type == 3:
                    # ori_dial x
                    # x': (pos_aug x_p, neg_aug x_n, other_ori_dial x_s) (len(cls_output)* batch_size-1)
                    # softmax(cos(g(f(x)), g(f(x')))/temperature)
                    # sim mat: [batch, n*batch]
                    assert self.data_args.nograd4aug
                    # ori vs (ori, pos, neg)
                    assert self.data_args.pos_aug_num==1
                    ori_cls_repr = cls_outputs[0]
                    # g-transform
                    ori_cls_z = self.cls_contrastive_head(ori_cls_repr)
                    # l2 normalized
                    ori_cls_z_normalized = ori_cls_z / ori_cls_z.norm(dim=1)[:, None]

                    all_cls_normalized = [ori_cls_repr.detach() / ori_cls_repr.detach().norm(dim=1)[:, None]]
                    all_cls_z_normalized = [ori_cls_z_normalized]
                    for cls_repr in cls_outputs[1:]:
                        all_cls_normalized.append(cls_repr / cls_repr.norm(dim=1)[:, None])
                        cls_z = self.cls_contrastive_head(cls_repr)
                        cls_z_normalized = cls_z / cls_z.norm(dim=1)[:, None]
                        all_cls_z_normalized.append(cls_z_normalized)
                    # [batch*(1+pos_num+neg_num), hidden]
                    all_cls_normalized = torch.cat(all_cls_normalized, dim=0)
                    all_z_normalized = torch.cat(all_cls_z_normalized, dim=0)

                    # [batch, batch*(1+pos_num+neg_num)], diagonal elements are self-similarity that should be masked
                    sim_mat = torch.mm(ori_cls_z_normalized, all_z_normalized.transpose(0, 1))
                    sim_mat_scaled = sim_mat / self.data_args.temperature
                    sim_mat_scaled[range(batch_size), range(batch_size)] = float('-inf')

                    # [batch] label: (batch+i) for i in range(batch)
                    sim_label = [(x + batch_size) for x in range(batch_size)]
                    sim_label = torch.tensor(sim_label, dtype=torch.long, device=sim_mat.device)

                    loss_fct = CrossEntropyLoss()
                    sim_loss = loss_fct(sim_mat_scaled, sim_label)

                    loss += sim_loss

                    acc = torch.sum(torch.max(sim_mat_scaled, dim=-1).indices == sim_label).item() / batch_size

                    with torch.no_grad():
                        sim_dist = F.softmax(sim_mat_scaled, dim=-1)
                        cls_sim_mat = torch.mm(all_cls_normalized, all_cls_normalized.transpose(0, 1))
                        cls_sim_mat[range(cls_sim_mat.size(0)), range(cls_sim_mat.size(0))] = float('-inf')

                    outputs["aug{}_sim_loss".format(0)] = sim_loss
                    outputs["aug{}_sim_dist".format(0)] = sim_dist
                    outputs["aug{}_cossim".format(0)] = cls_sim_mat
                    outputs["aug{}_sim_acc".format(0)] = acc
                    outputs["acc_loss"] = acc

                elif self.data_args.cls_contrastive_type == 4:
                    # ori_dial x
                    # x': (pos_aug x_p, neg_aug x_n)
                    # softmax(cos(g(f(x)), g(f(x')))/temperature)
                    # sim mat: [batch, n*batch]
                    assert self.data_args.nograd4aug
                    # ori vs (pos, neg)
                    assert self.data_args.pos_aug_num==1
                    ori_cls_repr = cls_outputs[0]
                    # g-transform
                    ori_cls_z = self.cls_contrastive_head(ori_cls_repr)
                    # l2 normalized
                    ori_cls_z_normalized = ori_cls_z / ori_cls_z.norm(dim=1)[:, None]

                    all_cls_normalized = []
                    all_cls_z_normalized = []
                    for cls_repr in cls_outputs[1:]:
                        all_cls_normalized.append(cls_repr / cls_repr.norm(dim=1)[:, None])
                        cls_z = self.cls_contrastive_head(cls_repr)
                        cls_z_normalized = cls_z / cls_z.norm(dim=1)[:, None]
                        all_cls_z_normalized.append(cls_z_normalized)
                    # [batch*(1+pos_num+neg_num), hidden]
                    all_cls_normalized = torch.cat(all_cls_normalized, dim=0)
                    all_z_normalized = torch.cat(all_cls_z_normalized, dim=0)

                    # [batch, batch*(pos_num+neg_num)], diagonal elements should have max similarity
                    sim_mat = torch.mm(ori_cls_z_normalized, all_z_normalized.transpose(0, 1))
                    sim_mat_scaled = sim_mat / self.data_args.temperature

                    # [batch] label: (batch+i) for i in range(batch)
                    sim_label = list(range(batch_size))
                    sim_label = torch.tensor(sim_label, dtype=torch.long, device=sim_mat.device)

                    loss_fct = CrossEntropyLoss()
                    sim_loss = loss_fct(sim_mat_scaled, sim_label)

                    loss += sim_loss

                    acc = torch.sum(torch.max(sim_mat_scaled, dim=-1).indices == sim_label).item() / batch_size

                    with torch.no_grad():
                        sim_dist = F.softmax(sim_mat_scaled, dim=-1)
                        cls_sim_mat = torch.mm(all_cls_normalized, all_cls_normalized.transpose(0, 1))

                    outputs["aug{}_sim_loss".format(0)] = sim_loss
                    outputs["aug{}_sim_dist".format(0)] = sim_dist
                    outputs["aug{}_cossim".format(0)] = cls_sim_mat
                    outputs["aug{}_sim_acc".format(0)] = acc
                    outputs["acc_loss"] = acc

                elif self.data_args.cls_contrastive_type == 5:
                    # classification way, sim_dist [2*batch, 2]
                    # ori dial x, pos aug x+, shifted pos aug x- from another dial
                    # f(g(x), g(x+)) -> 1, f(g(x), g(x-)) -> 0
                    # g: linear, f: softmax(linear(a,b,|a-b|,a*b))
                    assert self.data_args.pick1utt_num == self.data_args.pos_aug_num == 1
                    assert self.data_args.neg_aug_num == 0
                    ori_cls_repr = self.cls_contrastive_head1(cls_outputs[0])
                    pos_cls_repr = self.cls_contrastive_head1(cls_outputs[1])
                    shifted_pos_cls_repr = torch.roll(pos_cls_repr, shifts=1, dims=0)
                    pos_sim_class = torch.zeros(batch_size, dtype=torch.long, device=ori_cls_repr.device)
                    pos_sim_logit = self.cls_sim_head1(ori_cls_repr, pos_cls_repr)  # [batch, 2]
                    neg_sim_class = torch.ones(batch_size, dtype=torch.long, device=ori_cls_repr.device)
                    neg_sim_logit = self.cls_sim_head1(ori_cls_repr, shifted_pos_cls_repr)  # [batch, 2]
                    sim_logit = torch.cat((pos_sim_logit, neg_sim_logit), dim=0)
                    sim_class = torch.cat((pos_sim_class, neg_sim_class), dim=0)

                    loss_fct = CrossEntropyLoss()
                    sim_loss = loss_fct(sim_logit, sim_class)
                    loss += sim_loss

                    with torch.no_grad():
                        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                        sim_dist = F.softmax(sim_logit, dim=-1)
                        acc = torch.sum(torch.max(sim_logit, dim=-1).indices == sim_class).item() / sim_class.size(0)
                        cos_sim = torch.cat((cos(cls_outputs[0], cls_outputs[1]).unsqueeze(-1),
                                             cos(cls_outputs[0], torch.roll(cls_outputs[1], shifts=1, dims=0)).unsqueeze(-1)),
                                            dim=-1)
                    outputs["aug{}_sim_loss".format(1)] = sim_loss
                    outputs["aug{}_sim_dist".format(1)] = sim_dist
                    outputs["aug{}_sim_acc".format(1)] = acc
                    outputs["aug{}_cossim".format(1)] = cos_sim
                    outputs["acc_loss_posaug"] = acc

                    ori_cls_repr = self.cls_contrastive_head2(cls_outputs[0])
                    pos_cls_repr = self.cls_contrastive_head2(cls_outputs[2])
                    shifted_pos_cls_repr = torch.roll(pos_cls_repr, shifts=-1, dims=0)
                    pos_sim_class = torch.zeros(batch_size, dtype=torch.long, device=ori_cls_repr.device)
                    pos_sim_logit = self.cls_sim_head2(ori_cls_repr, pos_cls_repr)  # [batch, 2]
                    neg_sim_class = torch.ones(batch_size, dtype=torch.long, device=ori_cls_repr.device)
                    neg_sim_logit = self.cls_sim_head2(ori_cls_repr, shifted_pos_cls_repr)  # [batch, 2]
                    sim_logit = torch.cat((pos_sim_logit, neg_sim_logit), dim=0)
                    sim_class = torch.cat((pos_sim_class, neg_sim_class), dim=0)

                    loss_fct = CrossEntropyLoss()
                    sim_loss = loss_fct(sim_logit, sim_class)
                    loss += sim_loss

                    with torch.no_grad():
                        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                        sim_dist = F.softmax(sim_logit, dim=-1)
                        acc = torch.sum(torch.max(sim_logit, dim=-1).indices == sim_class).item() / sim_class.size(0)
                        cos_sim = torch.cat((cos(cls_outputs[0], cls_outputs[2]).unsqueeze(-1),
                                             cos(cls_outputs[0],
                                                 torch.roll(cls_outputs[2], shifts=-1, dims=0)).unsqueeze(-1)),
                                            dim=-1)
                    outputs["aug{}_sim_loss".format(2)] = sim_loss
                    outputs["aug{}_sim_dist".format(2)] = sim_dist
                    outputs["aug{}_sim_acc".format(2)] = acc
                    outputs["aug{}_cossim".format(2)] = cos_sim
                    outputs["acc_loss_pick1utt"] = acc

            outputs["loss"] = loss
            return outputs

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            # turn_ids=turn_ids,
            # position_ids=position_ids,
            # role_ids=role_ids,
            token_type_ids=None,
            position_ids=None,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        # other_outputs = outputs[2:] # (hidden_states), (attentions)
        outputs = {
            # "sequence_output": sequence_output,
            # "pooled_output": pooled_output,
            # "other_outputs": other_outputs
        }

        if task == 'dapt':
            if masked_lm_labels is not None:
                prediction_scores = self.cls(sequence_output)  # (batch_size, seq_len, vocab_size)

                outputs["prediction_scores"] = prediction_scores

                loss_fct = CrossEntropyLoss()  # ignore -100 index
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                          masked_lm_labels.view(-1))
                outputs["mlm_loss"] = masked_lm_loss
            if token_tag_ids is not None and dataset in self.slot_datasets:
                tokenslot_logits = self.tokenslot_heads[self.slot_datasets.index(dataset)](sequence_output)  # (batch_size, seq_len, N)

                outputs["tokenslot_logits"] = tokenslot_logits

                loss_fct = CrossEntropyLoss()  # ignore -100 index
                tokenslot_loss = loss_fct(tokenslot_logits.view(-1, tokenslot_logits.size(-1)), token_tag_ids.view(-1))
                outputs["tokenslot_loss"] = tokenslot_loss

            if intent_tag_ids is not None and dataset in self.intent_datasets:
                intent_logits = self.intent_heads[self.intent_datasets.index(dataset)](sequence_output, sen_cls_mask)
                outputs["intent_logits"] = intent_logits

                loss_fct = BCEWithLogitsLoss()
                active_tokens = sen_cls_mask.view(-1) == 1
                active_intent_labels = intent_tag_ids.view(-1, self.intent_dim[dataset])[active_tokens]
                intent_loss = loss_fct(intent_logits, active_intent_labels)
                outputs["intent_loss"] = intent_loss

            if domain_tag_ids is not None and dataset in self.domain_datasets:
                domain_logits = self.domain_heads[self.domain_datasets.index(dataset)](sequence_output, sen_cls_mask)
                outputs["domain_logits"] = domain_logits

                loss_fct = BCEWithLogitsLoss()
                active_tokens = sen_cls_mask.view(-1) == 1
                active_domain_labels = domain_tag_ids.view(-1, self.domain_dim[dataset])[active_tokens]
                domain_loss = loss_fct(domain_logits, active_domain_labels)
                outputs["domain_loss"] = domain_loss

            if slot_tag_ids is not None and dataset in self.slot_datasets:
                slot_logits = self.slot_heads[self.slot_datasets.index(dataset)](sequence_output, sen_cls_mask)
                outputs["slot_logits"] = slot_logits

                loss_fct = BCEWithLogitsLoss()
                active_tokens = sen_cls_mask.view(-1) == 1
                active_slot_labels = slot_tag_ids.view(-1, self.slot_dim[dataset])[active_tokens]
                slot_loss = loss_fct(slot_logits, active_slot_labels)
                outputs["slot_loss"] = slot_loss

            if cls_intent_tag_ids is not None and dataset in self.intent_datasets:
                cls_intent_logits = self.intent_heads[self.intent_datasets.index(dataset)](pooled_output)
                outputs["cls_intent_logits"] = cls_intent_logits

                loss_fct = BCEWithLogitsLoss()
                cls_intent_loss = loss_fct(cls_intent_logits, cls_intent_tag_ids)
                outputs["cls_intent_loss"] = cls_intent_loss

            if cls_domain_tag_ids is not None and dataset in self.domain_datasets:
                cls_domain_logits = self.domain_heads[self.domain_datasets.index(dataset)](pooled_output)
                outputs["cls_domain_logits"] = cls_domain_logits

                loss_fct = BCEWithLogitsLoss()
                cls_domain_loss = loss_fct(cls_domain_logits, cls_domain_tag_ids)
                outputs["cls_domain_loss"] = cls_domain_loss

            if cls_slot_tag_ids is not None and dataset in self.slot_datasets:
                cls_slot_logits = self.slot_heads[self.slot_datasets.index(dataset)](pooled_output)
                outputs["cls_slot_logits"] = cls_slot_logits

                loss_fct = BCEWithLogitsLoss()
                cls_slot_loss = loss_fct(cls_slot_logits, cls_slot_tag_ids)
                outputs["cls_slot_loss"] = cls_slot_loss

            outputs["loss"] = outputs.get("mlm_loss", 0) + \
                              outputs.get("tokenslot_loss", 0) + \
                              outputs.get("intent_loss", 0) + \
                              outputs.get("domain_loss", 0) + \
                              outputs.get("slot_loss", 0) + \
                              outputs.get("cls_intent_loss", 0) + \
                              outputs.get("cls_domain_loss", 0) + \
                              outputs.get("cls_slot_loss", 0)

        elif task == 'schema_linking':
            if masked_lm_labels is not None:
                prediction_scores = self.cls(sequence_output)  # (batch_size, seq_len, vocab_size)

                outputs["prediction_scores"] = prediction_scores

                loss_fct = CrossEntropyLoss()  # ignore -100 index
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                          masked_lm_labels.view(-1))
                outputs["mlm_loss"] = masked_lm_loss
            if bio_tag_ids is not None:
                bio_logits = self.bio_head(sequence_output)  # (batch_size, seq_len, N)

                outputs["bio_logits"] = bio_logits

                loss_fct = CrossEntropyLoss()   # ignore -100 index
                bio_tag_loss = loss_fct(bio_logits.view(-1, bio_logits.size(-1)), bio_tag_ids.view(-1))
                outputs["bio_tag_loss"] = bio_tag_loss
            if intent_tag_ids is not None and sen_cls_mask is not None and self.metadataloader.processors[task].intent_dim[dataset] > 1:
                intent_logits = self.intent_head(sequence_output, sen_cls_mask)
                outputs["intent_logits"] = intent_logits

                loss_fct = BCEWithLogitsLoss()
                active_tokens = sen_cls_mask.view(-1) == 1
                active_intent_labels = intent_tag_ids.view(-1, self.metadataloader.processors[task].intent_dim[dataset])[active_tokens]
                intent_loss = loss_fct(intent_logits, active_intent_labels)
                outputs["intent_loss"] = intent_loss

                # cls_intent_loss = loss_fct(cls_intent_logits, intent_tag_ids[:, 0, :])
                # outputs["cls_intent_loss"] = cls_intent_loss

            if domain_tag_ids is not None and sen_cls_mask is not None and self.metadataloader.processors[task].domain_dim[dataset] > 1:
                # domain_logits = self.domain_heads(dataset, sequence_output,
                #                                   sen_cls_mask)  # (total utts in the batch, domain_dim)
                # cls_domain_logits = self.domain_heads(dataset, pooled_output, sen_cls_mask[:, 1])
                domain_logits = self.domain_head(sequence_output, sen_cls_mask)
                # cls_domain_logits = self.domain_head(pooled_output)
                outputs["domain_logits"] = domain_logits
                # outputs["cls_domain_logits"] = cls_domain_logits

                loss_fct = BCEWithLogitsLoss()
                active_tokens = sen_cls_mask.view(-1) == 1
                active_domain_labels = domain_tag_ids.view(-1, self.metadataloader.processors[task].domain_dim[dataset])[active_tokens]
                domain_loss = loss_fct(domain_logits, active_domain_labels)
                outputs["domain_loss"] = domain_loss

                # cls_domain_loss = loss_fct(cls_domain_logits, domain_tag_ids[:, 0, :])
                # outputs["cls_domain_loss"] = cls_domain_loss

            if slot_tag_ids is not None and sen_cls_mask is not None and self.metadataloader.processors[task].slot_dim[dataset] > 1:
                # slot_logits = self.slot_heads(dataset, sequence_output,
                #                               sen_cls_mask)  # (total utts in the batch, slot_dim)
                # cls_slot_logits = self.slot_heads(dataset, pooled_output, sen_cls_mask[:, 1])
                slot_logits = self.slot_head(sequence_output, sen_cls_mask)
                # cls_slot_logits = self.slot_head(pooled_output)
                outputs["slot_logits"] = slot_logits
                # outputs["cls_slot_logits"] = cls_slot_logits

                loss_fct = BCEWithLogitsLoss()
                active_tokens = sen_cls_mask.view(-1) == 1
                active_slot_labels = slot_tag_ids.view(-1, self.metadataloader.processors[task].slot_dim[dataset])[active_tokens]
                slot_loss = loss_fct(slot_logits, active_slot_labels)
                outputs["slot_loss"] = slot_loss

                # cls_slot_loss = loss_fct(cls_slot_logits, slot_tag_ids[:, 0, :])
                # outputs["cls_slot_loss"] = cls_slot_loss

            outputs["loss"] = outputs.get("mlm_loss", 0) + \
                              outputs.get("bio_tag_loss", 0) + \
                              outputs.get("intent_loss", 0) + \
                              outputs.get("domain_loss", 0) + \
                              outputs.get("slot_loss", 0)

        elif task == 'bio':
            if masked_lm_labels is not None:
                prediction_scores = self.cls(sequence_output)  # (batch_size, seq_len, vocab_size)

                outputs["prediction_scores"] = prediction_scores

                loss_fct = CrossEntropyLoss()  # ignore -100 index
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                          masked_lm_labels.view(-1))
                outputs["mlm_loss"] = masked_lm_loss
            if bio_tag_ids is not None:
                bio_logits = self.bio_head(sequence_output)  # (batch_size, seq_len, 3)

                outputs["bio_logits"] = bio_logits

                loss_fct = CrossEntropyLoss()   # ignore -100 index
                bio_tag_loss = loss_fct(bio_logits.view(-1, 3), bio_tag_ids.view(-1))
                outputs["bio_tag_loss"] = bio_tag_loss
            outputs["loss"] = outputs.get("mlm_loss", 0) + 0.01 * outputs.get("bio_tag_loss", 0)

        elif task == 'ssl':
            if masked_lm_labels is not None:
                prediction_scores = self.cls(sequence_output)  # (batch_size, seq_len, vocab_size)

                outputs["prediction_scores"] = prediction_scores

                loss_fct = CrossEntropyLoss()  # ignore -100 index
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                          masked_lm_labels.view(-1))
                outputs["mlm_loss"] = masked_lm_loss
            if bio_tag_ids is not None:
                bio_logits = self.bio_head(sequence_output)  # (batch_size, seq_len, 3)

                outputs["bio_logits"] = bio_logits

                loss_fct = CrossEntropyLoss()   # ignore -100 index
                bio_tag_loss = loss_fct(bio_logits.view(-1, 3), bio_tag_ids.view(-1))
                outputs["bio_tag_loss"] = bio_tag_loss
            if tf_idf is not None:
                tf_idf_logits = self.tf_idf_head(sequence_output)

                outputs["tf_idf_logits"] = tf_idf_logits

                loss_fct = MSELoss()

                active_indexes = token_mask.view(-1) == 1
                active_logits = tf_idf_logits.view(-1)[active_indexes]
                active_labels = tf_idf.view(-1)[active_indexes]
                tf_idf_loss = loss_fct(active_logits, active_labels)
                outputs["tf_idf_loss"] = tf_idf_loss

            outputs["loss"] = outputs.get("mlm_loss", 0) + outputs.get("bio_tag_loss", 0) + outputs.get("tf_idf_loss", 0)

        elif task in ['mlm', 'one_side_mlm', 'last_turn_mlm', 'span_mlm']:
            prediction_scores = self.cls(sequence_output) # (batch_size, seq_len, vocab_size)

            outputs["prediction_scores"] = prediction_scores

            # 1. If a tensor that contains the indices of masked labels is provided,
            #    the cross-entropy is the MLM cross-entropy that measures the likelihood
            #    of predictions for masked words.
            if masked_lm_labels is not None:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                outputs["loss"] = masked_lm_loss

        elif task in ["cls_mlm"]:
            prediction_scores = self.cls_mlm_head(pooled_output, sequence_output)
            outputs["prediction_scores"] = prediction_scores

            if masked_lm_labels is not None:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                outputs["loss"] = masked_lm_loss

        elif task in ["cls_pos_mlm"]:
            prediction_scores = self.cls(sequence_output)
            outputs["prediction_scores"] = prediction_scores
            if masked_lm_labels is not None:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                outputs["mlm_loss"] = masked_lm_loss

            embeds = self.bert.embeddings
            pos_embeds = embeds.position_embeddings(position_ids) + embeds.turn_embeddings(turn_ids) + embeds.role_embeddings(role_ids)

            pos_embeds = pos_embeds.detach()

            cls_output = pooled_output.unsqueeze(1).repeat(1, input_ids.size(1), 1)
            cls_output = cls_output + pos_embeds
            cls_output = self.cls(cls_output)
            outputs["cls_output"] = cls_output
            if masked_lm_labels is not None:
                cls_masked_lm_loss = loss_fct(cls_output.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                outputs["cls_mlm_loss"] = cls_masked_lm_loss

            outputs["loss"] = outputs.get("mlm_loss", 0) + outputs.get("cls_mlm_loss", 0)

        elif task == 'resp_select':
            prediction_scores = self.cls(sequence_output) # (batch_size, seq_len, vocab_size)
            outputs["prediction_scores"] = prediction_scores
            if masked_lm_labels is not None:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                outputs["mlm_loss"] = masked_lm_loss

            bs = pooled_output.size(0) // 2
            ctxt_logits, resp_logits = pooled_output[:bs], pooled_output[bs:]
            resp_scores = torch.matmul(ctxt_logits, resp_logits.t()) # bs * bs

            outputs["resp_scores"] = resp_scores
            if resp_labels is not None:
                loss_fct = CrossEntropyLoss()
                resp_loss = loss_fct(resp_scores, resp_labels)
                outputs["resp_loss"] = resp_loss

            outputs["loss"] = outputs.get("mlm_loss", 0) + outputs.get("resp_loss", 0)

        elif task == 'moco':
            outputs["cls_output"] = pooled_output

        return outputs


@add_start_docstrings("""DialogBert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class DialogBertForMaskedLM(DialogBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = DialogBertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

        self.intent_head = DialogBertTokenHead(config, 3)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        turn_ids=None,
        position_ids=None,
        role_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
                Next token prediction loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Examples::

            from transformers import BertTokenizer, BertForMaskedLM
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, masked_lm_labels=input_ids)

            loss, prediction_scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            turn_ids=turn_ids,
            position_ids=position_ids,
            role_ids=role_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)
