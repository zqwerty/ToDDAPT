# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "2.9.1"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import logging

from .configuration_dialog_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DialogBertConfig
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_tf_available,
    is_torch_available,
)
from .hf_argparser import HfArgumentParser
from .modeling_tf_pytorch_utils import (
    convert_tf_weight_name_to_pt_weight_name,
    load_pytorch_checkpoint_in_tf2_model,
    load_pytorch_model_in_tf2_model,
    load_pytorch_weights_in_tf2_model,
    load_tf2_checkpoint_in_pytorch_model,
    load_tf2_model_in_pytorch_model,
    load_tf2_weights_in_pytorch_model,
)
from .tokenization_dialog_bert import BasicTokenizer, DialogBertTokenizer, WordpieceTokenizer
from .tokenization_bert import BertTokenizer
from .tokenization_utils import PreTrainedTokenizer
from .trainer_utils import EvalPrediction
from .training_args import ModelArguments, DataTrainingArguments, TrainingArguments


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# if is_sklearn_available():
#     from .data import glue_compute_metrics, xnli_compute_metrics


# Modeling
if is_torch_available():
    from .modeling_utils import PreTrainedModel, prune_layer, Conv1D, top_k_top_p_filtering, apply_chunking_to_forward

    from .modeling_dialog_bert import (
        DialogBertPreTrainedModel,
        DialogBertModel,
        DialogBertForPretraining,
        DialogBertForMaskedLM,
        load_tf_weights_in_bert,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        BertLayer,
    )

    from .DialMoCo import (
        DialMoCo
    )

    # Optimization
    from .optimization import (
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    # Trainer

if not is_torch_available():
    logger.warning(
        "PyTorch not found."
        "Models won't be available and only tokenizers, configuration"
        "and file/data utilities can be used."
    )
