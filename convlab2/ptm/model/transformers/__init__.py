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

# Benchmarking
# from .benchmark_utils import (
#     Frame,
#     Memory,
#     MemoryState,
#     MemorySummary,
#     MemoryTrace,
#     UsedMemoryState,
#     bytes_to_human_readable,
#     start_memory_tracing,
#     stop_memory_tracing,
# )

# Configurations
from .configuration_dialog_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DialogBertConfig

# from .data import (
#     DataProcessor,
#     InputExample,
#     InputFeatures,
#     SingleSentenceClassificationProcessor,
#     SquadExample,
#     SquadFeatures,
#     SquadV1Processor,
#     SquadV2Processor,
#     glue_convert_examples_to_features,
#     glue_output_modes,
#     glue_processors,
#     glue_tasks_num_labels,
#     is_sklearn_available,
#     squad_convert_examples_to_features,
#     xnli_output_modes,
#     xnli_processors,
#     xnli_tasks_num_labels,
# )

# Files and general utilities
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

# Model Cards
# from .modelcard import ModelCard

# TF 2.0 <=> PyTorch conversion utilities
from .modeling_tf_pytorch_utils import (
    convert_tf_weight_name_to_pt_weight_name,
    load_pytorch_checkpoint_in_tf2_model,
    load_pytorch_model_in_tf2_model,
    load_pytorch_weights_in_tf2_model,
    load_tf2_checkpoint_in_pytorch_model,
    load_tf2_model_in_pytorch_model,
    load_tf2_weights_in_pytorch_model,
)

# Pipelines
# from .pipelines import (
#     CsvPipelineDataFormat,
#     FeatureExtractionPipeline,
#     FillMaskPipeline,
#     JsonPipelineDataFormat,
#     NerPipeline,
#     PipedPipelineDataFormat,
#     Pipeline,
#     PipelineDataFormat,
#     QuestionAnsweringPipeline,
#     SummarizationPipeline,
#     TextClassificationPipeline,
#     TextGenerationPipeline,
#     TokenClassificationPipeline,
#     TranslationPipeline,
#     pipeline,
# )

# Tokenizers

# from .tokenization_bert import BasicTokenizer, BertTokenizer, BertTokenizerFast, WordpieceTokenizer
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_dialog_bert import BasicTokenizer, DialogBertTokenizer, WordpieceTokenizer

from .trainer_utils import EvalPrediction
from .training_args import TrainingArguments


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# if is_sklearn_available():
#     from .data import glue_compute_metrics, xnli_compute_metrics


# Modeling
if is_torch_available():
    from .modeling_utils import PreTrainedModel, prune_layer, Conv1D, top_k_top_p_filtering, apply_chunking_to_forward
    # from .modeling_auto import (
    #     AutoModel,
    #     AutoModelForPreTraining,
    #     AutoModelForSequenceClassification,
    #     AutoModelForQuestionAnswering,
    #     AutoModelWithLMHead,
    #     AutoModelForTokenClassification,
    #     AutoModelForMultipleChoice,
    #     ALL_PRETRAINED_MODEL_ARCHIVE_MAP,
    #     MODEL_MAPPING,
    #     MODEL_FOR_PRETRAINING_MAPPING,
    #     MODEL_WITH_LM_HEAD_MAPPING,
    #     MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    #     MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    #     MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    #     MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    # )

    from .modeling_dialog_bert import (
        BertPreTrainedModel,
        DialogBertModel,
        BertForPreTraining,
        DialogBertForMaskedLM,
        DialogBertForNextSentencePrediction,
        # DialogBertForSequenceClassification,
        # DialogBertForMultipleChoice,
        # DialogBertForTokenClassification,
        # DialogBertForQuestionAnswering,
        load_tf_weights_in_bert,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        BertLayer,
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
    from .trainer import Trainer, set_seed, torch_distributed_zero_first, EvalPrediction
    from .data.data_collator import DefaultDataCollator, DataCollator, DataCollatorForDialogMLM
    from .data.datasets import LineByLineTextDataset


if not is_torch_available():
    logger.warning(
        "PyTorch not found."
        "Models won't be available and only tokenizers, configuration"
        "and file/data utilities can be used."
    )
