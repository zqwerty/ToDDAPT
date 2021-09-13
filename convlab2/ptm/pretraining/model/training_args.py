import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .file_utils import cached_property, is_torch_available, torch_required


if is_torch_available():
    import torch


try:
    import torch_xla.core.xla_model as xm

    _has_tpu = True
except ImportError:
    _has_tpu = False


@torch_required
def is_tpu_available():
    return _has_tpu


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    moco_K: Optional[int] = field(
        default=4096, metadata={"help": "Queue size of MOCO pre-training"}
    )
    moco_m: Optional[float] = field(
        default=0.999, metadata={"help": "MOCO momentum"}
    )
    moco_T: Optional[float] = field(
        default=0.07, metadata={"help": "MOCO temperature"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    one_side_mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss, only mask one speaker utts in a batch"}
    )
    mixed_batch: bool = field(
        default=False, metadata={"help": "mix batch training, usr/sys masked"}
    )
    mask_user_probability: float = field(
        default=0.5, metadata={"help": "mask usr|sys utts with prob x|1-x"}
    )

    span_mlm: bool = field(
        default=False, metadata={"help": "Train with span mlm loss"}
    )
    one_side_mask: bool = field(
        default=False, metadata={"help": "Train with one side mask"}
    )
    span_mask_probability: float = field(
        default=0.5, metadata={"help": "Ratio of spans to mask for schema linking loss"}
    )

    last_turn_mlm: bool = field(
        default=False,
        metadata={"help": "Train with masked-language modeling loss, only mask last turn"}
    )
    last_turn_mlm_probability: float = field(
        default=0.3, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    bio: bool = field(
        default=False, metadata={"help": "Train with BIO tagging."}
    )
    bio_mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    ssl: bool = field(
        default=False, metadata={"help": "Train with Self-Supervised Learning."}
    )
    ssl_mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    ssl_pseudo_bio: bool = field(
        default=False, metadata={"help": "Whether use pseudo bio tag"}
    )
    ssl_tf_idf: bool = field(
        default=False, metadata={"help": "tf_idf regression"}
    )

    span_prediction: bool = field(
        default=False, metadata={"help": "Train with span prediction loss"}
    )

    schema_linking: bool = field(
        default=False, metadata={"help": "Train with schema linking"}
    )

    augdial_ssl: bool = field(
        default=False, metadata={"help": "Train with augmented dial"}
    )
    augdial: bool = field(
        default=False, metadata={"help": "Train with augmented dial"}
    )
    augdial_mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    pos_aug_num: int = field(
        default=1, metadata={"help": "number of positive augmented samples per dial"}
    )
    neg_aug_num: int = field(
        default=1, metadata={"help": "number of negative augmented samples per dial"}
    )
    pick1utt_num: int = field(
        default=1, metadata={"help": "number of negative augmented samples per dial"}
    )
    clip_aug: bool = field(
        default=False, metadata={"help": "clip aug_dial in data augmentation"}
    )
    clip_ori: bool = field(
        default=False, metadata={"help": "clip ori_dial in data augmentation"}
    )
    keep_value: bool = field(
        default=False, metadata={"help": "Value keeping data augmentation"}
    )
    use_label: bool = field(
        default=False, metadata={"help": "whether use label for supervised learning"}
    )
    cls_contrastive: bool = field(
        default=False, metadata={"help": "train CLS token with contrastive learning"}
    )
    temperature: float = field(
        default=0.1, metadata={"help": "temperature for similarity softmax"}
    )
    nograd4aug: bool = field(
        default=False, metadata={"help": "whether train the model on augmented samples"}
    )
    nolabel4aug: bool = field(
        default=False, metadata={"help": "whether use label for augmented samples"}
    )
    cls_contrastive_type: int = field(
        default=0, metadata={
            "help": "contrastive type"}
    )

    dapt: bool = field(
        default=False, metadata={"help": "Domain adaptive training"}
    )
    train_ratio: float = field(
        default=1.0, metadata={"help": "ratio of training data"}
    )
    biotagging: bool = field(
        default=False, metadata={"help": "bio-slot tagging"}
    )
    sencls: bool = field(
        default=False, metadata={"help": "sentence classification"}
    )
    dialcls: bool = field(
        default=False, metadata={"help": "dialogue classification"}
    )


    neg_samples: int = field(
        default=3, metadata={
            "help": "number of negative samples for schema linking task"}
    )

    cls_mlm: bool = field(
        default=False, metadata={
            "help": "Train with cls mlm"
        }
    )
    cls_mlm_probability: float = field(
        default=0.15, metadata={
            "help": "Ratio of tokens to mask for cls mlm"
        }
    )

    cls_pos_mlm: bool = field(
        default=False, metadata={
            "help": "Train with cls mlm, add position embeddings to cls token output"
        }
    )
    cls_pos_mlm_probability: float = field(
        default=0.15, metadata={
            "help": "Ratio of tokens to mask for cls pos mlm"
        }
    )

    resp_select: bool = field(
        default=False, metadata={
            "help": "Train with response selection"
        }
    )
    
    resp_select_mlm_probability: float = field(
        default=0.15, metadata={
            "help": "Ratio of tokens to mask for response selection"
        }
    )

    moco: bool = field(
        default=False, metadata={
            "help": "For moco pre-training"
        }
    )

    moco_keep_value: bool = field(
        default=False, metadata={
            "help": "Keep the value unchanged when augmenting"
        }
    )

    span_mask_probability: float = field(
        default=0.5, metadata={"help": "Ratio of spans to mask for schema linking loss"}
    )

    dataset_config: str = field(
        default='dataset_config.json', metadata={"help": "Datasets for each pretraining task"}
    )

    use_multiwoz: bool = field(
        default=False, metadata={"help": "Use multiwoz dataset to train"}
    )

    num_workers: int = field(
        default=0, metadata={
            "help": "how many subprocesses to use for data loading. "
            "0 means that the data will be loaded in the main process. default: 0"}
    )

    block_size: int = field(
        default=256,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    model_des: str = field(
        default="",
        metadata={"help": "description of current model"}
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluate_during_training: bool = field(
        default=False, metadata={"help": "Run evaluation during training at each logging step."},
    )

    per_gpu_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for training."})
    per_gpu_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    cuda_visible_devices: Optional[str] = field(default=None, metadata={"help": "CUDA_VISIBLE_DEVICES"})

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    # num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=120000,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=10000, metadata={"help": "Linear warmup over warmup_steps."})

    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir. default: $output_dir/log"})
    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    tpu_metrics_debug: bool = field(default=False, metadata={"help": "TPU: Whether to print debug metrics"})

    sche_type: str = field(
        default="linear",
        metadata={
            "help": (
                "How to schedule the learning rate"
            )
        }
    )

    @property
    def train_batch_size(self) -> int:
        return self.per_gpu_train_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        return self.per_gpu_eval_batch_size * max(1, self.n_gpu)

    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif is_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1
        return device, n_gpu

    @property
    @torch_required
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    @torch_required
    def n_gpu(self):
        return self._setup_devices[1]

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}
