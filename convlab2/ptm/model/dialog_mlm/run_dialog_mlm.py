import logging
import math
import os
from typing import Optional
from dataclasses import dataclass, field

import sys
sys.path.append('/home/libing/Convlab2-Pretraining')

from convlab2.ptm.model.transformers import Trainer

from convlab2.ptm.model.transformers import (
    DialogBertConfig,
    DialogBertForMaskedLM,
    DialogBertTokenizer,
    DataCollatorForDialogMLM,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
)

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
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: ['bert']. not used in dialog bert mlm"}
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    # history_length: int = field(
    #     default=0,
    #     metadata={'help': 'dialog history length, 1 means using turn_t and turn_t-1 for training'}
    # )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False, display=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        # for statistics
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, display=display)
        # return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=None, history_length=history_length, display=display)

    else:
        raise ValueError


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)


    if model_args.config_name:
        config = DialogBertConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = DialogBertConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = DialogBertConfig()

    if model_args.tokenizer_name:
        tokenizer = DialogBertTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = DialogBertTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )


    if model_args.model_name_or_path:
        model = DialogBertForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    else:
        logger.info("Training new model from scratch")
        model = DialogBertForMaskedLM(config)

    # add special tokens
    special_token_dict = {'additional_special_tokens': ['[USR]', '[SYS]', '[DOMAIN]', '[STATE]', '[DIALOG_ACT]',
                                                        '[NEXT_SENTENCE]']}
    tokenizer.add_special_tokens(special_token_dict)

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    import json

    if not os.path.isdir(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    dataset_pkl_path_train = os.path.join(training_args.output_dir, 'train.pkl')
    if not os.path.exists(dataset_pkl_path_train):
        train_dataset = get_dataset(data_args, tokenizer=tokenizer, display=True) if training_args.do_train else None
        if train_dataset is not None:
            json.dump([train_dataset.examples, train_dataset.num_example_turns], open(dataset_pkl_path_train, 'w'))
    else:
        train_examples, num_example_turns = json.load(open(dataset_pkl_path_train, 'r'))
        train_dataset = LineByLineTextDataset()
        train_dataset.examples = train_examples
        train_dataset.num_example_turns = num_example_turns

    dataset_pkl_path_dev = os.path.join(training_args.output_dir, 'dev.pkl')
    if not os.path.exists(dataset_pkl_path_dev):
        eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
        if eval_dataset is not None:
            json.dump(eval_dataset.examples, open(dataset_pkl_path_dev, 'w'))
    else:
        eval_examples = json.load(open(dataset_pkl_path_dev, 'r'))
        eval_dataset = LineByLineTextDataset()
        eval_dataset.examples = eval_examples

    data_collator = DataCollatorForDialogMLM(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    )

    num_train_turns = train_dataset.num_example_turns

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
        buckets_id=num_train_turns
    )


    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


if __name__ == "__main__":
    main()