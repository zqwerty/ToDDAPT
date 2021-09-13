import logging
import math
import os
from typing import Optional
from dataclasses import dataclass, field
import json
import sys
sys.path.append('../../../')
from convlab2.ptm.pretraining.trainer import Trainer, set_seed, torch_distributed_zero_first, EvalPrediction



from convlab2.ptm.pretraining.model import (
    DialogBertConfig,
    # DialogBertForMaskedLM,
    DialogBertTokenizer,
    DialogBertForPretraining,
    # DataCollatorForLanguageModeling,
    # DataCollatorForDialogMLM,
    # LineByLineTextDataset,
    # PreTrainedTokenizer,
    DialMoCo,
    ModelArguments,
    DataTrainingArguments,
    TrainingArguments,
    HfArgumentParser,
)
from convlab2.ptm.pretraining.dataloader import MetaDataloader

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = training_args.cuda_visible_devices

    if os.path.exists(data_args.dataset_config):
        data_args.dataset_config = json.load(open(data_args.dataset_config))
        print("pretraining dataset config")
        print(data_args)
    else:
        raise FileNotFoundError("Can not find dataset config: "+data_args.dataset_config)

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

    # add special tokens
    special_token_dict = {'additional_special_tokens': ['[USR]', '[SYS]',
                                                        # '[INTENT]', '[DOMAIN]', '[SLOT]', '[VALUE]',
                                                        # '[STATE]', '[DIALOG_ACT]', '[NEXT_SENTENCE]'
                                                        ]}
    tokenizer.add_special_tokens(special_token_dict)
    print(len(tokenizer), tokenizer.vocab_size, len(tokenizer.added_tokens_encoder))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    task_processors = []
    if data_args.mlm:
        from convlab2.ptm.pretraining.mlm.processor import MLMProcessor
        datasets = data_args.dataset_config['mlm']
        processor = MLMProcessor(datasets, tokenizer, mlm_probability=data_args.mlm_probability)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            use_multiwoz=data_args.use_multiwoz,
                            do_train=training_args.do_train)
        task_processors.append(processor)
    if data_args.one_side_mlm:
        from convlab2.ptm.pretraining.one_side_mlm.processor import OneSideMLMProcessor
        datasets = data_args.dataset_config['one_side_mlm']
        processor = OneSideMLMProcessor(datasets, tokenizer, mlm_probability=data_args.mlm_probability,
                                        mask_user_probability=data_args.mask_user_probability,
                                        mix_batch=data_args.mixed_batch)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            use_multiwoz=data_args.use_multiwoz)
        task_processors.append(processor)
    if data_args.last_turn_mlm:
        from convlab2.ptm.pretraining.last_turn_mlm.processor import LastTurnMLMProcessor
        datasets = data_args.dataset_config['mlm']
        processor = LastTurnMLMProcessor(datasets, tokenizer, mlm_probability=data_args.last_turn_mlm_probability)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            use_multiwoz=data_args.use_multiwoz)
        task_processors.append(processor)
    if data_args.span_prediction:
        from convlab2.ptm.pretraining.span_prediction.processor import StateUpdateProcessor
        datasets = data_args.dataset_config['span_prediction']
        processor = StateUpdateProcessor(datasets, tokenizer)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers)
        task_processors.append(processor)
    if data_args.schema_linking:
        from convlab2.ptm.pretraining.schema_linking.processor import SchemaProcessor
        datasets = data_args.dataset_config['schema_linking']
        processor = SchemaProcessor(datasets, tokenizer,
                                    mlm_probability=data_args.mlm_probability, mlm_ignore_idx=-100,
                                    one_side_mask=data_args.one_side_mask,
                                    mask_user_probability=data_args.mask_user_probability,
                                    mix_batch=data_args.mixed_batch)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            use_multiwoz=data_args.use_multiwoz,
                            do_train=training_args.do_train)
        task_processors.append(processor)
    if data_args.bio:
        from convlab2.ptm.pretraining.bio.processor import BIOProcessor
        datasets = data_args.dataset_config['bio']
        processor = BIOProcessor(datasets, tokenizer, mlm_probability=data_args.bio_mlm_probability)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            use_multiwoz=data_args.use_multiwoz,
                            do_train=training_args.do_train)
        task_processors.append(processor)
    if data_args.ssl:
        from convlab2.ptm.pretraining.ssl.processor import SSLProcessor
        datasets = data_args.dataset_config['mlm']
        processor = SSLProcessor(datasets, tokenizer, mlm_probability=data_args.ssl_mlm_probability,
                                 pseudo_bio=data_args.ssl_pseudo_bio,
                                 tf_idf=data_args.ssl_tf_idf)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            do_train=training_args.do_train)
        task_processors.append(processor)
    if data_args.span_mlm:
        from convlab2.ptm.pretraining.span_mlm.processor import SpanMLMProcessor
        datasets = data_args.dataset_config['span_mlm']
        processor = SpanMLMProcessor(datasets, tokenizer,
                                     mlm_probability=data_args.mlm_probability, mlm_ignore_idx=-100,
                                     one_side_mask=data_args.one_side_mask,
                                     mask_user_probability=data_args.mask_user_probability,
                                     mix_batch=data_args.mixed_batch,
                                     span_mask_probability=data_args.span_mask_probability)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            do_train=training_args.do_train)
        task_processors.append(processor)

    if data_args.augdial:
        from convlab2.ptm.pretraining.augdial.processor import AugDialProcessor
        datasets = data_args.dataset_config['augdial']
        processor = AugDialProcessor(datasets, tokenizer,
                                     mlm_probability=data_args.augdial_mlm_probability, mlm_ignore_idx=-100,
                                     pos_aug_num=data_args.pos_aug_num,
                                     neg_aug_num=data_args.neg_aug_num,
                                     pick1utt_num=data_args.pick1utt_num,
                                     clip_ori=data_args.clip_ori,
                                     clip_aug=data_args.clip_aug,
                                     keep_value=data_args.keep_value,
                                     use_label=data_args.use_label,
                                     nolabel4aug=data_args.nolabel4aug)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            do_train=training_args.do_train)
        task_processors.append(processor)

    if data_args.augdial_ssl:
        from convlab2.ptm.pretraining.augdial_ssl.processor import AugDialProcessor
        datasets = data_args.dataset_config['augdial_ssl']
        processor = AugDialProcessor(datasets, tokenizer,
                                     mlm_probability=data_args.augdial_mlm_probability, mlm_ignore_idx=-100,
                                     pos_aug_num=data_args.pos_aug_num,
                                     neg_aug_num=data_args.neg_aug_num,
                                     pick1utt_num=data_args.pick1utt_num)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            do_train=training_args.do_train)
        task_processors.append(processor)

    if data_args.dapt:
        from convlab2.ptm.pretraining.dapt.processor import DAPTProcessor
        train_datasets = data_args.dataset_config['dapt_train']
        dev_datasets = data_args.dataset_config['dapt_dev']
        processor = DAPTProcessor(train_datasets, dev_datasets, tokenizer,
                                  mlm_probability=data_args.mlm_probability, mlm_ignore_idx=-100,
                                  train_ratio=data_args.train_ratio)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            do_train=training_args.do_train)
        task_processors.append(processor)

    if data_args.cls_mlm:
        from convlab2.ptm.pretraining.cls_mlm.processor import CLSMLMProcessor
        datasets = data_args.dataset_config["cls_mlm"]
        processor = CLSMLMProcessor(datasets, tokenizer, mlm_probability=data_args.cls_mlm_probability)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            use_multiwoz=data_args.use_multiwoz)
        task_processors.append(processor)

    if data_args.cls_pos_mlm:
        from convlab2.ptm.pretraining.cls_pos_mlm.processor import CLSPosMLMProcessor
        datasets = data_args.dataset_config["cls_pos_mlm"]
        processor = CLSPosMLMProcessor(datasets, tokenizer, mlm_probability=data_args.cls_pos_mlm_probability)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            use_multiwoz=data_args.use_multiwoz)
        task_processors.append(processor)

    if data_args.resp_select:
        from convlab2.ptm.pretraining.resp_select.processor import RSProcessor
        datasets = data_args.dataset_config["resp_select"]
        processor = RSProcessor(datasets, tokenizer, mlm_probability=data_args.resp_select_mlm_probability)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            use_multiwoz=data_args.use_multiwoz)
        task_processors.append(processor)

    if data_args.moco:
        from convlab2.ptm.pretraining.moco.processor import MOCOProcessor
        datasets = data_args.dataset_config['moco']
        processor = MOCOProcessor(datasets, tokenizer, mlm_probability=data_args.mlm_probability, keep_value=data_args.moco_keep_value)
        processor.load_data(train_batch_size=training_args.train_batch_size,
                            dev_batch_size=training_args.eval_batch_size,
                            max_length=data_args.block_size,
                            num_workers=data_args.num_workers,
                            use_multiwoz=data_args.use_multiwoz)
        task_processors.append(processor)

    task_processors = {x.task_name: x for x in task_processors}

    metadataloader = MetaDataloader(task_processors)

    if data_args.moco:
        model = DialMoCo(
            config=config,
            training_args=training_args,
            model_args=model_args,
            data_args=data_args,
            metadataloader=metadataloader
        )

    elif model_args.model_name_or_path:
        model = DialogBertForPretraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            training_args=training_args,
            model_args=model_args,
            data_args=data_args,
            metadataloader=metadataloader
        )

    else:
        logger.info("Training new model from scratch")
        model = DialogBertForPretraining(config, training_args, model_args, data_args, metadataloader)

    set_seed(training_args.seed)  # set seed for fix data feeding order, since models may be different

    model.resize_token_embeddings(len(tokenizer))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        model_args=model_args,
        data_args=data_args,
        metadataloader=metadataloader,
        prediction_loss_only=True,
        tokenizer=tokenizer
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
    if training_args.do_eval:
        set_seed(training_args.seed)  # set seed for fix data feeding order, since models may be different
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        if trainer.is_world_master():
            output_eval_file = os.path.join(training_args.output_dir, "eval_results.json")
            json.dump(eval_output, open(output_eval_file, 'w'), indent=2)


if __name__ == "__main__":
    main()
