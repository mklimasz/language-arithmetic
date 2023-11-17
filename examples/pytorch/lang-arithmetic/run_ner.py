#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.
import ast
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from seqeval.metrics import f1_score

import dataset
import lang_adapter
import merging
import model as am
import trainer as ct
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.adapters import AdapterArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    languages: Optional[str] = field(default="en", metadata={"help": "Comma separated languages."})
    eval_languages: Optional[str] = field(default="", metadata={"help": "Comma separated eval languages."})
    lang_mapping: Optional[str] = field(default="", metadata={"help": "Dict of lang mapping"})
    custom_adapter: Optional[str] = field(
        default="",
        metadata={
            "help": (
                "Custom adapter path, adapeter should be named following pattern: lang@rest, e.g. sw@wiki"
            )
        },
    )
    adapter_path_template: Optional[str] = field(default="", metadata={"help": "Adapter path as template with lang"})
    task_adapter_path: Optional[str] = field(default="", metadata={"help": "Task adapter path"})
    metric_file_prefix: Optional[str] = field(default="", metadata={"help": "Custom metric file prefix"})
    weights: Optional[str] = field(default="", metadata={"help": "Comma separated weights."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load datasets
    languages = data_args.languages.split(",")
    eval_languages = data_args.eval_languages.split(",")
    assert languages
    dataset_languages = (languages if training_args.do_train else [] +
                         eval_languages if (training_args.do_eval or training_args.do_predict) else [])
    raw_datasets = dataset.load_tasks(
        task_name=data_args.task_name,
        languages=dataset_languages
    )

    assert data_args.task_name in {"wikiann", "masakhaner"},\
        "Current implementation handle only wikiann and masakhaner."
    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-DATE', 'I-DATE']
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"bloom", "gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = am.AutoMultiAdapterModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )


    model.config.lang2id = {l: i for i, l in enumerate(languages + eval_languages)}
    model.config.id2lang = {i: l for i, l in enumerate(languages + eval_languages)}
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False
    dataset_converter = dataset.DatasetFeatureConverter(
        tokenizer=tokenizer,
        padding=padding,
        max_length=data_args.max_seq_length,
        lang2idx=model.config.lang2id,
        label2idx=label_to_id,
    )

    if training_args.do_train:
        if "train" not in list(raw_datasets.values())[0]:
            raise ValueError("--do_train requires a train dataset")
        raw_train_datasets = {n: d["train"] for n, d in raw_datasets.items()}
        if data_args.max_train_samples is not None:
            subset = {}
            for n, d in raw_train_datasets.items():
                subset[n] = d.select(range(data_args.max_train_samples))
            raw_train_datasets = subset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_datasets = dataset.tokenize_datasets(
                task_name=data_args.task_name,
                datasets=raw_train_datasets,
                converter=dataset_converter,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
                overwrite_cache=data_args.overwrite_cache,
                split="train",
            )

    if training_args.do_eval:
        if "validation" not in list(raw_datasets.values())[0]:
            raise ValueError("--do_eval requires a validation dataset")
        raw_valid_datasets = {n: d["validation"] for n, d in raw_datasets.items()}
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_datasets = dataset.tokenize_datasets(
                task_name=data_args.task_name,
                datasets=raw_valid_datasets,
                converter=dataset_converter,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
                overwrite_cache=data_args.overwrite_cache,
                split="validation",
            )


    if training_args.do_predict:
        if "test" not in list(raw_datasets.values())[0]:
            raise ValueError("--do_predict requires a test dataset")
        raw_test_datasets = {n: d["test"] for n, d in raw_datasets.items()}
        with training_args.main_process_first(desc="test dataset map pre-processing"):
            predict_datasets = dataset.tokenize_datasets(
                task_name=data_args.task_name,
                datasets=raw_test_datasets,
                converter=dataset_converter,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
                overwrite_cache=data_args.overwrite_cache,
                split="validation",
            )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metrics
    metric = evaluate.load("seqeval", experiment_id=training_args.run_name)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Setup adapters
    model = lang_adapter.init_adapters(model, languages, data_args.adapter_path_template, "ner")
    # Initialize our Trainer
    trainer_class = ct.MultiAdapterTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.model.train_adapter(["ner"])
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        train_len = sum(len(x) for x in train_datasets.values())
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else train_len
        )
        metrics["train_samples"] = min(max_train_samples, train_len)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:

        logger.info("*** Evaluate ***")
        for lang in languages:
            model = lang_adapter.load_adapter(model, data_args.adapter_path_template.format(lang=lang),
                                              overwrite_name=lang)
        model.load_adapter(data_args.task_adapter_path)

        if eval_languages:
            eval_datasets = {l: d for l, d in eval_datasets.items() if l in eval_languages}

            if data_args.lang_mapping:

                if data_args.custom_adapter:
                    suffix = data_args.custom_adapter.split('/')
                    suffix = suffix[-1] if suffix[-1] else suffix[-2]
                    lang = suffix.split("@")[0]
                    model = lang_adapter.load_adapter(model, data_args.custom_adapter, lang)
                    logger.info(f"Loaded a custom adapter as {lang}")

                w1, w2 = [float(x) for x in data_args.weights.split(",")]
                name = data_args.task_name.replace("/", "-")

                metric_prefix = f"eval_zeroshot_interpolation_{name}_{w1}_{w2}"
                if data_args.metric_file_prefix:
                    metric_prefix = f"{data_args.metric_file_prefix}_{name}_{w1}_{w2}"
                lang_mapping = ast.literal_eval(data_args.lang_mapping)
                for at, (a1, a2) in lang_mapping.items():
                    try:
                        model.add_adapter(
                            at, config=model.config.adapters.get(a2)
                        )
                    except ValueError:
                        logger.warning(f"Adapter exists, skip adding for {at}")

                if "roberta" in model_args.model_name_or_path:
                    layers = model.roberta.encoder.layer
                    inv_adapters = model.roberta.invertible_adapters
                else:
                    layers = model.bert.encoder.layer
                    inv_adapters = model.bert.invertible_adapters

                for layer in layers:
                    for at, (a1, a2) in lang_mapping.items():
                        print(f"Merging regular adapter {at} = {w1}*{a1} + {w2}*{a2}")
                        merging.merge_adapters(
                            w1=w1, w2=w2,
                            a1=layer.output.adapters[a1].cpu(),
                            a2=layer.output.adapters[a2].cpu(),
                            at=layer.output.adapters[at].cpu()
                        )

                for at, (a1, a2) in lang_mapping.items():
                    print(f"Merging inv adapter {at} = {w1}*{a1} + {w2}*{a2}")
                    merging.merge_inv_adapters(w1, w2,
                                               inv_adapters[a1].F.cpu(),
                                               inv_adapters[a2].F.cpu(),
                                               inv_adapters[at].F.cpu())
                    merging.merge_inv_adapters(w1, w2,
                                               inv_adapters[a1].G.cpu(),
                                               inv_adapters[a2].G.cpu(),
                                               inv_adapters[at].G.cpu())
            else:
                assert data_args.custom_adapter
                assert len(eval_languages) == 1, "Evaluation using custom adapter can be done only if" \
                                                 "there is exactly one eval lang."
                eval_datasets = {l: d for l, d in eval_datasets.items() if l in eval_languages}
                eval_language = eval_languages[0]
                name = data_args.task_name.replace("/", "-")

                metric_prefix = f"eval_zeroshot_" \
                                f"{name}_{eval_language}_{data_args.custom_adapter.split('/')[-1]}"
                if data_args.metric_file_prefix:
                    metric_prefix = f"{data_args.metric_file_prefix}_" \
                                    f"{name}_{eval_language}_{data_args.custom_adapter.split('/')[-1]}"

                model = lang_adapter.load_adapter(model, data_args.custom_adapter, eval_language)

        trainer._move_model_to_device(model, trainer.args.device)

        trainer.eval_dataset = eval_datasets
        metrics = trainer.evaluate()
        print(metrics)
        trainer.log_metrics(metric_prefix, metrics)
        trainer.save_metrics(metric_prefix, metrics, combined=False)

    # Predict
    if training_args.do_predict:
        logger.info("*** Test set evaluation ***")
        for lang in languages:
            model = lang_adapter.load_adapter(model, data_args.adapter_path_template.format(lang=lang),
                                              overwrite_name=lang)
        model.load_adapter(data_args.task_adapter_path)

        if eval_languages:
            predict_datasets = {l: d for l, d in predict_datasets.items() if l in eval_languages}

            if data_args.lang_mapping:

                if data_args.custom_adapter:
                    suffix = data_args.custom_adapter.split('/')
                    suffix = suffix[-1] if suffix[-1] else suffix[-2]
                    lang = suffix.split("@")[0]
                    model = lang_adapter.load_adapter(model, data_args.custom_adapter, lang)
                    logger.info(f"Loaded a custom adapter as {lang}")

                w1, w2 = [float(x) for x in data_args.weights.split(",")]
                name = data_args.task_name.replace("/", "-")

                metric_prefix = f"test_zeroshot_interpolation_{name}_{w1}_{w2}"
                if data_args.metric_file_prefix:
                    metric_prefix = f"{data_args.metric_file_prefix}_{name}_{w1}_{w2}"
                lang_mapping = ast.literal_eval(data_args.lang_mapping)
                for at, (a1, a2) in lang_mapping.items():
                    try:
                        model.add_adapter(
                            at, config=model.config.adapters.get(a2)
                        )
                    except ValueError:
                        logger.warning(f"Adapter exists, skip adding for {at}")

                if "roberta" in model_args.model_name_or_path:
                    layers = model.roberta.encoder.layer
                    inv_adapters = model.roberta.invertible_adapters
                else:
                    layers = model.bert.encoder.layer
                    inv_adapters = model.bert.invertible_adapters

                for layer in layers:
                    for at, (a1, a2) in lang_mapping.items():
                        print(f"Merging regular adapter {at} = {w1}*{a1} + {w2}*{a2}")
                        merging.merge_adapters(
                            w1=w1, w2=w2,
                            a1=layer.output.adapters[a1].cpu(),
                            a2=layer.output.adapters[a2].cpu(),
                            at=layer.output.adapters[at].cpu()
                        )

                for at, (a1, a2) in lang_mapping.items():
                    print(f"Merging inv adapter {at} = {w1}*{a1} + {w2}*{a2}")
                    merging.merge_inv_adapters(w1, w2,
                                               inv_adapters[a1].F.cpu(),
                                               inv_adapters[a2].F.cpu(),
                                               inv_adapters[at].F.cpu())
                    merging.merge_inv_adapters(w1, w2,
                                               inv_adapters[a1].G.cpu(),
                                               inv_adapters[a2].G.cpu(),
                                               inv_adapters[at].G.cpu())
            else:
                assert data_args.custom_adapter
                assert len(eval_languages) == 1, "Evaluation using custom adapter can be done only if" \
                                                 "there is exactly one eval lang."
                predict_datasets = {l: d for l, d in predict_datasets.items() if l in eval_languages}
                eval_language = eval_languages[0]
                name = data_args.task_name.replace("/", "-")

                metric_prefix = f"test_zeroshot_" \
                                f"{name}_{eval_language}_{data_args.custom_adapter.split('/')[-1]}"
                if data_args.metric_file_prefix:
                    metric_prefix = f"{data_args.metric_file_prefix}_" \
                                    f"{name}_{eval_language}_{data_args.custom_adapter.split('/')[-1]}"

                model = lang_adapter.load_adapter(model, data_args.custom_adapter, eval_language)

        trainer._move_model_to_device(model, trainer.args.device)

        trainer.eval_dataset = predict_datasets
        metrics = trainer.evaluate()

        if data_args.weights:
            w1, w2 = [float(x) for x in data_args.weights.split(",")]
            metrics["eval_w1"] = w1
            metrics["eval_w2"] = w2

        trainer.log_metrics(metric_prefix, metrics)
        trainer.save_metrics(metric_prefix, metrics, combined=False)

        for predict_dataset_name, predict_dataset in predict_datasets.items():
            labels = predict_dataset["labels"]
            predict_dataset = predict_dataset.remove_columns("labels")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.argmax(predictions, axis=2)
            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            output_predict_file = os.path.join(training_args.output_dir, f"predict_{data_args.metric_file_prefix}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {predict_dataset_name} *****")
                    writer.write("index\tprediction\n")
                    for index, (tp, tl) in enumerate(zip(true_predictions, true_labels)):
                        item = f1_score([tl], [tp])
                        writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
