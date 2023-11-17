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
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.
import ast
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets

import dataset
import lang_adapter
import merging
import model as am
import trainer as ct

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.adapters import AdapterArguments, setup_adapter_training
from transformers.trainer_utils import get_last_checkpoint, RemoveColumnsCollator
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

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
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
                " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
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
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
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

    # Create validation/test split
    if data_args.task_name == "xquad":
        for lang in raw_datasets.keys():
            lang_raw_dataset = raw_datasets[lang]
            assert "test" not in lang_raw_dataset
            split = lang_raw_dataset["validation"].train_test_split(test_size=0.5, shuffle=False)
            lang_raw_dataset["validation"] = split["train"]
            lang_raw_dataset["test"] = split["test"]
            raw_datasets[lang] = lang_raw_dataset

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = am.AutoMultiAdapterModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.config.lang2id = {l: i for i, l in enumerate(languages + eval_languages)}
    model.config.id2lang = {i: l for i, l in enumerate(languages + eval_languages)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    pad_on_right = tokenizer.padding_side == "right"
    dataset_converter = dataset.DatasetFeatureConverter(
        tokenizer=tokenizer,
        padding=pad_on_right,
        max_length=max_seq_length,
        lang2idx=model.config.lang2id,
    )

    if training_args.do_train:
        if "train" not in list(raw_datasets.values())[0]:
            raise ValueError("--do_train requires a train dataset")
        raw_train_datasets = {n: d["train"] for n, d in raw_datasets.items()}
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
        # Validation Feature Creation
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

        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_datasets = dataset.tokenize_datasets(
                task_name=data_args.task_name,
                datasets=raw_test_datasets,
                converter=dataset_converter,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
                overwrite_cache=data_args.overwrite_cache,
                split="test",
            )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        answer_key = "answer" if "answer" in examples[0] else "answers"
        references = [{"id": ex["id"], "answers": ex[answer_key]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad", experiment_id=training_args.run_name)

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    model = lang_adapter.init_adapters(model, languages, data_args.adapter_path_template, "qa")
    # Initialize our Trainer
    trainer_class = ct.MultiAdapterTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        eval_examples=raw_valid_datasets if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,

    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

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
        trainer.log_metrics(metric_prefix, metrics)
        trainer.save_metrics(metric_prefix, metrics, combined=False)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Test set evaluation ***")
        trainer.eval_examples = raw_test_datasets
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
                predict_datasets = {l: d for l, d in predict_datasets.items() if l in eval_languages}
                eval_language = eval_languages[0]
                name = data_args.task_name.replace("/", "-")

                metric_prefix = f"eval_zeroshot_" \
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
            trainer.compute_metrics = None
            trainer.data_collator = RemoveColumnsCollator(
                data_collator=trainer.data_collator,
                signature_columns=list(set(predict_dataset[0].keys()) - {"example_id", "offset_mapping"}),
                model_name=trainer.model.__class__.__name__,
            )
            eval_loop = trainer.prediction_loop if trainer.args.use_legacy_prediction_loop else trainer.evaluation_loop
            eval_dataloader = trainer.get_eval_dataloader(predict_dataset)
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=None,
                metric_key_prefix="pred",
            )
            trainer.compute_metrics = compute_metrics

            suffix = predict_dataset_name.split("_")[-1]
            eval_examples = trainer.eval_examples[suffix]
            # Only the main node write the results by default
            eval_preds = trainer.post_process_function(eval_examples, predict_dataset, output.predictions)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_{data_args.metric_file_prefix}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {predict_dataset_name} *****")
                    writer.write("index\tprediction\n")
                    for index, (pred, label) in enumerate(zip(eval_preds.predictions, eval_preds.label_ids)):
                        item = metric.compute(predictions=[pred], references=[label])["f1"]
                        writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
