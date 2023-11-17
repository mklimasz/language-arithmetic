import math
import time
from typing import Optional, Dict, List, Union, Callable, Tuple

import datasets
import numpy as np
import torch
from torch import nn

from transformers import AdapterTrainer, is_datasets_available, PreTrainedModel, DataCollator, TrainerCallback, \
    TrainingArguments, PreTrainedTokenizerBase

from dataloader import HomogenousDataloader
from torch.utils.data import DataLoader, Dataset

from transformers.trainer_utils import seed_worker, speed_metrics, EvalPrediction, RemoveColumnsCollator


class MultiAdapterTrainer(AdapterTrainer):

    def __init__(self, model: Union[PreTrainedModel, nn.Module] = None, args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None, train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None, tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None, adapter_names: Optional[List[List[str]]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 sampling_strategy: str = "no_sampling", temperature: int = 5,
                 eval_examples=None, post_process_function=None):

        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, adapter_names, optimizers, preprocess_logits_for_metrics)
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def _get_manually_train_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        old_train_dataset = self.train_dataset
        self.train_dataset = train_dataset
        sampler = self._get_train_sampler()
        self.train_dataset = old_train_dataset
        return sampler

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        assert type(train_dataset) is dict
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        return HomogenousDataloader(
            {
                dataset_config_name: DataLoader(
                    dataset,
                    batch_size=self._train_batch_size,
                    collate_fn=data_collator,
                    drop_last=self.args.dataloader_drop_last,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                    worker_init_fn=seed_worker,
                    sampler=self._get_manually_train_sampler(dataset)
                ) for dataset_config_name, dataset in train_dataset.items()
            },
            sampling_strategy=self.sampling_strategy,
            temperature=self.temperature,
        )

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        start_time = time.time()

        if eval_dataset is None:
            eval_datasets = {f"{metric_key_prefix}_{k}": v for k, v in self.eval_dataset.items()}
        else:
            eval_datasets = {metric_key_prefix: eval_dataset}

        all_metrics = {}
        all_outputs = []
        losses = []
        accuracies = []
        for eval_dataset_name, eval_dataset in eval_datasets.items():
            if self.post_process_function is not None:
                base_collator = self.data_collator
                self.data_collator = RemoveColumnsCollator(
                    data_collator=base_collator,
                    signature_columns=list(set(eval_dataset[0].keys()) - {"example_id", "offset_mapping"}),
                    model_name=self.model.__class__.__name__,
                )

            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

            if self.post_process_function is not None:
                # QA loop
                compute_metrics = self.compute_metrics
                self.compute_metrics = None
                try:
                    output = eval_loop(
                        eval_dataloader,
                        description="Evaluation",
                        # No point gathering the predictions if there are no metrics, otherwise we defer to
                        # self.args.prediction_loss_only
                        prediction_loss_only=True if compute_metrics is None else None,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=eval_dataset_name,
                    )
                finally:
                    self.data_collator = base_collator
                    self.compute_metrics = compute_metrics
            else:
                output = eval_loop(
                    eval_dataloader,
                    description="Evaluation",
                    # No point gathering the predictions if there are no metrics, otherwise we defer to
                    # self.args.prediction_loss_only
                    prediction_loss_only=True if self.compute_metrics is None else None,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=eval_dataset_name,
                )

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )
            if self.post_process_function is not None and self.compute_metrics is not None and self.args.should_save:
                suffix = eval_dataset_name.split("_")[-1]
                eval_examples = self.eval_examples[suffix]
                # Only the main node write the results by default
                eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
                metrics = self.compute_metrics(eval_preds)

                # Prefix all keys with metric_key_prefix + '_'
                for key in list(metrics.keys()):
                    if not key.startswith(f"{metric_key_prefix}_"):
                        metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
                metrics.update(output.metrics)
            else:
                metrics = output.metrics

            self.log(metrics)

            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

            self._memory_tracker.stop_and_update_metrics(metrics)

            all_outputs.append(output)
            all_metrics.update(metrics)

            losses.extend([v for k, v in metrics.items() if "eval" in k and "loss" in k])
            accuracies.extend([v for k, v in metrics.items() if "eval" in k and "acc" in k])

        all_metrics["eval_loss"] = np.mean(losses).item()
        all_metrics["eval_acc"] = np.mean(accuracies).item()
        self.log(all_metrics)
        return all_metrics
