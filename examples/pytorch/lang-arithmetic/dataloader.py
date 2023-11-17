import math
from itertools import chain, tee

import numpy as np


class HomogenousDataloader:
    """
    Data loader that combines and samples from multiple data loaders.
    """

    def __init__(self, dataloader_dict, evaluation=False, sampling_strategy="no_sampling", temperature=5):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(len(dataloader.dataset) for dataloader in self.dataloader_dict.values())
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature
        self.evaluation = evaluation

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        if self.evaluation:
            if self.sampling_strategy != 'no_sampling':
                print(f"Eval must be in 'no_sampling' mode but had {self.sampling_strategy}, "
                      f"changing to 'no_sampling'.")
            self.sampling_strategy = 'no_sampling'

        if self.sampling_strategy == 'temperature':
            sampled_batch_numbers = self.temperature_sampling(self.num_batches_dict)
        elif self.sampling_strategy == 'no_sampling':
            sampled_batch_numbers = self.no_sampling(self.num_batches_dict)
        else:
            raise ValueError(f"Unknown sampling strategy {self.sampling_strategy}")
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * sampled_batch_numbers[task_name]

        task_choice_list = np.array(task_choice_list)
        if not self.evaluation:
            np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(chain(*tee(dataloader,
                                       math.ceil(sampled_batch_numbers[task_name] / self.num_batches_dict[task_name]))))
            if (self.sampling_strategy == 'temperature' and
                sampled_batch_numbers[task_name] > self.num_batches_dict[task_name])
            else iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])

    def temperature_sampling(self, num_batches_dict):
        total_size = sum(num_batches_dict.values())
        sampling_ratios = {task_name: (size / total_size) ** (1.0 / self.temperature)
                           for task_name, size in num_batches_dict.items()}
        sampling_ratios = {task_name: sampling_ratios[task_name] / sum(sampling_ratios.values())
                           for task_name in num_batches_dict.keys()}
        sampled_numbers = {task_name: int(sampling_ratios[task_name] * sum(num_batches_dict.values()))
                           for task_name in num_batches_dict.keys()}
        return sampled_numbers

    def no_sampling(self, num_batches_dict):
        return num_batches_dict
