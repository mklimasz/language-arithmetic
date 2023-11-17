import functools
from typing import Dict, List

from datasets import load_dataset, Dataset

from transformers import PreTrainedTokenizerBase

TASK2KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "xnli": ("premise", "hypothesis"),
    "Divyanshu/indicxnli": ("premise", "hypothesis"),
    "wikiann": ("tokens", "ner_tags"),
    "xquad": ("question", "context", "answers"),
    "rajpurkar/squad": ("question", "context", "answers")
}


def load_tasks(task_name: str, languages: List[str]):
    datasets = {}
    for lang in languages:
        config = lang
        if task_name == "xquad":
            config = f"{task_name}.{lang}"
        elif task_name == "rajpurkar/squad":
            config = None
        datasets[lang] = load_dataset(
            task_name,
            config
        )
    return datasets


def tokenize_datasets(
        converter: "DatasetFeatureConverter",
        task_name: str,
        datasets: Dict[str, Dataset],
        preprocessing_num_workers: int,
        overwrite_cache: bool,
        split: str) -> Dict[str, Dataset]:
    mapped_datasets = {}
    for language, dataset in datasets.items():
        mapped_datasets[language] = dataset.map(
            converter.converter_fn(task_name, language, split),
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
            remove_columns=converter.columns_to_remove[task_name],
            desc=f"Running tokenizer on {split} {task_name} dataset for language {language}",
        )
    return mapped_datasets


class DatasetFeatureConverter:

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 padding: str,
                 max_length: int,
                 lang2idx: Dict[str, int],
                 label2idx: Dict[str, int] = None,
                 ):
        self.padding = padding
        self.max_length = max_length
        self.lang2idx = lang2idx
        self.label2idx = label2idx
        self.columns_to_remove = {
            "xnli": list(TASK2KEYS["xnli"]),
            "Divyanshu/indicxnli": list(TASK2KEYS["Divyanshu/indicxnli"]),
            "wikiann": ["langs", "spans"] + list(TASK2KEYS["wikiann"]),
            "xquad": ["id"] + list(TASK2KEYS["xquad"]),
            "rajpurkar/squad": ["id", "title"] + list(TASK2KEYS["rajpurkar/squad"]),

        }
        self.task2fn = {
            "xnli": self.convert_sequence_classification,
            "Divyanshu/indicxnli": self.convert_sequence_classification,
            "wikiann": self.convert_token_classification,
            "rajpurkar/squad": self.convert_question_answering,
            "rajpurkar/squad_eval": self.convert_question_answering_eval,
            "xquad": self.convert_question_answering_eval,

        }
        self.tokenizer = tokenizer

    def converter_fn(self, task_name: str, language: str, split: str):
        fn = self.task2fn[task_name]
        if split != "train" and f"{task_name}_eval" in self.task2fn:
            fn = self.task2fn[f"{task_name}_eval"]
        return functools.partial(fn, task_name, language)

    def convert_sequence_classification(self, task_name: str, language: str, examples):
        sentence1_key, sentence2_key = TASK2KEYS[task_name]
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = self.tokenizer(*args, max_length=self.max_length, padding=self.padding, truncation=True)
        result["lang_id"] = [self.lang2idx[language] for _ in examples[sentence1_key]]
        result["label"] = [int(x) for x in examples["label"]]
        return result

    def convert_token_classification(self, task_name: str, language: str, examples):
        text_column_name, label_column_name = TASK2KEYS[task_name]
        result = self.tokenizer(
            examples[text_column_name],
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = result.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    # Exceeds mapping -> make O
                    if label[word_idx] >= len(self.label2idx):
                        label_ids.append(self.label2idx["O"])
                    else:
                        label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        result["labels"] = labels
        result["lang_id"] = [self.lang2idx[language] for _ in examples["tokens"]]
        return result

    def convert_question_answering(self, task_name: str, language: str, examples):
        question_column_name, context_column_name, answer_column_name = TASK2KEYS[task_name]
        pad_on_right = self.padding
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        tokenized_examples["lang_id"] = [self.lang2idx[language] for _ in tokenized_examples["input_ids"]]
        return tokenized_examples

    def convert_question_answering_eval(self, task_name: str, language: str, examples):
        question_column_name, context_column_name, answer_column_name = TASK2KEYS[task_name]
        pad_on_right = self.padding
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        tokenized_examples["lang_id"] = [self.lang2idx[language] for _ in tokenized_examples["input_ids"]]
        return tokenized_examples
