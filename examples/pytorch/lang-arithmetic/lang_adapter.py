from typing import Optional

from transformers import AdapterConfig


def init_empty_adapters(model, languages, adapter_args):
    for lang in languages:
        model.add_adapter(lang, config=adapter_args.adapter_config)
    return model


def init_adapters(model, languages, adapter_path_template: str, task_adapter_name: str):
    config = AdapterConfig.load("pfeiffer", non_linearity="gelu_new", reduction_factor=16, inv_adapter=None)
    model.add_adapter(task_adapter_name, config=config, set_active=True)
    for lang in languages:
        model = load_adapter(model, adapter_path_template.format(lang=lang), lang)
    model.train_adapter(task_adapter_name)
    return model


def load_adapter(model, adapter_name, overwrite_name: Optional[str] = None):
    model.load_adapter(adapter_name, set_active=False, load_as=overwrite_name)
    return model
