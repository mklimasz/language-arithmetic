import functools

from transformers.adapters import Stack

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering
)


class AutoMultiAdapterModelForSequenceClassification(AutoModelForSequenceClassification):

    @classmethod
    def forward(cls, model, fwd, **kwargs):
        adapter = kwargs.pop("lang_id")
        adapter_name = model.config.id2lang[adapter[0].item()]
        model.set_active_adapters(Stack(adapter_name, "nli"))
        return fwd(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super(AutoMultiAdapterModelForSequenceClassification, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs)
        model.forward = functools.partial(cls.forward, model, model.forward)
        return model


class AutoMultiAdapterModelForTokenClassification(AutoModelForTokenClassification):

    @classmethod
    def forward(cls, model, fwd, **kwargs):
        adapter = kwargs.pop("lang_id")
        adapter_name = model.config.id2lang[adapter[0].item()]
        model.set_active_adapters(Stack(adapter_name, "ner"))
        return fwd(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super(AutoMultiAdapterModelForTokenClassification, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs)
        model.forward = functools.partial(cls.forward, model, model.forward)
        return model


class AutoMultiAdapterModelForQuestionAnswering(AutoModelForQuestionAnswering):

    @classmethod
    def forward(cls, model, fwd, **kwargs):
        adapter = kwargs.pop("lang_id")
        adapter_name = model.config.id2lang[adapter[0].item()]
        model.set_active_adapters(Stack(adapter_name, "qa"))
        return fwd(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super(AutoMultiAdapterModelForQuestionAnswering, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs)
        model.forward = functools.partial(cls.forward, model, model.forward)
        return model
