# Ties-merging part based on:
# https://github.com/prateeky2806/ties-merging/blob/main/src/ties_minimal.ipynb
import copy
from collections import OrderedDict

import torch

from transformers.activations import get_activation
import torch.nn as nn


# TODO refactor, possible the worst code ive even written
def redefine_adapters(model: nn.Module, lang1: str, lang2: str, tgt: str):
    model1 = copy.deepcopy(model.cpu())
    model2 = copy.deepcopy(model.cpu())

    layers1 = model1.roberta.encoder.layer
    layers2 = model2.roberta.encoder.layer
    for l1, l2 in zip(layers1, layers2):
        adapter1 = l1.output.adapters[lang1]
        adapter2 = l2.output.adapters[lang2]
        l1.output.adapters = nn.ModuleDict()
        l2.output.adapters = nn.ModuleDict()
        l1.output.adapters[tgt] = adapter1
        l2.output.adapters[tgt] = adapter2

    inv_adapter1 = model1.roberta.invertible_adapters[lang1]
    inv_adapter2 = model2.roberta.invertible_adapters[lang2]
    model1.roberta.invertible_adapters = nn.ModuleDict()
    model2.roberta.invertible_adapters = nn.ModuleDict()
    model1.roberta.invertible_adapters[tgt] = inv_adapter1
    model2.roberta.invertible_adapters[tgt] = inv_adapter2
    return model1, model2


def merge_ties(base_model: nn.Module, model: nn.Module,
               lang1: str, lang2: str, tgt_lang: str, lambda_: float, K: int):

    model1, model2 = redefine_adapters(model, lang1, lang2, tgt_lang)
    ft_checks = [model1.state_dict(), model2.state_dict()]
    # TODO this must be model with initialized adapters
    ptm_check = base_model.state_dict()

    remove_keys = [n for n, p in ptm_check.items() if f"adapters.{tgt_lang}" not in n]

    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    print(f"Flattening out base model")
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

    # Creating Task vectors
    tv_flat_checks = flat_ft - flat_ptm

    # check if the vectorized state dicts can be converted back to the original state dicts
    # covnert back the flat task vectors to state dict and see if the original and converted sd's are equal
    # assert check_state_dicts_equal(
    #     vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check
    # )
    # assert all(
    #     [
    #         check_state_dicts_equal(
    #             vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]
    #         )
    #         for i in range(len(ft_checks))
    #     ]
    # )

    merge_func = "dis-mean"

    # return merged flat task vector
    merged_tv = ties_merging(
        tv_flat_checks,
        reset_thresh=K,
        merge_func=merge_func,
    )

    # add back the PTM to the flat merged task vector
    merged_check = flat_ptm + lambda_ * merged_tv
    # merged_check = lamda * merged_tv

    # convert the flat merged checkpoint to a state dict
    merged_state_dict = vector_to_state_dict(
        merged_check, model1.state_dict(), remove_keys=remove_keys
    )

    model.load_state_dict(merged_state_dict, strict=False)
    return model


def merge_default(layers_collection, invertible_adapters, lang_mapping, weights):
    w1, w2 = weights
    for layer in layers_collection:
        for at, (a1, a2) in lang_mapping.items():
            print(f"Merging regular adapter {at} = {w1}*{a1} + {w2}*{a2}")
            merge_adapters(
                w1=w1, w2=w2,
                a1=layer.output.adapters[a1].cpu(),
                a2=layer.output.adapters[a2].cpu(),
                at=layer.output.adapters[at].cpu()
            )

    for at, (a1, a2) in lang_mapping.items():
        print(f"Merging inv adapter {at} = {w1}*{a1} + {w2}*{a2}")
        merge_inv_adapters(w1, w2,
                           invertible_adapters[a1].F.cpu(),
                           invertible_adapters[a2].F.cpu(),
                           invertible_adapters[at].F.cpu())
        merge_inv_adapters(w1, w2,
                           invertible_adapters[a1].G.cpu(),
                           invertible_adapters[a2].G.cpu(),
                           invertible_adapters[at].G.cpu())


def merge_adapters(w1, w2, a1, a2, at):
    assert type(a1.non_linearity.f) == type(a2.non_linearity.f) == type(get_activation("gelu_new")), \
        (type(a1.non_linearity.f), type(a2.non_linearity.f), type(get_activation("gelu_new")))
    at.adapter_up.load_state_dict({
        "weight": w1 * a1.adapter_up.weight + w2 * a2.adapter_up.weight,
        "bias": w1 * a1.adapter_up.bias + w2 * a2.adapter_up.bias
    })
    at.adapter_down[0].load_state_dict({
        "weight": w1 * a1.adapter_down[0].weight + w2 * a2.adapter_down[0].weight,
        "bias": w1 * a1.adapter_down[0].bias + w2 * a2.adapter_down[0].bias
    })


def merge_adapters_three(w1, w2, w3, a1, a2, a3, at):
    assert type(a1.non_linearity.f) == type(a2.non_linearity.f) == type(a3.non_linearity.f) == type(get_activation("gelu_new")), \
        (type(a1.non_linearity.f), type(a2.non_linearity.f), type(a3.non_linearity.f), type(get_activation("gelu_new")))
    at.adapter_up.load_state_dict({
        "weight": w1 * a1.adapter_up.weight + w2 * a2.adapter_up.weight + w3 * a3.adapter_up.weight,
        "bias": w1 * a1.adapter_up.bias + w2 * a2.adapter_up.bias + w3 * a3.adapter_up.bias
    })
    at.adapter_down[0].load_state_dict({
        "weight": w1 * a1.adapter_down[0].weight + w2 * a2.adapter_down[0].weight + w3 * a3.adapter_down[0].weight,
        "bias": w1 * a1.adapter_down[0].bias + w2 * a2.adapter_down[0].bias + w3 * a3.adapter_down[0].bias
    })


def merge_inv_adapters(w1, w2, a1, a2, at):
    at[0].load_state_dict({
        "weight": w1 * a1[0].weight + w2 * a2[0].weight,
        "bias": w1 * a1[0].bias + w2 * a2[0].bias
    })
    at[2].load_state_dict({
        "weight": w1 * a1[2].weight + w2 * a2[2].weight,
        "bias": w1 * a1[2].bias + w2 * a2[2].bias
    })


def merge_inv_adapters_three(w1, w2, w3, a1, a2, a3, at):
    at[0].load_state_dict({
        "weight": w1 * a1[0].weight + w2 * a2[0].weight + w3 * a3[0].weight,
        "bias": w1 * a1[0].bias + w2 * a2[0].bias + w3 * a3[0].bias
    })
    at[2].load_state_dict({
        "weight": w1 * a1[2].weight + w2 * a2[2].weight + w3 * a3[2].weight,
        "bias": w1 * a1[2].bias + w2 * a2[2].bias + w3 * a3[2].bias
    })


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )

def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(Tensor):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(Tensor, merge_func, sign_to_mult):

    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def ties_merging(
        flat_task_checks,
        reset_thresh=None,
        merge_func="",
):
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

    return merged_tv