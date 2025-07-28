"""
    file adapted from: https://github.com/CVLAB-Unibo/nf2vec/blob/main/nerf2vec/utils.py
"""

import torch

from collections import OrderedDict
from typing import Any, Dict
from torch import Tensor

TINY_CUDA_MIN_SIZE = 16

def next_multiple(val, divisor):
    """
    Implementation ported directly from TinyCuda implementation
    See https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/common.h#L300
    """
    return next_pot(div_round_up(val, divisor) * divisor)


def div_round_up(val, divisor):
	return next_pot((val + divisor - 1) / divisor)


def next_pot(v):
    v=int(v)
    v-=1
    v | v >> 1
    v | v >> 2
    v | v >> 4
    v | v >> 8
    v | v >> 16
    return v+1


def next_multiple_2(val, divisor):
    """
    Additional implementation added for testing purposes
    """
    return ((val - 1) | (divisor -1)) + 1


def get_mlp_params_as_matrix(flattened_params: Tensor, sd: Dict[str, Any]) -> Tensor:
    if sd is None:
        raise NotImplementedError

    params_shapes = [p.shape for p in sd.values()]
    feat_dim = params_shapes[0][0]

    # padding_size = (feat_dim-params_shapes[-1][0]) * params_shapes[-1][1]
    # padding_tensor = torch.zeros(padding_size)
    # params = torch.cat((flattened_params, padding_tensor), dim=0)
    return flattened_params.reshape((-1, feat_dim))

def get_prediction_accuracy(logits, labels):
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred_classes = torch.argmax(probs, dim=1)
    acc = (pred_classes == labels).float().mean().item()

    tp_mask = (pred_classes == 1) & (labels == 1)
    tn_mask = (pred_classes == 0) & (labels == 0)
    fp_mask = (pred_classes == 1) & (labels == 0)
    fn_mask = (pred_classes == 0) & (labels == 1)

    encoded_labels = torch.empty_like(labels)
    encoded_labels[tp_mask] = 0
    encoded_labels[tn_mask] = 1
    encoded_labels[fp_mask] = 2
    encoded_labels[fn_mask] = 3

    return {
        "preds":pred_classes,
        "preds_labeled":encoded_labels,
        "accuracy": acc,
        "tp": tp_mask.sum().item(),
        "tn": tn_mask.sum().item(),
        "fp": fp_mask.sum().item(),
        "fn": fn_mask.sum().item(),
    }
