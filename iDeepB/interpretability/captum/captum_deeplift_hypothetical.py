
# from: https://github.com/pytorch/captum/blob/2d074c39f4e43fe7d1cd4ab105ac3dc560e5bcd2/tests/attr/test_deeplift_basic.py#L144

import random
import numpy as np
import torch
from typing import Tuple
from torch import Tensor
import sys

def set_all_random_seeds(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def hypothetical_contrib_func(
    multipliers: Tuple[Tensor, ...],  # gradient
    inputs: Tuple[Tensor, ...],
    baselines: Tuple[Tensor, ...],
) -> Tuple[Tensor, ...]:
    r"""
    Implements hypothetical input contributions based on the logic described here:
    https://github.com/kundajelab/deeplift/pull/36/files
    https://github.com/kundajelab/deeplift/blob/master/deeplift/util.py
    This is using a dummy model for test purposes
    """
    # we assume that multiplies, inputs and baselines have the following shape:
    # tuple((bsz x len x channel), )
    assert len(multipliers[0].shape) == 3, multipliers[0].shape
    assert len(inputs[0].shape) == 3, inputs[0].shape
    assert len(baselines[0].shape) == 3, baselines[0].shape
    assert len(multipliers) == len(inputs) and len(inputs) == len(baselines), (
        "multipliers, inputs and baselines must have the same shape but"
        "multipliers: {}, inputs: {}, baselines: {}".format(
            len(multipliers), len(inputs), len(baselines)
        )
    )

    attributions = []
    for k in range(len(multipliers)):
        sub_attributions = torch.zeros_like(inputs[k])
        for i in range(inputs[k].shape[-1]):
            hypothetical_input = torch.zeros_like(inputs[k])
            hypothetical_input[:, :, i] = 1.0
            hypothetical_input_ref_diff = hypothetical_input - baselines[k]
            sub_attributions[:, :, i] = torch.sum(
                hypothetical_input_ref_diff * multipliers[k], dim=-1
            )
        attributions.append(sub_attributions)
    return tuple(attributions)

