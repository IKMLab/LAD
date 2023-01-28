r"""Helper functions for loading optimizer.

Usage:
    import fine_tune

    optimizer = fine_tune.util.load_optimizer(...)
    optimizer = fine_tune.util.load_optimizer_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Typing
from typing import Tuple
from typing import List
from typing import Union

# 3rd party modules

import torch
import torch.optim

# my own modules

import fine_tune.config
import fine_tune.model


def load_optimizer(
        betas: Tuple[float, float],
        eps: float,
        lr: float,
        model: fine_tune.model.Model,
        weight_decay: float
) -> torch.optim.AdamW:
    r"""Load `torch.optim.AdamW` optimizer.

    Args:
        betas:
            Optimizer `torch.optim.AdamW`'s beta coefficients.
        eps:
            Optimizer `torch.optim.AdamW`'s epsilon.
        lr:
            Optimizer `torch.optim.AdamW`'s learning rate.
        model:
            Model name of the current experiment.
        weight_decay:
            Optimizer `torch.optim.AdamW` weight decay regularization.

    Returns:
        AdamW optimizer.
    """
    # Remove weight decay on bias and layer-norm.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                param for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]

    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        betas=betas,
        eps=eps
    )


def load_optimizer_by_config(
        config: fine_tune.config.BaseConfig,
        model: fine_tune.model.Model
) -> torch.optim.AdamW:
    r"""Load AdamW optimizer.

    Args:
        config:
            Configuration object which contains attributes
            `lr`, `betas`, `eps` and `weight_decay`.
        model:
            Source parameters to be optimized.

    Returns:
        Same as `fine_tune.util.load_optimizer`.
    """

    return load_optimizer(
        betas=config.betas,
        eps=config.eps,
        lr=config.lr,
        model=model,
        weight_decay=config.weight_decay
    )

def load_gate_networks_optimizer(
    betas: Tuple[float, float],
    eps: float,
    lr: float,
    weight_decay: float,
    gate_networks: List[fine_tune.model.HighwayGate],
) -> torch.optim.AdamW:
    """Load `torch.optim.AdamW` optimizer.

    Parameters
    ----------
    betas : Tuple[float, float]
        Optimizer `torch.optim.AdamW`'s beta coefficients.
    eps : float
        Optimizer `torch.optim.AdamW`'s epsilon.
    lr : float
        Optimizer `torch.optim.AdamW`'s learning rate.
    weight_decay : float
        Optimizer `torch.optim.AdamW` weight decay regularization.
    gate_networks : List[fine_tune.model.HighwayGate]
        A list of `fine_tune.model.HighwayGate`.

    Returns
    -------
    torch.optim.AdamW
        AdamW optimizer.
    """
    # Remove weight decay on bias and layer-norm.
    no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
    optimizer_grouped_parameters = []

    # Add trainable parameters of each gate to optimizer.
    for gate in gate_networks:
        # Add parameters that need weight decay.
        optimizer_grouped_parameters.append(
            {
                'params': [
                    param for name, param in gate.named_parameters()
                    if not any(nd in name for nd in no_decay)
                ],
                'weight_decay': weight_decay,
            }
        )

        # Add parameters that don not need weight decay.
        optimizer_grouped_parameters.append(
            {
                'params': [
                    param for name, param in gate.named_parameters()
                    if any(nd in name for nd in no_decay)
                ],
                'weight_decay': 0.0,
            }
        )

    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        betas=betas,
        eps=eps
    )
