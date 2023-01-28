r"""Helper functions for loading scheduler.

Usage:
    import fine_tune

    scheduler = fine_tune.util.load_scheduler(...)
    scheduler = fine_tune.util.load_scheduler_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd party modules

import torch
import torch.optim
import transformers

# my own modules

import fine_tune.config


def load_scheduler(
        optimizer: torch.optim.AdamW,
        total_step: int,
        warmup_step: int
) -> torch.optim.lr_scheduler.LambdaLR:
    r"""Load linear warmup scheduler.

    Args:
        optimizer:
            `torch.optim.AdamW` optimizer.
        total_step:
            Total number of step to perform training.
        warmup_step:
            Linear scheduler warmup step.

    Returns:
        Linear warmup scheduler provided by `transformers` package.
    """
    return transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_step,
        num_training_steps=total_step
    )


def load_scheduler_by_config(
        config: fine_tune.config.BaseConfig,
        optimizer: torch.optim.AdamW,
) -> torch.optim.lr_scheduler.LambdaLR:
    r"""Load linear warmup scheduler.

    Args:
        config:
            Configuration object which contains attributes `total_step` and
            `warmup_step`.
        optimizer:
            `torch.optim.AdamW` optimizer.

    Returns:
        Same as `fine_tune.util.load_scheduler`.
    """
    return load_scheduler(
        optimizer=optimizer,
        total_step=config.total_step,
        warmup_step=config.warmup_step
    )

def load_gate_networks_scheduler(
    optimizer: torch.optim.AdamW,
    total_step: int,
    warmup_step: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Load linear warmup scheduler for gate networks.

    Parameters
    ----------
    optimizer : torch.optim.AdamW
        `torch.optim.AdamW` optimizer.
    total_step : int
        Total number of step to perform training.
    warmup_step : int
        Linear scheduler warmup step.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Linear warmup scheduler provided by `transformers` package.
    """
    return transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_step,
        num_training_steps=total_step
    )
