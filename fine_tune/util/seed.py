r"""Helper functions for setting random seed.

Usage:
    import fine_tune

    fine_tune.util.set_seed(...)
    fine_tune.util.set_seed_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

# 3rd party modules

import numpy as np
import torch

# my own modules

import fine_tune.config


def set_seed(device_id: int, seed: int):
    r"""Control random seed for experiment reproducibility.

    Args:
        device_id:
            Number of GPUs used to run experiment.
        seed:
            Random seed value to be initialized.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available() and device_id > -1:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_seed_by_config(
        config: fine_tune.config.BaseConfig
):
    r"""Control random seed for experiment reproducibility.

    Args:
        config:
            Configuration object which contains attributes `seed` and
            `num_gpu`.
    """
    set_seed(
        device_id=config.device_id,
        seed=config.seed
    )
