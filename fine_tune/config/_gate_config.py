r"""Configuration for `fine_tune.model.HighwayGate`.
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os

from typing import Generator
from typing import Tuple
# from typing import Type
from typing import Union

# 3rd party modules

import torch

# my own modules

# import fine_tune.path

class GateConfig():

    """Configuration object for `fine_tune.model.HighwayGate`.

        Attributes
        ----------
        dimension : int
            Dimension of linear layer
        max_seq_length : int
            Max sequence length of input hidden states
        beta1 : float, optional
            Optimizer `torch.optim.AdamW`'s beta coefficients.
            `beta1` must be ranging from `0` to `1` (inclusive), by default 0.9
        beta2 : float, optional
            Optimizer `torch.optim.AdamW`'s beta coefficients
            `beta2` must be ranging from `0` to `1` (inclusive), by default 0.999
        eps : float, optional
            Optimizer `torch.optim.AdamW`'s epsilon. `eps` must be bigger than
            `0`, by default 1e-8
        total_step : int, optional
            Total number of step to perform training.
            `total_step` must be bigger than or equal to `1`, by default 50000
        warmup_step : int, optional
            Linear scheduler warm up step. `warmup_step` must be bigger than
            or equal to `1`, by default 10000
        lr : float, optional
            Optimizer `torch.optim.AdamW`'s learning rate. `lr` must be bigger
            than `0`, by default 3e-5
        max_norm : float, optional
            Maximum norm of gradient. Used when performing gradient cliping.
            `max_norm` must be bigger than `0`, by default 1.0
        weight_decay : float, optional
            Optimizer `torch.optim.AdamW` weight decay regularization.
            `weight_decay` must be bigger than or equal to `0`, by default 0.1
        device_id : int, optional
            ID of GPU device, set to `-1` to use CPU, by default -1
    """
    def __init__(
        self,
        dimension: int,
        max_seq_length: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        total_step: int = 50000,
        warmup_step: int = 10000,
        lr: float = 3e-5,
        max_norm: float = 1.0,
        weight_decay: float = 0.1,
        device_id: int = -1
    ):

        if dimension < 1:
            raise ValueError(
                '`dimension` should be greater than `1`'
            )

        if max_seq_length < 1:
            raise ValueError(
                '`max_seq_length` should be greater than `1`'
            )

        if not 0 <= beta1 <= 1:
            raise ValueError(
                '`beta1` must be ranging from `0` to `1` (inclusive).'
            )

        if not 0 <= beta2 <= 1:
            raise ValueError(
                '`beta2` must be ranging from `0` to `1` (inclusive).'
            )

        if eps <= 0:
            raise ValueError(
                '`eps` must be bigger than `0`.'
            )

        if lr <= 0:
            raise ValueError(
                '`lr` must be bigger than `0`.'
            )

        if max_norm <= 0:
            raise ValueError(
                '`max_norm` must be bigger than `0`.'
            )

        if total_step < 1:
            raise ValueError(
                '`total_step` must be bigger than or equal to `1`.'
            )

        if warmup_step < 1:
            raise ValueError(
                '`warmup_step` must be bigger than or equal to `1`.'
            )

        if weight_decay < 0:
            raise ValueError(
                '`weight_decay` must be bigger than or equal to `0`.'
            )

        if device_id < -1:
            raise ValueError(
                '`device_id` must be -1 (CPU) or any cuda device number(GPU)'
            )

        if device_id > torch.cuda.device_count():
            raise OSError(
                'Invalid `device_id` please check you GPU device count.'
            )


        self.dimension = dimension
        self.max_seq_length = max_seq_length
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.total_step = total_step
        self.warmup_step = warmup_step
        self.lr = lr
        self.max_norm = max_norm
        self.weight_decay = weight_decay
        self.device_id = device_id

    def __iter__(self) -> Generator[
        Tuple[str, Union[bool, float, int, str]], None, None
    ]:
        yield 'dimension', self.dimension
        yield 'max_seq_length', self.max_seq_length
        yield 'beta1', self.beta1
        yield 'beta2', self.beta2
        yield 'eps', self.eps
        yield 'total_step', self.total_step
        yield 'warmup_step', self.warmup_step
        yield 'lr', self.lr
        yield 'max_norm', self.max_norm
        yield 'weight_decay', self.weight_decay
        yield 'device_id', self.device_id

    def __str__(self) -> str:
        col_width = max([
            max(len(k), len(str(v)))
            for k, v in self
        ])
        table_width = 2 * (col_width + 2) + 1
        sep = '\n+' + '-' * table_width + '+'
        row = '\n| {:<{col_width}} | {:<{col_width}} |'
        table = (
            sep +
            row.format('configuration', 'value', col_width=col_width) +
            sep +
            ''.join([
                row.format(k, v, col_width=col_width)
                for k, v in self
            ]) +
            sep
        )

        return table

    @property
    def betas(self) -> Tuple[float, float]:
        r"""Optimizer `torch.optim.AdamW`'s beta coefficients.

        Returns:
            A tuple contain two values, `self.beta1, self.beta2`.
        """
        return self.beta1, self.beta2

    @property
    def device(self) -> torch.device:
        r"""Get running model device.

        If `self.num_gpu == 0`, then run model on CPU.
        Else run model on CUDA device.

        Returns:
            Device create by `torch.device`.
        """
        if self.device_id == -1:
            return torch.device('cpu')
        return torch.device(f'cuda:{self.device_id}')

    def save(self, file_path: str) -> None:
        """Save configuration to specified file path.
        Just pass in the desired file path,
        and the file name would be `file_path/gate_config.json`.

        Parameters
        ----------
        file_path : str
            Path of configuration to be saved.
        """
        file_path = os.path.join(file_path, 'gate_config.json')
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(
                dict(self),
                json_file,
                ensure_ascii=False
            )

    @classmethod
    def load(cls, file_path: str):
        """Load configuration from json file.

        Parameters
        ----------
        file_path : str
            Path of the configuration file to be loaded.
        """
        if 'gate_config.json' not in file_path:
            file_path = os.path.join(file_path, 'gate_config.json')

        with open(file_path, 'r') as json_file:
            return cls(**json.load(json_file))
