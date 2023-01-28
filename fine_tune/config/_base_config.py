r"""Configuration for fine-tune experiments.

This module serve as base class of all configurations, and should never be used
directly. Use configuration class derived from `fine_tune.config.BaseConfig`
instead (e.g. `fine_tune.config.StudentConfig`).

Usage:
    import fine_tune

    class MyConfig(BaseConfig):
        ...
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
from typing import Type
from typing import Union

# 3rd party modules

import torch

# my own modules

import fine_tune.path


class BaseConfig:
    r"""Configuration for fine-tune experiments.

    All models will be optimized with `torch.optim.AdamW`. Optimization will
    be paired with linear warmup scheduler.

    Attributes:
        accum_step:
            Gradient accumulation step. Used when GPU memory cannot fit in
            whole batch. `accum_step` must be bigger than or equal to `1`;
            `accum_step` must be smaller than or equal to `batch_size`.
        batch_size:
            Training batch size. `batch_size` must be bigger than or equal to
            `1`; `batch_size` must be greater than or equal to `accum_step`.
        beta1:
            Optimizer `torch.optim.AdamW`'s beta coefficients.
            `beta1` must be ranging from `0` to `1` (inclusive).
        beta2:
            Optimizer `torch.optim.AdamW`'s beta coefficients
            `beta2` must be ranging from `0` to `1` (inclusive).
        ckpt_step:
            Checkpoint save interval. `ckpt_step` must be bigger than or equal
            to `1`.
        dataset:
            Dataset name of the fine-tune task. (e.g., task `MNLI` have dataset
            'train', 'dev_matched' and 'dev_mismatched'.) `dataset` must not be
            empty string.
        dropout:
            Dropout probability. `dropout` must be ranging from `0` to `1`
            (inclusive).
        eps:
            Optimizer `torch.optim.AdamW`'s epsilon. `eps` must be bigger than
            `0`.
        experiment:
            Name of the current experiment. `experiment` must not be empty
            string.
        log_step:
            Logging interval. `log_step` must be bigger than or equal to `1`.
        lr:
            Optimizer `torch.optim.AdamW`'s learning rate. `lr` must be bigger
            than `0`.
        max_norm:
            Maximum norm of gradient. Used when performing gradient cliping.
            `max_norm` must be bigger than `0`.
        max_seq_len:
            Maximum input sequence length of model input. `max_seq_len` must be
            bigger than or equal to `1`.
        model:
            Model name of the current experiment.
        num_class:
            Number of classes to classify. `num_class` must be bigger than or
            equal to `2`.
        seed:
            Control random seed. `seed` must be bigger than or equal to `1`.
        task:
            Name of the fine-tune task. `task` must not be empty string.
        total_step:
            Total number of step to perform training. `total_step` must be
            bigger than or equal to `1`.
        warmup_step
            Linear scheduler warm up step. `warmup_step` must be bigger than
            or equal to `1`.
        weight_decay:
            Optimizer `torch.optim.AdamW` weight decay regularization.
            `weight_decay` must be bigger than or equal to `0`.
        device_id:
            ID of GPU device, set to `-1` to use CPU.

    Raises:
        OSError:
            If `num_gpu > 0` and CUDA device are not available.
        TypeError:
            If type constrains on any attributes failed.
            See type annotation on arguments.
        ValueError:
            If constrains on any attributes failed.
            See attributes section for details.
    """

    def __init__(
            self,
            accum_step: int = 1,
            batch_size: int = 32,
            beta1: float = 0.9,
            beta2: float = 0.999,
            ckpt_step: int = 1000,
            dataset: str = '',
            dropout: float = 0.1,
            eps: float = 1e-8,
            experiment: str = '',
            log_step: int = 500,
            lr: float = 3e-5,
            max_norm: float = 1.0,
            max_seq_len: int = 512,
            model: str = '',
            num_class: int = 2,
            seed: int = 42,
            task: str = '',
            total_step: int = 50000,
            warmup_step: int = 10000,
            weight_decay: float = 0.01,
            device_id: int = -1
    ):

        self.__class__.type_check(accum_step, 'accum_step', int)
        self.__class__.type_check(batch_size, 'batch_size', int)
        self.__class__.type_check(beta1, 'beta1', float)
        self.__class__.type_check(beta2, 'beta2', float)
        self.__class__.type_check(ckpt_step, 'ckpt_step', int)
        self.__class__.type_check(dataset, 'dataset', str)
        self.__class__.type_check(dropout, 'dropout', float)
        self.__class__.type_check(eps, 'eps', float)
        self.__class__.type_check(experiment, 'experiment', str)
        self.__class__.type_check(log_step, 'log_step', int)
        self.__class__.type_check(lr, 'lr', float)
        self.__class__.type_check(max_norm, 'max_norm', float)
        self.__class__.type_check(max_seq_len, 'max_seq_len', int)
        self.__class__.type_check(model, 'model', str)
        self.__class__.type_check(num_class, 'num_class', int)
        self.__class__.type_check(seed, 'seed', int)
        self.__class__.type_check(task, 'task', str)
        self.__class__.type_check(total_step, 'total_step', int)
        self.__class__.type_check(warmup_step, 'warmup_step', int)
        self.__class__.type_check(weight_decay, 'weight_decay', float)
        self.__class__.type_check(device_id, 'device_id', int)

        if accum_step < 1:
            raise ValueError(
                '`accum_step` must be bigger than or equal to `1`.'
            )

        if batch_size < 1:
            raise ValueError(
                '`batch_size` must be bigger than or equal to `1`.'
            )

        if accum_step > batch_size:
            raise ValueError(
                '`batch_size` must be bigger than or equal to `accum_step`.'
            )

        if not 0 <= beta1 <= 1:
            raise ValueError(
                '`beta1` must be ranging from `0` to `1` (inclusive).'
            )

        if not 0 <= beta2 <= 1:
            raise ValueError(
                '`beta2` must be ranging from `0` to `1` (inclusive).'
            )

        if ckpt_step < 1:
            raise ValueError(
                '`ckpt_step` must be bigger than or equal to `1`.'
            )

        if not dataset:
            raise ValueError(
                '`dataset` must not be empty string.'
            )

        if not 0 <= dropout <= 1:
            raise ValueError(
                '`dropout` must be ranging from `0` to `1` (inclusive).'
            )

        if eps <= 0:
            raise ValueError(
                '`eps` must be bigger than `0`.'
            )

        if not experiment:
            raise ValueError(
                '`experiment` must not be empty string.'
            )

        if log_step < 1:
            raise ValueError(
                '`log_step` must be bigger than or equal to `1`.'
            )

        if lr <= 0:
            raise ValueError(
                '`lr` must be bigger than `0`.'
            )

        if max_norm <= 0:
            raise ValueError(
                '`max_norm` must be bigger than `0`.'
            )

        if max_seq_len < 1:
            raise ValueError(
                '`max_seq_len` must be bigger than or equal to `1`.'
            )

        if not model:
            raise ValueError(
                '`model` must not be empty string.'
            )

        if num_class < 1:
            raise ValueError(
                # '`num_class` must be bigger than or equal to `2`.'
                'Incorrect num_class.'
            )

        if seed < 1:
            raise ValueError(
                '`seed` must be bigger than or equal to `1`.'
            )

        if not task:
            raise ValueError(
                '`task` must not be empty string.'
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

        if device_id + 1  > torch.cuda.device_count():
            raise OSError(
                'Invalid `device_id` please check you GPU device count.\n'+
                f'There are only {torch.cuda.device_count()} cuda device \n'+
                f'Available device id range: 0~{torch.cuda.device_count()-1}\n'+
                f'Received device_id: {device_id}'
            )

        self.accum_step = accum_step
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.ckpt_step = ckpt_step
        self.dataset = dataset
        self.dropout = dropout
        self.eps = eps
        self.experiment = experiment
        self.log_step = log_step
        self.lr = lr
        self.max_norm = max_norm
        self.max_seq_len = max_seq_len
        self.model = model
        self.num_class = num_class
        self.seed = seed
        self.task = task
        self.total_step = total_step
        self.warmup_step = warmup_step
        self.weight_decay = weight_decay
        self.device_id = device_id

    @staticmethod
    def type_check(
            arg: Union[bool, float, int, str],
            arg_name: str,
            arg_type: Union[Type[bool], Type[float], Type[int], Type[str]]
    ):
        r"""Perform type checking using built-in function `isinstance`.

        Args:
            arg:
                Argument to perform typing checking on.
            arg_name:
                Name of `arg`.
            arg_type:
                Type of `arg`.

        Raises:
            TypeError:
                If type checking failed.
        """
        if not isinstance(arg, arg_type):
            raise TypeError(
                f'`{arg_name}` must be type `{arg_type.__name__}`.'
            )

    def __iter__(self) -> Generator[
            Tuple[str, Union[bool, float, int, str]], None, None
    ]:
        yield 'accum_step', self.accum_step
        yield 'batch_size', self.batch_size
        yield 'beta1', self.beta1
        yield 'beta2', self.beta2
        yield 'ckpt_step', self.ckpt_step
        yield 'dataset', self.dataset
        yield 'dropout', self.dropout
        yield 'eps', self.eps
        yield 'experiment', self.experiment
        yield 'log_step', self.log_step
        yield 'lr', self.lr
        yield 'max_norm', self.max_norm
        yield 'max_seq_len', self.max_seq_len
        yield 'model', self.model
        yield 'num_class', self.num_class
        yield 'seed', self.seed
        yield 'task', self.task
        yield 'total_step', self.total_step
        yield 'warmup_step', self.warmup_step
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

    def save(self) -> None:
        r"""Save configuration into json file."""
        file_path = BaseConfig.file_path(
            experiment=self.experiment,
            model=self.model,
            task=self.task
        )

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(
                dict(self),
                json_file,
                ensure_ascii=False
            )

    @classmethod
    def load(
            cls,
            experiment: str = '',
            model: str = '',
            task: str = '',
            device_id: int = None
    ):
        """Load config from json file.

        Parameters
        ----------
        experiment : str, optional
            Name of the experiment, by default ''
        model : str, optional
            Model name of the experiment, by default ''
        task : str, optional
            Name of the fine-tune task, by default ''
        device_id : int, optional
            Model device id.
            Specify which device a model should reside
            or follow the value of configuration file, by default None
        """
        file_path = cls.file_path(
            experiment=experiment,
            model=model,
            task=task
        )
        with open(file_path, 'r') as json_file:
            config = json.load(json_file)
            if device_id is not None:
                config['device_id'] = device_id
            return cls(**config)

    @staticmethod
    def experiment_name(
            experiment: str,
            model: str,
            task: str
    ) -> str:
        r"""Return formatted experiment name.

        Args:
            experiment:
                Name of the experiment.
            model:
                Model name of the experiment.
            task:
                Name of the fine-tune task.

        Returns:
            Formatted experiment name.
        """
        return f'{experiment}_{model}_{task}'

    @staticmethod
    def file_path(
            experiment: str,
            model: str,
            task: str
    ) -> str:
        r"""Get configuration file path.

        Args:
            experiment:
                Name of the experiment.
            model:
                Model name of the experiment.
            task:
                Name of the fine-tune task.

        Returns:
            Configuration file path.
        """
        return os.path.join(
            fine_tune.path.FINE_TUNE_EXPERIMENT,
            BaseConfig.experiment_name(
                experiment=experiment,
                model=model,
                task=task
            ),
            'config.json'
        )
