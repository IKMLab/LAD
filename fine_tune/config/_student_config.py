r"""Configuration for fine-tune distillation experiments.

Usage:
    import fine_tune

    config = fine_tune.config.StudentConfig(...)
    config.save()
    config = fine_tune.config.StudentConfig.load(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Generator
from typing import Tuple
from typing import Union

# my own modules

from fine_tune.config._base_config import BaseConfig


class StudentConfig(BaseConfig):
    r"""Configuration for fine-tune distillation student model.

    All student models will be optimized with `torch.optim.AdamW`. Optimization
    will be paired with linear warmup scheduler.

    Attributes:
        accum_step:
            Gradient accumulation step. Used when GPU memory cannot fit in
            whole batch. `accum_step` must be bigger than or equal to `1`;
            `accum_step` must be smaller than or equal to `batch_size`.
        batch_size:
            Distillation batch size. `batch_size` must be bigger than or equal
            to `1`; `batch_size` must be greater than or equal to `accum_step`.
        beta1:
            Optimizer `torch.optim.AdamW`'s beta coefficients.
            `beta1` must be ranging from `0` to `1` (inclusive).
        beta2:
            Optimizer `torch.optim.AdamW`'s beta coefficients
            `beta2` must be ranging from `0` to `1` (inclusive).
        ckpt_step:
            Checkpoint save interval. `ckpt_step` must be bigger than or equal
            to `1`.
        d_emb:
            Embedding dimension.
            Must be bigger than or equal to `1`.
        d_ff:
            Transformer layers feed forward dimension.
            Must be bigger than `0`.
        d_model:
            Transformer layers hidden dimension.
            Must be bigger than `0`.
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
        num_attention_heads:
            Number of attention heads in Transformer layers.
            Must be bigger than `0`.
        num_class:
            Number of classes to classify. `num_class` must be bigger than or
            equal to `2`.
        num_hidden_layers:
            Number of Transformer layers.
            Must be bigger than or equal to `1`.
        seed:
            Control random seed. `seed` must be bigger than or equal to `1`.
        task:
            Name of the fine-tune task. `task` must not be empty string.
        total_step:
            Total number of step to perform training. `total_step` must be
            bigger than or equal to `1`.
        type_vocab_size:
            BERT-like models token type embedding range.
            Must be bigger than `0`.
        warmup_step
            Linear scheduler warm up step. `warmup_step` must be bigger than
            or equal to `1`.
        weight_decay:
            Optimizer `torch.optim.AdamW` weight decay regularization.
            `weight_decay` must be bigger than or equal to `0`.
        device_id:
            ID of GPU device, set to `-1` to use CPU.
        softmax_temp:
            Softmax temperature.
        hard_weight:
            Weight of hard target.
        soft_weight:
            Weight of soft target.
        hidden_mse_weight:
            Weight of hidden MSE loss.

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
            d_emb: int = 128,
            d_ff: int = 3072,
            d_model: int = 768,
            dataset: str = '',
            dropout: float = 0.1,
            eps: float = 1e-8,
            experiment: str = '',
            log_step: int = 500,
            lr: float = 3e-5,
            max_norm: float = 1.0,
            max_seq_len: int = 512,
            model: str = '',
            num_attention_heads: int = 16,
            num_class: int = 2,
            num_hidden_layers: int = 6,
            seed: int = 42,
            task: str = '',
            total_step: int = 50000,
            type_vocab_size: int = 2,
            warmup_step: int = 10000,
            weight_decay: float = 0.01,
            device_id: int = -1,
            softmax_temp: int = 1,
            hard_weight: float = 0,
            soft_weight: float = 1,
            hidden_mse_weight: int = 1,
    ):
        super().__init__(
            accum_step=accum_step,
            batch_size=batch_size,
            beta1=beta1,
            beta2=beta2,
            ckpt_step=ckpt_step,
            dataset=dataset,
            dropout=dropout,
            eps=eps,
            experiment=experiment,
            log_step=log_step,
            lr=lr,
            max_norm=max_norm,
            max_seq_len=max_seq_len,
            model=model,
            num_class=num_class,
            seed=seed,
            task=task,
            total_step=total_step,
            warmup_step=warmup_step,
            weight_decay=weight_decay,
            device_id=device_id
        )

        self.__class__.type_check(d_emb, 'd_emb', int)
        self.__class__.type_check(d_ff, 'd_ff', int)
        self.__class__.type_check(d_model, 'd_model', int)
        self.__class__.type_check(
            num_attention_heads, 'num_attention_heads', int)
        self.__class__.type_check(num_hidden_layers, 'num_hidden_layers', int)
        self.__class__.type_check(type_vocab_size, 'type_vocab_size', int)
        self.__class__.type_check(softmax_temp, 'softmax_temp', float)
        self.__class__.type_check(hard_weight, 'hard_weight', float)
        self.__class__.type_check(soft_weight, 'soft_weight', float)
        self.__class__.type_check(hidden_mse_weight, 'hidden_mse_weight', float)

        self.d_emb = d_emb
        self.d_ff = d_ff
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.softmax_temp = softmax_temp
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.hidden_mse_weight = hidden_mse_weight

    def __iter__(self) -> Generator[
            Tuple[str, Union[float, int, str]], None, None
    ]:
        attrs = list(super().__iter__()) + [
            ('d_emb', self.d_emb),
            ('d_ff', self.d_ff),
            ('d_model', self.d_model),
            ('num_attention_heads', self.num_attention_heads),
            ('num_hidden_layers', self.num_hidden_layers),
            ('type_vocab_size', self.type_vocab_size),
            ('softmax_temp', self.softmax_temp),
            ('hard_weight', self.hard_weight),
            ('soft_weight', self.soft_weight),
            ('hidden_mse_weight', self.hidden_mse_weight),
        ]
        # Sorted by attributes' name.
        attrs = sorted(attrs, key=lambda t: t[0])
        for attr_name, attr_value in attrs:
            yield attr_name, attr_value
