r"""Helper functions for loading dataset.

Usage:
    import fine_tune

    dataset = fine_tune.util.load_dataset(...)
    dataset = fine_tune.util.load_dataset_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# my own modules

import fine_tune.config
import fine_tune.task


def load_dataset(
        dataset: str,
        task: str,
) -> fine_tune.task.Dataset:
    r"""Load fine-tune task's dataset.

    Args:
        dataset:
            Name of the dataset file to be loaded.
        task:
            Name of the fine-tune task.

    Raises:
        ValueError:
            If `task` does not supported.

    Returns:
        `fine_tune.task.MNLI`:
            If `task` is 'mnli'.
        `fine_tune.task.BoolQ`:
            If `task` is 'boolq'.
        `fine_tune.task.QNLI`:
            If `task` is `qnli`
        `fine_tune.task.QQP`:
            If `task` is `qqp`
        `fine_tune.task.RTE`:
            If `task` is `rte`
        `fine_tune.task.MRPC`:
            If `task` is `mrpc`
    """
    if task == 'mnli':
        return fine_tune.task.MNLI(dataset)
    if task == 'boolq':
        return fine_tune.task.BoolQ(dataset)
    if task == 'qnli':
        return fine_tune.task.QNLI(dataset)
    if task == 'sst2':
        return fine_tune.task.SST2(dataset)
    if task == 'qqp':
        return fine_tune.task.QQP(dataset)
    if task == 'rte':
        return fine_tune.task.RTE(dataset)
    if task == 'mrpc':
        return fine_tune.task.MRPC(dataset)
    if task == 'cola':
        return fine_tune.task.CoLA(dataset)
    if task == 'stsb':
        return fine_tune.task.STSB(dataset)

    raise ValueError(
        f'`task` {task} is not supported.\nSupported options:' +
        ''.join(list(map(
            lambda option: f'\n\t--task {option}',
            [
                'mnli',
                'boolq'
            ]
        )))
    )

def load_dataset_by_config(
        config: fine_tune.config.BaseConfig
) -> fine_tune.task.Dataset:
    r"""Load fine-tune task's dataset.

    Args:
        config:
            Configuration object which contains attributes `task`
            and `dataset`.

    Returns:
        Same as `fine_tune.util.load_data`.
    """
    return load_dataset(
        dataset=config.dataset,
        task=config.task
    )
