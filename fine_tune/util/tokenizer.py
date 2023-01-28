r"""Helper functions for loading tokenizer.

In future, this module might need to split into multiple files, each file
contains only one model specific tokenizer.

Usage:
    import fine_tune

    teacher_tokenizer = fine_tune.util.load_teacher_tokenizer(...)
    teacher_tokenizer = fine_tune.util.load_teacher_tokenizer_by_config(...)

    student_tokenizer = fine_tune.util.load_student_tokenizer(...)
    student_tokenizer = fine_tune.util.load_student_tokenizer_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd party modules

import transformers

# my own modules

import fine_tune.config
import fine_tune.model


def load_teacher_tokenizer(
        model: str,
        ptrain_ver: str
) -> transformers.PreTrainedTokenizer:
    r"""Load teacher model paired tokenizer.

    Args:
        model:
            Name of the teacher model.
        ptrain_ver:
            Pretrained model version provided by `transformers` package.

    Raises:
        ValueError:
            If `model` does not supported.

    Returns:
        `transformers.AlbertTokenizer`:
            If `model == 'albert'`.
        `transformers.BertTokenizer`:
            If `model == 'bert'`.
    """

    if model == 'bert':
        return transformers.BertTokenizer.from_pretrained(
            ptrain_ver
        )

    raise ValueError(
        f'`model` {model} is not supported.\nSupported options:' +
        ''.join(list(map(
            lambda option: f'\n\t--model {option}',
            [
                'bert',
                'albert'
            ]
        )))
    )


def load_teacher_tokenizer_by_config(
        config: fine_tune.config.TeacherConfig
) -> transformers.PreTrainedTokenizer:
    r"""Load teacher model paired tokenizer.

    Args:
        config:
            `fine_tune.config.TeacherConfig` which contains attributes
            `model` and `ptrain_ver`.

    Returns:
        Same as `fine_tune.util.load_teacher_tokenizer`.
    """
    return load_teacher_tokenizer(
        model=config.model,
        ptrain_ver=config.ptrain_ver
    )


def load_student_tokenizer(
        model: str
) -> transformers.PreTrainedTokenizer:
    r"""Load student model paired tokenizer.

    Args:
        model:
            Name of the teacher model.

    Raises:
        ValueError:
            If `model` does not supported.

    Returns:
        `transformers.AlbertTokenizer`:
            If `model == 'albert'`. Using pre-trained tokenizer version
            'albert-base-v2'.
        `transformers.BertTokenizer`:
            If `model == 'bert'`. Using pre-trained tokenizer version
            'bert-base-uncased'.
    """

    if model == 'albert':
        return transformers.AlbertTokenizer.from_pretrained(
            'albert-base-v2'
        )
    if model == 'bert':
        return transformers.BertTokenizer.from_pretrained(
            'bert-base-uncased'
        )

    raise ValueError(
        f'`model` {model} is not supported.\n' +
        'Supported options:' +
        ''.join(list(map(
            lambda option: f'\n\t--model {option}',
            [
                'bert',
                'albert'
            ]
        )))
    )


def load_student_tokenizer_by_config(
        config: fine_tune.config.StudentConfig
) -> transformers.PreTrainedTokenizer:
    r"""Load student model paired tokenizer.

    Args:
        config:
            `fine_tune.config.StudentConfig` which contains attribute
            `student`.

    Returns:
        Same as `fine_tune.util.load_student_tokenizer`.
    """
    return load_student_tokenizer(
        model=config.model
    )
