r"""All fine-tune tasks.

All fine-tune tasks sub-module must be renamed in this file. This help to avoid
unnecessary import structure (we prefer using `fine_tune.task.MNLI` over
`fine_tune.task.task_mnli.MNLI`).

Usage:
    import fine_tune

    mnli_dataset = fine_tune.task.MNLI(...)
    boolq_dataset = fine_tune.task.BoolQ(...)

    mnli_num_class = fine_tune.task.get_num_class(fine_tune.task.MNLI)
    boolq_num_class = fine_tune.task.get_num_class(fine_tune.task.Boolq)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# my own modules

from fine_tune.task._dataset import Dataset
from fine_tune.task._dataset import get_num_class
from fine_tune.task._dataset import label_decoder
from fine_tune.task._dataset import label_encoder
from fine_tune.task._mnli import MNLI
from fine_tune.task._qnli import QNLI
from fine_tune.task._sst2 import SST2
from fine_tune.task._qqp import QQP
from fine_tune.task._rte import RTE
from fine_tune.task._mrpc import MRPC
from fine_tune.task._cola import CoLA
from fine_tune.task._stsb import STSB