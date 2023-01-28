r"""QQP dataset.

Usage:
    import torch.utils.data.Dataloader
    import fine_tune

    dataset = fine_tune.task.QQP('train')
    dataset = fine_tune.task.QQP('dev')

    dataloader = torch.utils.data.Dataloader(
        dataset,
        collate_fn=QQP.create_collate_fn(...)
    )
"""

# Built-in modules.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

from typing import List

# 3rd party modules

from tqdm import tqdm

# my own modules

import fine_tune.path

from fine_tune.task._dataset import (
    Dataset,
    Label,
    Sample,
    label_encoder
)

# Get logger.

logger = logging.getLogger('fine_tune.task')

# Define QQP dataset.

class QQP(Dataset):
    r"""QQP dataset and its utilities

    Parameters
    ----------
    Dataset : string
        Name of QQP dataset file to be loaded.
    """
    allow_dataset: List[str] = [
        'train',
        'dev',
        'test'
    ]

    allow_labels: List[Label] = [
        0,
        1
    ]

    task_path: str = os.path.join(
        fine_tune.path.FINE_TUNE_DATA,
        'QQP'
    )

    @staticmethod
    def load(dataset: str) -> List[Sample]:
        """Load QQP dataset into memory.
        This is a heavy IO method and might required lots of memory since
        dataset might be huge. QQP dataset must be download previously. See
        QQP document in 'project_root/doc/fine_tune_qqp.md' for downloading
        details.

        Parameters
        ----------
        dataset : str
            Name of the QQP dataset to be loaded.

        Returns
        -------
        List[Sample]
            A list of QQP samples.
        """
        try:
            dataset_path = os.path.join(
                QQP.task_path,
                f'{dataset}.tsv'
            )
            if not 'test' in dataset:
                with open(dataset_path, 'r') as tsv_file:
                    # Skip first line.
                    tsv_file.readline()
                    samples = []
                    for sample in tqdm(tsv_file, desc=f'Loading QQP {dataset}'):
                        sample = sample.strip()
                        idx, _, _, question1, question2, is_duplicate = sample.split('\t')
                        samples.append(
                            Sample({
                                'index': int(idx),
                                'text': question1,
                                'text_pair': question2,
                                'label': label_encoder(QQP, int(is_duplicate))
                            })
                        )

                    logger.info(
                        'Number of samples: %d',
                        len(samples)
                    )

                return samples
            else:
                with open(dataset_path, 'r') as tsv_file:
                    # Skip first line.
                    tsv_file.readline()
                    samples = []
                    for sample in tqdm(tsv_file, desc=f'Loading QQP {dataset}'):
                        sample = sample.strip()
                        idx,question1, question2 = sample.split('\t')
                        samples.append(
                            Sample({
                                'index': int(idx),
                                'text': question1,
                                'text_pair': question2,
                                'label': -1
                            })
                        )

                    logger.info(
                        'Number of samples: %d',
                        len(samples)
                    )

                return samples
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'QQP dataset file {dataset} does not exist.\n' +
                'You must downloaded previously and put it in the path:\n' +
                f'{dataset_path}\n' +
                "See '" +
                os.path.join(fine_tune.path.DOC, 'fine_tune_qqp.md') +
                "' for downloading details."
            ) from error
