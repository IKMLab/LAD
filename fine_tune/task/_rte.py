r"""RTE dataset.

Usage:
    import torch.util.data.Dataloader
    import fine_tune

    dataset = fine_tune.task.RTE('train')
    dataset = fine_tune.task.RTE('dev')

    dataloader = torch.utils.data.Dataloader(
        dataset,
        collate_fn=RTE.create_collate_fn(...)
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

# Define QNLI dataset.

class RTE(Dataset):
    """RTE dataset and its utilities

    Parameters
    ----------
    Dataset : str
        Name of the RTE dataset file to be loaded.
    """
    allow_dataset: List[str] = [
        'train',
        'dev',
        'test'
    ]

    allow_labels: List[Label] = [
        'not_entailment',
        'entailment'
    ]

    task_path: str = os.path.join(
        fine_tune.path.FINE_TUNE_DATA,
        'RTE'
    )

    @staticmethod
    def load(dataset: str) -> List[Sample]:
        """Load RTE dataset into memory.
        This is a heavy IO method and might required lots of memory since
        dataset might be huge. RTE dataset must be download previously. See
        RTE document in 'project_root/doc/fine_tune_rte.md' for downloading
        details.


        Parameters
        ----------
        dataset : str
            Name of the RTE dataset to be loaded.

        Returns
        -------
        List[Sample]
            A list of RTE samples.
        """
        try:
            dataset_path = os.path.join(
                RTE.task_path,
                f'{dataset}.tsv'
            )
            if not 'test' in dataset:
                with open(dataset_path, 'r') as tsv_file:
                    # Skip first line.
                    tsv_file.readline()
                    samples = []
                    for sample in tqdm(tsv_file, desc=f'Loading RTE {dataset}'):
                        sample = sample.strip()
                        index, sentence1, sentence2, label = sample.split('\t')
                        samples.append(
                            Sample({
                                'index': int(index),
                                'text': sentence1,
                                'text_pair': sentence2,
                                'label': label_encoder(RTE, label)
                            })
                        )

                    logger.info(
                        'Number of samples: %d',
                        len(samples)
                    )

                return samples
            else:
                # Loading testing set.
                with open(dataset_path, 'r') as tsv_file:
                    # Skip first line.
                    tsv_file.readline()
                    samples = []
                    for sample in tqdm(tsv_file, desc=f'Loading RTE {dataset}'):
                        sample = sample.strip()
                        index, sentence1, sentence2 = sample.split('\t')
                        samples.append(
                            Sample({
                                'index': int(index),
                                'text': sentence1,
                                'text_pair': sentence2,
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
                f'RTE dataset file {dataset} does not exist.\n' +
                'You must downloaded previously and put it in the path:\n' +
                f'{dataset_path}\n' +
                "See '" +
                os.path.join(fine_tune.path.DOC, 'fine_tune_rte.md') +
                "' for downloading details."
            ) from error
