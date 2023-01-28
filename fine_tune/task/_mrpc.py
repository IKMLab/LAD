r"""MRPC dataset.

Usage:
    import torch.utils.data.Dataloader
    import fine_tune

    dataset = fine_tune.task.MRPC('train')
    dataset = fine_tune.task.MRPC('dev')

    dataloader = torch.utils.data.Dataloader(
        dataset,
        collate_fn=MRPC.create_collate_fn(...)
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

# Define MRPC dataset.

class MRPC(Dataset):
    """MRPC dataset and its utilities

    Parameters
    ----------
    Dataset : string
        Name of MRPC dataset file to be loaded.
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
        'MRPC'
    )

    @staticmethod
    def load(dataset: str) -> List[Sample]:
        """Load MRPC dataset into memory.
        This is a heavy IO method and might required lots of memory since
        dataset might be huge. MRPC dataset must be download previously. See
        MRPC document in 'project_root/doc/fine_tune_mrpc.md' for downloading
        details.

        Parameters
        ----------
        dataset : str
            Name of the MRPC dataset to be loaded.

        Returns
        -------
        List[Sample]
            A list of MRPC samples
        """
        try:
            dataset_path = os.path.join(
                MRPC.task_path,
                f'{dataset}.tsv'
            )
            if not 'test' in dataset:
                with open(dataset_path, 'r') as tsv_file:
                    # Skip first line.
                    tsv_file.readline()
                    samples = []
                    for idx, sample in enumerate(tqdm(tsv_file, desc=f'Loading MRPC {dataset}')):
                        sample = sample.strip()
                        quality, _, _, string1, string2 = sample.split('\t')
                        samples.append(
                            Sample({
                                'index': idx,
                                'text': string1,
                                'text_pair': string2,
                                'label': label_encoder(MRPC, int(quality))
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
                    for sample in tqdm(tsv_file, desc=f'Loading MRPC {dataset}'):
                        sample = sample.strip()
                        index, _, _, string1, string2 = sample.split('\t')
                        samples.append(
                            Sample({
                                'index': int(index),
                                'text': string1,
                                'text_pair': string2,
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
                f'MRPC dataset file {dataset} does not exist.\n' +
                'You must downloaded previously and put it in the path:\n' +
                f'{dataset_path}\n' +
                "See '" +
                os.path.join(fine_tune.path.DOC, 'fine_tune_mrpc.md') +
                "' for downloading details."
            ) from error
