r"""STS-B dataset.

Usage:
    import torch.utils.data.Dataloader
    import fine_tune
    dataset = fine_tune.task.STSB('train')
    dataset = fine_tune.task.STSB('dev')
    dataloader = torch.utils.data.Dataloader(
        dataset,
        collate_fn=STSB.create_collate_fn(...)
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

# Define STS-B dataset.

class STSB(Dataset):
    """STS-B dataset and its utilities

    Parameters
    ----------
    Dataset : string
        Name of STS-B dataset file to be loaded.
    """
    allow_dataset: List[str] = [
        'train',
        'dev',
        'test'
    ]
    task_path: str = os.path.join(
        fine_tune.path.FINE_TUNE_DATA,
        'STS-B'
    )

    @staticmethod
    def load(dataset: str) -> List[Sample]:
        """Load STS-B dataset into memory.
        
        Parameters
        ----------
        dataset : str
            Name of the STS-B dataset to be loaded.
        Returns
        -------
        List[Sample]
            A list of STS-B dataset to be loaded.
        """
        try:
            dataset_path = os.path.join(STSB.task_path, f'{dataset}.tsv')
            if dataset == 'test':
                print("Loading test set!!")
                with open(dataset_path, 'r') as tsv_file:
                    # Skip first line.
                    tsv_file.readline()
                    samples = []
                    for sample in tqdm(tsv_file, desc=f'Loading STS-B {dataset}'):
                        sample = sample.strip()
                        idx, _, _, _, _, _, _, string1, string2 = sample.split('\t')
                        samples.append(
                            Sample({
                                'index': int(idx),
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
            else:
                with open(dataset_path, 'r') as tsv_file:
                    # Skip first line.
                    tsv_file.readline()
                    samples = []
                    for sample in tqdm(tsv_file, desc=f'Loading STS-B {dataset}'):
                        sample = sample.strip()
                        idx, _, _, _, _, _, _, string1, string2, label = sample.split('\t')
                        samples.append(
                            Sample({
                                'index': idx,
                                'text': string1,
                                'text_pair': string2,
                                'label': float(label)
                            })
                        )
                    logger.info(f'Number of original samples: {len(samples)}')
                return samples
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'STS-B dataset file {dataset} does not exist.\n' +
                'You must downloaded previously and put it in the path:\n' +
                f'{dataset_path}\n' +
                "See '" +
                os.path.join(fine_tune.path.DOC, 'fine_tune_stsb.md') +
                "' for downloading details."
            ) from error            