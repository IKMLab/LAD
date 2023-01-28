r"""CoLA dataset.

Usage:
    import torch.utils.data.Dataloader
    import fine_tune
    dataset = fine_tune.task.CoLA('train')
    dataset = fine_tune.task.CoLA('dev')
    dataloader = torch.utils.data.Dataloader(
        dataset,
        collate_fn=CoLA.create_collate_fn(...)
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

# Define CoLA dataset.

class CoLA(Dataset):
    """CoLA dataset and its utilities
    Parameters
    ----------
    Dataset : string
        Name of CoLA dataset file to be loaded.
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
        'CoLA'
    )

    @staticmethod
    def load(dataset: str) -> List[Sample]:
        """Load CoLA dataset into memory.
        This is a heavy IO method and might require lots of memory since
        dataset might be huge. CoLA dataset must be download previously. See
        MRPC document in 'project_root/doc/fine_tune_cola.md' for downloading
        details.
        Parameters
        ----------
        dataset : str
            Name of the CoLA dataset to be loaded.
        Returns
        -------
        List[Sample]
            A list of CoLA dataset to be loaded.
        """
        try:
            dataset_path = os.path.join(
                CoLA.task_path,
                f'{dataset}.tsv'
            )
            if dataset == 'test':
                print("Loading test set!!")
                with open(dataset_path, 'r') as tsv_file:
                    # Skip first line.
                    tsv_file.readline()
                    samples = []
                    for sample in tqdm(tsv_file, desc=f'Loading CoLA {dataset}'):
                        sample = sample.strip()
                        index, sentence = sample.split('\t')
                        samples.append(
                            Sample({
                                'index': int(index),
                                'text': sentence,
                                'text_pair': None,
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
                    samples = []
                    for sample in tqdm(tsv_file, desc=f'Loading CoLA {dataset}'):
                        sample = sample.strip()
                        _, label, _, sentence = sample.split('\t')
                        samples.append(
                            Sample({
                                'text': sentence,
                                'text_pair': None,
                                'label': label_encoder(CoLA, int(label))
                            })
                        )

                    logger.info(
                        'Number of samples: %d',
                        len(samples)
                    )

                return samples
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'CoLA dataset file {dataset} does not exist.\n' +
                'You must downloaded previously and put it in the path:\n' +
                f'{dataset_path}\n' +
                "See '" +
                os.path.join(fine_tune.path.DOC, 'fine_tune_mrpc.md') +
                "' for downloading details."
            ) from error