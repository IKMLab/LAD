r"""SST2 dataset.

Usage:
    import torch.utils.data.Dataloader
    import fine_tune

    dataset = fine_tune.task.SST2('train')
    dataset = fine_tune.task.SST2('dev')

    dataloader = torch.utils.data.Dataloader(
        dataset,
        collate_fn=SST2.create_collate_fn(...)
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

# Define SST2 dataset.

class SST2(Dataset):
    r"""SST2 dataset and its utilities

    Parameters
    ----------
    Dataset : string
        Name of SST2 dataset file to be loaded.
    """
    allow_dataset: List[str] = [
        'train',
        'dev',
        'test'
    ]

    allow_labels: List[Label] = [
        '0',
        '1'
    ]

    task_path: str = os.path.join(
        fine_tune.path.FINE_TUNE_DATA,
        'SST-2'
    )

    @staticmethod
    def load(dataset: str) -> List[Sample]:
        r"""Load SST2 dataset into memory.

        This is a heavy IO method and might required lots of memory since
        dataset might be huge. SST2 dataset must be download previously. See
        SST2 document in 'project_root/doc/fine_tune_sst2.md' for downloading
        details.
        Parameters
        ----------
        dataset : str
            Name of the SST2 dataset to be loaded.
        Returns
        -------
        List[Sample]
            A list of SST2 samples.
        """
        try:
            dataset_path = os.path.join(
                SST2.task_path,
                f'{dataset}.tsv'
            )
            if not 'test' in dataset:
                with open(dataset_path, 'r') as tsv_file:
                    # Skip first line.
                    tsv_file.readline()
                    samples = []
                    for idx, sample in enumerate(tqdm(tsv_file, desc=f'Loading SST2 {dataset}')):
                        sample = sample.strip()
                        sentence, label = sample.split('\t')
                        samples.append(
                            Sample({
                                'index' : int(idx),
                                'text' : sentence,
                                'text_pair' : None,
                                'label': label_encoder(SST2, label)
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
                    for sample in tqdm(tsv_file, desc=f'Loading SST2 {dataset}'):
                        sample = sample.strip()
                        index, sentence = sample.split('\t')
                        samples.append(
                            Sample({
                                'index' : int(index),
                                'text' : sentence,
                                'text_pair' : None,
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
                f'SST2 dataset file {dataset} does not exist.\n' +
                'You must downloaded previously and put it in the path:\n' +
                f'{dataset_path}\n' +
                "See '" +
                os.path.join(fine_tune.path.DOC, 'fine_tune_sst2.md') +
                "' for downloading details."
            ) from error
