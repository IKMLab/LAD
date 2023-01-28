r"""Helper functions for evaluating fine-tuned model.

Usage:
    import fine_tune

    acc = fine_tune.util.evaluation(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd party modules

import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import torch.utils.tensorboard
import transformers

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Typing
from typing import Tuple
from typing import List
from fine_tune.task._dataset import Label

# my own modules

import fine_tune.config
import fine_tune.task
import fine_tune.model

@torch.no_grad()
def evaluate_acc_and_f1(
    config: fine_tune.config.BaseConfig,
    dataset: fine_tune.task.Dataset,
    model: fine_tune.model.Model,
    tokenizer: transformers.PreTrainedTokenizer
) -> Tuple[float, float]:
    """Evaluate model accuracy and F1 score on downstream tasks.

    Parameters
    ----------
    config : fine_tune.config.BaseConfig
        `fine_tune.config.BaseConfig` subclass which attributes are used
        for experiment setup.
    dataset : fine_tune.task.Dataset
        Task specific dataset.
    model : fine_tune.model.Model
        Model which will be evaluated on `dataset`.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with `model`.

    Returns
    -------
    Tuple[float, float]
        accuracy and f1 score.
    """
    # Evaluation mode.
    model.eval()

    # Model running device.
    device = config.device

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_collate_fn(),
        shuffle=False
    )

    # Record label and prediction for calculating accuracy.
    all_label = []
    all_pred_label = []

    # Evaluate through mini-batch loop.
    mini_batch_iterator = tqdm(dataloader)
    loss = 0
    for text, text_pair, label in mini_batch_iterator:

        # Get `input_ids`, `token_type_ids` and `attention_mask` from via tokenizer.
        batch_encode = tokenizer(
            text=text,
            text_pair=text_pair,
            padding='max_length',
            max_length=config.max_seq_len,
            return_tensors='pt',
            truncation=True
        )
        input_ids = batch_encode['input_ids']
        token_type_ids = batch_encode['token_type_ids']
        attention_mask = batch_encode['attention_mask']

        # Mini-batch prediction.
        pred_label, logits = model.predict(
            input_ids=input_ids.to(device),
            token_type_ids=token_type_ids.to(device),
            attention_mask=attention_mask.to(device)
        )
        pred_label = pred_label.argmax(dim=-1).to('cpu')
        loss += F.cross_entropy(logits, torch.LongTensor(label).to(device))

        all_label.extend(label)
        all_pred_label.extend(pred_label.tolist())

    # Calculate accuracy.
    acc = accuracy_score(all_label, all_pred_label)
    f1 = f1_score(all_label, all_pred_label)

    # Show accuracy.
    mini_batch_iterator.set_description(f'accuracy: {acc:.6f} f1: {f1:.6f}')

    # Release IO resources.
    mini_batch_iterator.close()

    return acc, f1, loss

@torch.no_grad()
def evaluate_matthews_corrcoef(
    config: fine_tune.config.BaseConfig,
    dataset: fine_tune.task.Dataset,
    model: fine_tune.model.Model,
    tokenizer: transformers.PreTrainedTokenizer
) -> float:
    """Evaluate Matthew correlation coefficient.
    Parameters
    ----------
    config : fine_tune.config.BaseConfig
        `fine_tune.config.BaseConfig` subclass which attributes are used
        for experiment setup.
    dataset : fine_tune.task.Dataset
        Task specific dataset.
    model : fine_tune.model.Model
        Model which will be evaluated on `dataset`.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with `model`.
    Returns
    -------
    float
        Matthew correlation coefficient
    """
    # Evaluation mode.
    model.eval()

    # Model running device.
    device = config.device

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_collate_fn(),
        shuffle=False
    )

    # Record label and prediction for calculating accuracy.
    all_label = []
    all_pred_label = []

    # Evaluate through mini-batch loop.
    mini_batch_iterator = tqdm(dataloader)

    loss = 0
    for text, text_pair, label in mini_batch_iterator:

        # Get `input_ids`, `token_type_ids` and `attention_mask` from via tokenizer.
        batch_encode = tokenizer(
            text=text,
            text_pair=text_pair,
            padding='max_length',
            max_length=config.max_seq_len,
            return_tensors='pt',
            truncation=True
        )
        input_ids = batch_encode['input_ids']
        token_type_ids = batch_encode['token_type_ids']
        attention_mask = batch_encode['attention_mask']

        # Mini-batch prediction.
        pred_label, logits = model.predict(
            input_ids=input_ids.to(device),
            token_type_ids=token_type_ids.to(device),
            attention_mask=attention_mask.to(device)
        )
        pred_label = pred_label.argmax(dim=-1).to('cpu')
        loss += F.cross_entropy(logits, torch.LongTensor(label).to(device))

        all_label.extend(label)
        all_pred_label.extend(pred_label.tolist())

    mcc = matthews_corrcoef(all_label, all_pred_label)

    # Show accuracy.
    mini_batch_iterator.set_description(f'Matthews correlation coef: {mcc}')

    # Release IO resources.
    mini_batch_iterator.close()

    # print(f'{all_label=}')
    # print(f'{all_pred_label=}')
    # input()
    print(loss)

    return mcc, loss

@torch.no_grad()
def evaluate_stsb(
    config: fine_tune.config.BaseConfig,
    dataset: fine_tune.task.Dataset,
    model: fine_tune.model.Model,
    tokenizer: transformers.PreTrainedTokenizer    
) -> Tuple:

    # Evaluation mode.
    model.eval()

    # Model running device.
    device = config.device

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_collate_fn(),
        shuffle=False
    )

    # Record ground-truth labels and predictions.
    all_label = []
    all_pred_label = []

    # Evaluate through mini-batch loop.
    mini_batch_iterator = tqdm(dataloader)

    for text, text_pair, label in mini_batch_iterator:

        # Get `input_ids`, `token_type_ids` and `attention_mask` from via tokenizer.
        batch_encode = tokenizer(
            text=text,
            text_pair=text_pair,
            padding='max_length',
            max_length=config.max_seq_len,
            return_tensors='pt',
            truncation=True
        )
        input_ids = batch_encode['input_ids']
        token_type_ids = batch_encode['token_type_ids']
        attention_mask = batch_encode['attention_mask']

        # Mini-batch prediction.
        pred_label = model.predict(
            input_ids=input_ids.to(device),
            token_type_ids=token_type_ids.to(device),
            attention_mask=attention_mask.to(device),
            do_softmax=False
        ).to('cpu')

        all_label.extend(label)
        all_pred_label.extend(pred_label.tolist())

    pcs = pearsonr(all_pred_label, all_label)[0]
    scs = spearmanr(all_pred_label, all_label)[0]
    loss_func = torch.nn.MSELoss()
    loss = loss_func(torch.tensor(all_pred_label), torch.tensor(all_label))

    # Show accuracy.
    mini_batch_iterator.set_description(f'Pearson correlation coef: {pcs}')
    mini_batch_iterator.set_description(f'Spearman correlation coef: {scs}')

    # Release IO resources.
    mini_batch_iterator.close()

    # print(f'{all_label=}')
    # print(f'{all_pred_label=}')
    # input()

    return (pcs, scs, loss)

@torch.no_grad()
def predict_testing_set(
    config: fine_tune.config.BaseConfig,
    dataset: fine_tune.task.Dataset,
    model: fine_tune.model.Model,
    tokenizer: transformers.PreTrainedTokenizer
) -> List[Label]:
    """Generate prediction result of testing set.

    Parameters
    ----------
    config:
        `fine_tune.config.BaseConfig` subclass which attributes are used
            for experiment setup.
        dataset:
            Task specific dataset.
        model:
            Model which will be evaluated on `dataset`.
        tokenizer:
            Tokenizer paired with `model`.

    Returns
    -------
    List[Label]
        A list of `Label` object.
    """
    # Evaluation mode.
    model.eval()

    # Model running device.
    device = config.device

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_collate_fn(),
        shuffle=False
    )

    # Record prediction.
    all_pred_label = []
    # Evaluate through mini-batch loop.
    mini_batch_iterator = tqdm(dataloader)

    for text, text_pair, _ in mini_batch_iterator:

        # Get `input_ids`, `token_type_ids` and `attention_mask` from via tokenizer.
        batch_encode = tokenizer(
            text=text,
            text_pair=text_pair,
            padding='max_length',
            max_length=config.max_seq_len,
            return_tensors='pt',
            truncation=True
        )
        input_ids = batch_encode['input_ids']
        token_type_ids = batch_encode['token_type_ids']
        attention_mask = batch_encode['attention_mask']

        # Mini-batch prediction.
        pred_label, _ = model.predict(
            input_ids=input_ids.to(device),
            token_type_ids=token_type_ids.to(device),
            attention_mask=attention_mask.to(device)
        )
        pred_label = pred_label.argmax(dim=-1).to('cpu')

        all_pred_label.extend(pred_label)

    return [
            fine_tune.task.label_decoder(dataset, label)
            for label in all_pred_label
        ]

@torch.no_grad()
def predict_stsb_testing_set(
    config: fine_tune.config.BaseConfig,
    dataset: fine_tune.task.Dataset,
    model: fine_tune.model.Model,
    tokenizer: transformers.PreTrainedTokenizer
) -> List[Label]:
    """Generate prediction result of testing set.

    Parameters
    ----------
    config:
        `fine_tune.config.BaseConfig` subclass which attributes are used
            for experiment setup.
        dataset:
            Task specific dataset.
        model:
            Model which will be evaluated on `dataset`.
        tokenizer:
            Tokenizer paired with `model`.

    Returns
    -------
    List[Label]
        A list of `Label` object.
    """
    # Evaluation mode.
    model.eval()

    # Model running device.
    device = config.device

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_collate_fn(),
        shuffle=False
    )

    # Record prediction.
    all_pred_label = []
    # Evaluate through mini-batch loop.
    mini_batch_iterator = tqdm(dataloader)

    for text, text_pair, _ in mini_batch_iterator:

        # Get `input_ids`, `token_type_ids` and `attention_mask` from via tokenizer.
        batch_encode = tokenizer(
            text=text,
            text_pair=text_pair,
            padding='max_length',
            max_length=config.max_seq_len,
            return_tensors='pt',
            truncation=True
        )
        input_ids = batch_encode['input_ids']
        token_type_ids = batch_encode['token_type_ids']
        attention_mask = batch_encode['attention_mask']

        # Mini-batch prediction.
        pred_label = model.predict(
            input_ids=input_ids.to(device),
            token_type_ids=token_type_ids.to(device),
            attention_mask=attention_mask.to(device),
            do_softmax=False
        ).to('cpu')

        all_pred_label.extend(pred_label)

    return all_pred_label