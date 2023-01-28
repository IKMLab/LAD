r"""Helper functions for training model.

Usage:
    import fine_tune

    fine_tune.util.train(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

# 3rd party modules

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
import transformers

from tqdm import tqdm

# my own modules

import fine_tune.config
import fine_tune.task
import fine_tune.model
import fine_tune.path


def train_teacher(
        config: fine_tune.config.BaseConfig,
        dataset: fine_tune.task.Dataset,
        model: fine_tune.model.Model,
        optimizer: torch.optim.AdamW,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        tokenizer: transformers.PreTrainedTokenizer,
):
    r"""Fine-tune or distill model on task specific dataset.

    Args:
        config:
            `fine_tune.config.BaseConfig` subclass which attributes are used
            for experiment setup.
        dataset:
            Task specific dataset.
        model:
            Model which will be fine-tuned on `dataset`.
        optimizer:
            `torch.optim.AdamW` optimizer.
        schduler:
            Linear warmup scheduler provided by `transformers` package.
        tokenizer:
            Tokenizer paired with `model`.
    """
    # Training mode.
    model.train()

    # Model running device.
    device = config.device

    # Clean all gradient.
    optimizer.zero_grad()

    # Get experiment name and path.
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=config.experiment,
        model=config.model,
        task=config.task
    )
    experiment_dir = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name
    )

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size // config.accum_step,
        collate_fn=dataset.create_collate_fn(),
        num_workers=os.cpu_count(),
        shuffle=True
    )

    # Create tensorboard's `SummaryWriter`.
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(
            fine_tune.path.LOG,
            experiment_name
        )
    )
    if config.task == 'stsb':
        objective = nn.MSELoss()
    else:
        # Use cross-entropy as objective.
        objective = nn.CrossEntropyLoss()

    # Step and accumulation step counter.
    step = 0
    accum_step = 0
    total_accum_step = config.total_step * config.accum_step

    # Mini-batch loss and accumulate loss.
    # Update when accumulate to `config.batch_size`.
    loss = 0
    accum_loss = 0

    # `tqdm` CLI Logger. We will manually update progress bar.
    cli_logger = tqdm(
        desc=f'loss: {loss:.6f}',
        total=config.total_step
    )

    # Total update times: `config.total_step`.
    while accum_step < total_accum_step:

        # Mini-batch loop.
        for (
                text,
                text_pair,
                label
        ) in dataloader:

            if config.task == 'stsb':
                label = torch.tensor(label, dtype=torch.float32)
            else:
                # Transform `label` to tensor.
                label = torch.LongTensor(label)

            # Get `input_ids`, `token_type_ids` and `attention_mask` via tokenizer.
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

            # Accumulate cross-entropy loss.
            # Use `model(...)` to do forward pass.
            logits, _ = model(
                    input_ids=input_ids.to(device),
                    token_type_ids=token_type_ids.to(device),
                    attention_mask=attention_mask.to(device)
                )
            if config.task == 'stsb':
                logits = logits.squeeze(-1)
            accum_loss = objective(
                input=logits,
                target=label.to(device)
            ) / config.accum_step

            # Mini-batch cross-entropy loss. Only used as log.
            loss += accum_loss.item()

            # Backward pass accumulation loss.
            accum_loss.backward()

            # Increment accumulation step.
            accum_step += 1

            # Perform gradient descend when achieve actual mini-batch size.
            if accum_step % config.accum_step == 0:
                # Gradient clipping.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_norm
                )

                # Gradient descend.
                optimizer.step()

                # Update learning rate.
                scheduler.step()

                # Log on CLI.
                cli_logger.update()
                cli_logger.set_description(
                    f'loss: {loss:.6f}'
                )

                # Increment actual step.
                step += 1

                # Log loss and learning rate for each `config.log_step` step.
                if step % config.log_step == 0:
                    writer.add_scalar(
                        f'{config.task}/{config.dataset}/loss',
                        loss,
                        step
                    )
                    writer.add_scalar(
                        f'{config.task}/{config.dataset}/lr',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        step
                    )

                # Clean up mini-batch loss.
                loss = 0

                # Clean up gradient.
                optimizer.zero_grad()

                # Save model for each `config.ckpt_step` step.
                if step % config.ckpt_step == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(experiment_dir, f'model-{step}.pt')
                    )

            # Stop training condition.
            if accum_step >= total_accum_step:
                break

    # Release IO resources.
    writer.flush()
    writer.close()
    cli_logger.close()

    # Save the lastest model.
    torch.save(
        model.state_dict(),
        os.path.join(experiment_dir, f'model-{step}.pt')
    )
