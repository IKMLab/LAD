r"""Run fine-tune evaluation.

Usage:
    python run_fine_tune_eval.py ...

Run `python run_fine_tune_eval.py -h` for help, or see 'doc/fine_tune_*.md' for
more information.
"""

# built-in modules

import argparse
import logging
import os
import re

# 3rd-party modules

import torch
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
import pandas as pd
import numpy as np

# my own modules

import fine_tune

def write_csv(config, checkpoint, max_value, do_write):
    if do_write:
        csv_path = f'data/fine_tune_experiment/{config.task}.csv'
        column_names = list(dict(config).keys())
        column_names.extend(["checkpoint", "max_value" ])
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns=column_names)
            df.to_csv(csv_path, index=None)

        df = pd.read_csv(csv_path)
        values = list(dict(config).values())
        values.extend([checkpoint, max_value])
        print(values)
        df = df.append(
            pd.Series(values, index=column_names),
            ignore_index=True
        )
        df.to_csv(csv_path, index=None)
    else:
        pass

# Get main logger.
logger = logging.getLogger('fine_tune.eval')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO
)

# Filter out message not begin with name 'fine_tune'.
for handler in logging.getLogger().handlers:
    handler.addFilter(logging.Filter('fine_tune'))

if __name__ == '__main__':
    # Parse arguments from STDIN.
    parser = argparse.ArgumentParser()

    # Required parameters.
    parser.add_argument(
        '--experiment',
        help='Name of the previous experiment to evalutate.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--task',
        help='Name of the fine-tune task.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--dataset',
        help='Dataset name of the fine-tune task.',
        required=True,
        type=str,
    )

    # Optional parameters.
    parser.add_argument(
        '--batch_size',
        default=32,
        help='Evaluation batch size.',
        type=int,
    )
    parser.add_argument(
        '--device_id',
        default=None,
        help='Run evaluation on dedicated device.',
        type=int,
    )
    parser.add_argument(
        '--ckpt',
        default=0,
        help='Start evaluation from specified checkpoint',
        type=int
    )
    parser.add_argument(
        '--model',
        help='Name of the model to distill.',
        default='bert',
        type=str,
    )
    parser.add_argument(
        '--write_csv',
        action="store_true"
    )
    parser.add_argument(
        '--do_predict',
        action="store_true"
    )

    # Parse arguments.
    args = parser.parse_args()

    # Load fine-tune teacher model configuration.
    # `fine_tune.config.TeacherConfig.load` will trigger `TypeError` if the
    # actual configuration file is saved by `fine_tune.config.StudentConfig`.
    if args.device_id is not None:
        logger.info("Use device: %s to run evaluation", args.device_id)
    try:
        config = fine_tune.config.TeacherConfig.load(
            experiment=args.experiment,
            model=args.model,
            task=args.task,
            device_id=args.device_id
        )
    # Load fine-tune distillation student model configuration.
    except TypeError:
        config = fine_tune.config.StudentConfig.load(
            experiment=args.experiment,
            model=args.model,
            task=args.task,
            device_id=args.device_id
        )

    # Change batch size for faster evaluation.
    if args.batch_size:
        config.batch_size = args.batch_size

    # Set evaluation dataset.
    config.dataset = args.dataset

    # Log configuration.
    logger.info(config)

    # Control random seed for reproducibility.
    fine_tune.util.set_seed_by_config(
        config=config
    )

    # Load validation/development dataset.
    dataset = fine_tune.util.load_dataset_by_config(
        config=config
    )

    # Load teacher tokenizer and model.
    if isinstance(config, fine_tune.config.TeacherConfig):
        tokenizer = fine_tune.util.load_teacher_tokenizer_by_config(
            config=config
        )
        model = fine_tune.util.load_teacher_model_by_config(
            config=config
        )
    # Load student tokenizer and model.
    else:
        tokenizer = fine_tune.util.load_student_tokenizer_by_config(
            config=config
        )
        model = fine_tune.util.load_student_model_by_config(
            config=config,
            tokenizer=tokenizer
        )

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

    # Get all checkpoint file names.
    ckpt_pattern = r'model-(\d+)\.pt'
    all_ckpts = sorted(map(
        lambda file_name: int(re.match(ckpt_pattern, file_name).group(1)),
        filter(
            lambda file_name: re.match(ckpt_pattern, file_name),
            os.listdir(experiment_dir)
        ),
    ))

    # Filt unnecessary checkpoint.
    all_ckpts = list(filter(lambda ckpt: ckpt >= args.ckpt, all_ckpts))

    # Create tensorboard's `SummaryWriter`.
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(
            fine_tune.path.LOG,
            experiment_name
        )
    )

    # Record maximum accuracy or f1 score and its respective checkpoint.
    # For tasks except CoLA and STS-B
    max_acc = 0.0
    max_acc_ckpt = 0
    max_f1 = 0.0
    max_f1_ckpt = 0

    if config.task == 'cola':
        scores = {
            "mcc": [],
            "ckpt": [],
        }

    elif config.task == 'stsb':
        scores = {
            "pcs": [],
            "scs": [],
            "ckpt": [],
        } 

    else:
        scores = {
            "f1": [],
            "acc": [],
            "ckpt": [],
        }
    results = {}
    # Evaluate every checkpoints.
    for i, ckpt in enumerate(all_ckpts):
        logger.info("Load from ckpt: %s", ckpt)
        results.setdefault('ckpt', []).append(ckpt)
        # Clean all gradient.
        model.zero_grad()

        # Load model from checkpoint.
        model.load_state_dict(
            torch.load(
                os.path.join(experiment_dir,f'model-{ckpt}.pt'),
                map_location=config.device
            )
        )
        if config.task == 'cola':
            mcc, loss = fine_tune.util.evaluate_matthews_corrcoef(
                config=config,
                dataset=dataset,
                model=model,
                tokenizer=tokenizer
            )
            results.setdefault('mcc', []).append(mcc)
            results.setdefault('loss', []).append(loss.item())

            # if max_mcc <= mcc:
            #     max_mcc = mcc
            #     max_mcc_ckpt = ckpt

            # if i == 0:
            #     min_loss = loss
            #     min_loss_ckpt = ckpt
            # else:
            #     if loss < min_loss:
            #         min_loss = loss
            #         min_loss_ckpt = ckpt

            writer.add_scalar(
                f'{config.task}/{config.dataset}/MCC',
                mcc,
                ckpt
            )
            writer.add_scalar(
                f'{config.task}/{config.dataset}/val_loss',
                loss,
                ckpt
            )
        elif config.task == 'stsb':
            pcs, scs, loss = fine_tune.util.evaluate_stsb(
                config=config,
                dataset=dataset,
                model=model,
                tokenizer=tokenizer                
            )
            results.setdefault('pcs', []).append(pcs)
            results.setdefault('scs', []).append(scs)
            results.setdefault('loss', []).append(loss.item())
            # if max_pcs <= pcs:
            #     max_pcs = pcs
            #     max_pcs_ckpt = ckpt
            # if max_scs <= scs:
            #     max_scs = scs
            #     max_scs_ckpt = ckpt
            # if i == 0:
            #     min_loss = loss
            #     min_loss_ckpt = ckpt
            # else:
            #     if loss < min_loss:
            #         min_loss = loss
            #         min_loss_ckpt = ckpt

            writer.add_scalar(
                f'{config.task}/{config.dataset}/pcs', pcs, ckpt
            )
            writer.add_scalar(
                f'{config.task}/{config.dataset}/scs', scs, ckpt
            )
            writer.add_scalar(
                f'{config.task}/{config.dataset}/val_loss',
                loss,
                ckpt
            )
        else:
            acc, f1, loss = fine_tune.util.evaluate_acc_and_f1(
                config=config,
                dataset=dataset,
                model=model,
                tokenizer=tokenizer
            )
            results.setdefault('f1', []).append(f1)
            results.setdefault('acc', []).append(acc)
            results.setdefault('loss', []).append(loss.item())

            if config.task == 'qqp' or config.task == 'mrpc':
                writer.add_scalar(
                    f'{config.task}/{config.dataset}/accuracy',
                    acc,
                    ckpt
                )
                writer.add_scalar(
                    f'{config.task}/{config.dataset}/f1_score',
                    f1,
                    ckpt
                )
                writer.add_scalar(
                    f'{config.task}/{config.dataset}/val_loss',
                    loss,
                    ckpt
                )
            else:
                writer.add_scalar(
                    f'{config.task}/{config.dataset}/accuracy',
                    acc,
                    ckpt
                )
                writer.add_scalar(
                    f'{config.task}/{config.dataset}/val_loss',
                    loss,
                    ckpt
                )

    # Release IO resources.
    writer.flush()
    writer.close()
    min_losses = sorted(results['loss'])[:3]
    logger.info(f'Top3 smallest losses:  {min_losses}')
    min_loss_indice = np.argsort(results['loss'])[:3]
    min_loss_ckpt = [results['ckpt'][i] for i in min_loss_indice]
    logger.info(f'Checkpoints of top3 smallest losses: {min_loss_ckpt}')

    if config.task == 'qqp' or config.task == 'mrpc':
        max_acc = sorted(results['acc'])[::-1][:3]
        max_acc_indice = np.argsort(results['acc'])[::-1][:3]
        max_acc_ckpt = [results['ckpt'][i] for i in max_acc_indice]
        logger.info(f'Top3 accuracy:            {max_acc}')
        logger.info(f'Top3 accuracy checkpoints: {max_acc_ckpt}')

        max_f1 = sorted(results['f1'])[::-1][:3]
        max_f1_indice = np.argsort(results['f1'])[::-1][:3]
        max_f1_ckpt = [results['ckpt'][i] for i in max_f1_indice]
        logger.info(f'Top3 F1: {max_f1}')
        logger.info(f'Top3 f1 score checkpoints: {max_f1_ckpt}')
        write_csv(config, [max_f1_ckpt, max_acc_ckpt], [max_f1, max_acc], args.write_csv)

    elif config.task == 'cola':
        max_mcc = sorted(results['mcc'])[::-1][:3]
        max_mcc_indice = np.argsort(results['mcc'])[::-1][:3]
        max_mcc_ckpt = [results['ckpt'][i] for i in max_mcc_indice]
        logger.info(f'Top3 mcc:             {max_mcc}')
        logger.info(f'Top3 mcc checkpoints: {max_mcc_ckpt}')
        write_csv(config, max_mcc_ckpt, max_mcc, args.write_csv)

    elif config.task == 'stsb':
        logger.info('max Pearson :            %f', max_pcs)
        logger.info('max Pearson checkpoint:  %d', max_pcs_ckpt)
        logger.info('max Spearman :           %f', max_scs)
        logger.info('max Spearman checkpoint: %d', max_scs_ckpt)
        write_csv(config, [max_pcs_ckpt, max_scs_ckpt], [max_pcs, max_scs], args.write_csv)

    else:
        max_acc = sorted(results['acc'])[::-1][:3]
        max_acc_indice = np.argsort(results['acc'])[::-1][:3]
        max_acc_ckpt = [results['ckpt'][i] for i in max_acc_indice]
        logger.info(f'Top3 accuracy:             {max_acc}')
        logger.info(f'Top3 accuracy checkpoints: {max_acc_ckpt}')
        write_csv(config, max_acc_ckpt, max_acc, args.write_csv)

    if args.do_predict:
        from generate_test_prediction import gen_predictions

        if config.task == 'mrpc':
            for best_ckpt in max_acc_ckpt:
                gen_predictions(args, best_ckpt)

        elif config.task == 'qqp':
            gen_predictions(args, max_f1_ckpt)

        elif config.task == 'cola':
            gen_predictions(args, max_mcc_ckpt)

        elif config.task == 'stsb':
            gen_predictions(args, max_scs_ckpt)

        else:
            for best_ckpt in max_acc_ckpt:
                gen_predictions(args, best_ckpt)