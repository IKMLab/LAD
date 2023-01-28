r"""Generate prediction result file of testing set.
"""

# built-in modules

import argparse
import logging
import os

# 3rd-party modules

import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm

# my own modules

import fine_tune


def gen_predictions(args, best_ckpt):
    # Get main logger.
    logger = logging.getLogger('fine_tune.gen_test_predict')
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO
    )

    # Filter out message not begin with name 'fine_tune'.
    for handler in logging.getLogger().handlers:
        handler.addFilter(logging.Filter('fine_tune'))
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

    # Set prediction dataset.
    config.dataset = 'test'

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

    # Load model state dict.
    logger.info("Load model from ckpt: %s", best_ckpt)
    model.load_state_dict(
        torch.load(
            os.path.join(experiment_dir, f'model-{best_ckpt}.pt'),
            map_location=config.device
        )
    )


    logger.info("Start to make prediction")
    if args.task == 'stsb':
        all_pred = fine_tune.util.predict_stsb_testing_set(
            config=config,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer
        )
    else:
        all_pred = fine_tune.util.predict_testing_set(
            config=config,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer
        )

    if args.task.lower() != 'mnli':
        if args.task.lower() == 'stsb':
            filename = os.path.join(
                experiment_dir,
                f'STS-B_{best_ckpt}.tsv'
            )
        elif args.task.lower() != 'sst2':
            filename = os.path.join(
                experiment_dir,
                f'{args.task.upper()}_{best_ckpt}.tsv'
            )
        else:
            filename = os.path.join(
                experiment_dir,
                'SST-2.tsv'
            )
    else:
        if "mismatched" in args.dataset.lower():
            filename = os.path.join(
                experiment_dir,
                f'{args.task.upper()}-mm.tsv'
            )
        else:
            filename = os.path.join(
                experiment_dir,
                f'{args.task.upper()}-m.tsv'
            )

    logger.info("Write result to %s", filename)

    if args.task == 'stsb':
        with open(filename, 'w') as f:
            f.write("index\tprediction\n")
            for idx, pred in enumerate(tqdm(all_pred)):
                if pred > 5:
                    pred = 5
                f.write(f"{idx}\t{pred}\n")        
    else:
        with open(filename, 'w') as f:
            f.write("index\tprediction\n")
            for idx, pred in enumerate(tqdm(all_pred)):
                f.write(f"{idx}\t{pred}\n")

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
        '--ckpt',
        help='Start generate prediction from specified checkpoint',
        required=True,
        type=int
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
        '--model',
        help='Name of the model to distill.',
        default='bert',
        type=str,
    )

    # Parse arguments.
    args = parser.parse_args()

    if args.device_id is not None:
        logger.info("Use device: %s to run evaluation", args.device_id)

    gen_predictions(args, args.ckpt)
