r"""Run fine-tune training.

Usage:
    python run_fine_tune.py ...

Run `python run_fine_tune.py -h` for help, or see 'doc/fine_tune_*.md' for more
information.
"""

# built-in modules

import argparse
import logging

# my own modules

import fine_tune

# Get main logger.
logger = logging.getLogger('fine_tune.train')
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
        help='Name of the current fine-tune experiment.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--ptrain_ver',
        help='Pretrained model version provided by `transformers` package.',
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
    parser.add_argument(
        '--num_class',
        default=2,
        help='Number of classes to classify.',
        required=True,
        type=int,
    )

    # Optional parameters.
    parser.add_argument(
        '--model',
        help='Name of the model to fine-tune.',
        default='bert',
        type=str,
    )
    parser.add_argument(
        '--accum_step',
        default=1,
        help='Gradient accumulation step.',
        type=int,
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        help='Training batch size.',
        type=int,
    )
    parser.add_argument(
        '--beta1',
        default=0.9,
        help="Optimizer `torch.optim.AdamW`'s beta coefficients.",
        type=float,
    )
    parser.add_argument(
        '--beta2',
        default=0.999,
        help="Optimizer `torch.optim.AdamW`'s beta coefficients.",
        type=float,
    )
    parser.add_argument(
        '--ckpt_step',
        default=1000,
        help='Checkpoint save interval.',
        type=int,
    )
    parser.add_argument(
        '--dropout',
        default=0.1,
        help='Dropout probability.',
        type=float,
    )
    parser.add_argument(
        '--device_id',
        help='Device ID of model.',
        default=0,
        type=int,
    )
    parser.add_argument(
        '--eps',
        default=1e-8,
        help="Optimizer `torch.optim.AdamW`'s epsilon.",
        type=float,
    )
    parser.add_argument(
        '--log_step',
        default=500,
        help='Logging interval.',
        type=int,
    )
    parser.add_argument(
        '--lr',
        default=3e-5,
        help="Optimizer `torch.optim.AdamW`'s learning rate.",
        type=float,
    )
    parser.add_argument(
        '--max_norm',
        default=1.0,
        help='Maximum norm of gradient.',
        type=float,
    )
    parser.add_argument(
        '--max_seq_len',
        default=512,
        help='Maximum input sequence length for fine-tune model.',
        type=int,
    )
    parser.add_argument(
        '--seed',
        default=42,
        help='Control random seed.',
        type=int,
    )
    parser.add_argument(
        '--total_step',
        default=50000,
        help='Total number of step to perform training.',
        type=int,
    )
    parser.add_argument(
        '--warmup_step',
        default=10000,
        help='Linear scheduler warmup step.',
        type=int,
    )
    parser.add_argument(
        '--weight_decay',
        default=0.01,
        help="Optimizer AdamW's parameter `weight_decay`.",
        type=float,
    )

    # Parse arguments.
    args = parser.parse_args()

    # Construct configuration.
    config = fine_tune.config.TeacherConfig(
        accum_step=args.accum_step,
        batch_size=args.batch_size,
        beta1=args.beta1,
        beta2=args.beta2,
        ckpt_step=args.ckpt_step,
        dataset=args.dataset,
        dropout=args.dropout,
        eps=args.eps,
        experiment=args.experiment,
        log_step=args.log_step,
        lr=args.lr,
        max_norm=args.max_norm,
        max_seq_len=args.max_seq_len,
        model=args.model,
        num_class=args.num_class,
        ptrain_ver=args.ptrain_ver,
        seed=args.seed,
        task=args.task,
        total_step=args.total_step,
        warmup_step=args.warmup_step,
        weight_decay=args.weight_decay,
        device_id=args.device_id,
    )

    # Log configuration.
    logger.info(config)

    # Save configuration.
    config.save()

    # Control random seed for reproducibility.
    fine_tune.util.set_seed_by_config(
        config=config
    )

    # Load fine-tune dataset.
    dataset = fine_tune.util.load_dataset_by_config(
        config=config
    )

    # Load tokenizer.
    tokenizer = fine_tune.util.load_teacher_tokenizer_by_config(
        config=config
    )

    # Load model.
    model = fine_tune.util.load_teacher_model_by_config(
        config=config
    )

    # Load optimizer.
    optimizer = fine_tune.util.optimizer.load_optimizer_by_config(
        config=config,
        model=model
    )

    # Load scheduler.
    scheduler = fine_tune.util.scheduler.load_scheduler_by_config(
        config=config,
        optimizer=optimizer
    )

    fine_tune.util.train_teacher(
        config=config,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer
    )
