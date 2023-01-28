# LAD: Layer-wise Adaptive Distillation for BERT Model Compression
This repository is under construction. The paper link will be available when the paepr is officially published.

## Quick introduction
- LAD is a novel framework to improve the efficiency of the task-specific distillation for BERT model compression.
- We design an iterative aggregation mechanism to adaptively distill layer-wise internal knowledge from a BERT teacher model to a student model, enabling an effective knowledge distillation process for a student model without skipping any teacher layers.
- The experimental results on several GLUE tasks show that our proposed method outperforms previous task-specific distillation approaches.

## Installation
```bash
# Clone this project.
git clone https://github.com/IKMLab/LAD.git

# Change to the project directory.
cd LAD

# Create folders to store data and model checkpoints.
mkdir data data/fine_tune data/fine_tune_experiment

# Create your own virtual environment for Python first.
bash install.sh
```

## Prepare GLUE datasets.
```bash
# Take QNLI for example.
export TASK=qnli
bash scripts/prepare_${TASK}.sh
```

## Train the teacher model.
- Before knowledge distillation, you should have a teacher model first.
```bash
# Take QNLI for example.
export TASK=qnli
bash scripts/tune_${TASK}_teacher.sh
```

## Use our trained teacher model
You can download the pre-trained teacher models from [this link](https://drive.google.com/drive/folders/1sa-sHvy8B4-ym3Io57LG1CGYIAykuEfC?usp=share_link).
- Place the downloaded folder (e.g., `teacher_base_bert_qnli`) under `LAD/data/fine_tune_experiment/`

## Folder architecture
```
LAD/
├─ data/
... ├─ fine_tune/
    │  ├─ QNLI/
    │  │  ├─ train.tsv
    │  │  ├─ dev.tsv
    │  │  ├─ test.tsv
    ├─ fine_tune_experiment/
    │  ├─ LAD_6_layer_bert_qnli/
    │  │  ├─ model-13096.pt
    │  │  ├─ config.json
    │  │  ├─ gate_config.json
```

## Train the student model.
```bash
# Take QNLI for example.
export TASK=qnli
bash scripts/${TASK}.sh
```

## Arguments of run_fine_tune.py
The following table gives an explanation of the arguments listed above.  
If you want to view the details of each arguments (e.g., dropout rate, weight decay, maximum gradient norm and so on), type `python run_fine_tune.py -h` in your terminal.

| Arguments     | Description                                                                                                                                                                |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `experiment`  | Name of the experiment to create folders to  save model checkpoints, config files, and tensorboard logging files. It would help if you gave each experiment a unique name. |
| `ptrain_ver`  | Specify which pre-trained BERT model you want to fine-tune. We support the cased and uncased models of `bert-base` and `bert-large`.                                       |
| `task`        | Specify the downstream task you want to solve. It should be a lower case string.                                                                                           |
| `dataset`     | Specify the dataset of a downstream task. For example, `train` refers to the training set and `dev` refers to the development set.                                         |
| `num_class`   | Number of classes to classify. `MNLI` is a classification task with `3` labels, so we set it to `3`.                                                                                |
| `accum_step`  | Gradient accumulation step.                                                                                                                                                |
| `batch_size`  | Training batch size.                                                                                                                                                       |
| `ckpt_step`   | Checkpoint save interval.                                                                                                                                                  |
| `log_step`    | Logging interval of `tensorboard`.                                                                                                                                         |
| `lr`          | Optimizer `torch.optim.AdamW`'s learning rate.                                                                                                                             |
| `max_seq_len` | Maximum input sequence length for a model.                                                                                                                                 |
| `device_id`   | Device ID of a model. `-1` refers to `CPU` and other values which is greater and equal to `0` means other CUDA devices.                                                     |
| `seed`        | Random seed of each experiment.                                                                                                                                            |
| `total_step`  | Total number of step to perform training. `step= (#data instances) // batch_size`.                                                                                         |
| `warmup_step` | Linear scheduler warmup step.                                                                                                                                              |

## Arguments of run_lad_distil_epoch.py
The following table gives an explanation of the arguments listed above.  
If you want to view the details of each arguments (e.g., dropout rate, weight decay, maximum gradient norm, number of attention heads and so on), type `python run_lad_distil_epoch.py -h` in your terminal.

| Arguments           | Description                                                                                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `teacher_exp`       | Experiment name of the fine-tuned teacher model. It is used to find the path of the teacher model's checkpoint.                                                             |
| `tckpt`             | Checkpoint of the teacher model.                                                                                                                                          |
| `experiment`        | Name of the experiment to create folders to save model checkpoints, config files, and tensorboard logging files. It would help if you gave each experiment a unique name. |
| `task`              | Specify the downstream task you want to solve. It should be a lower case string.                                                                                          |
| `batch_size`        | Training batch size.                                                                                                                                                      |
| `ckpt_step`         | Checkpoint save interval.                                                                                                                                                 |
| `log_step`          | Logging interval of `tensorboard`.                                                                                                                                        |
| `d_ff`              | Dimension of the feed forward layers of the Transformer.                                                                                                                  |
| `d_model`           | Hidden dimension of the Transformer layers.                                                                                                                               |
| `lr`                | Optimizer `torch.optim.AdamW`'s learning rate.                                                                                                                            |
| `gate_lr`           | Gate Network's optimizer: `torch.optim.AdamW`'s learning rate.                                                                                                            |
| `num_hidden_layers` | Number of Transformer layers.                                                                                                                                             |
| `epoch`        | Total number of training steps. `training_steps = round(#data instances / batch_size * epoch)`.                                                                                        |
| `warmup_rate`       | The ratio of Linear scheduler warmup steps. `warmup_steps = training_steps * warmup_rate`                                                                                                            |
| `device_id`         | Device ID of the student model. `-1` refers to `CPU` and other values which is greater and equal to `0` means other CUDA devices.                                         |
| `tdevice_id`        | Device ID of the teacher model. `-1` refers to `CPU` and other values which is greater and equal to `0` means other CUDA devices.                                         |
| `gate_device_id`    | Device ID of the Gate Network. `-1` refers to `CPU` and other values which is greater and equal to `0` means other CUDA devices.                                          |
| `seed`              | Random seed of each experiment.                                                                                                                                           |
| `softmax_temp`      | Softmax temperature.                                                                                                                                                      |
| `soft_weight`       | Weight of the soft target loss.                                                                                                                                           |
| `hard_weight`       | Weight of the hard target loss.                                                                                                                                           |
| `hidden_mse_weight` | Weight of the MSE loss of the hidden states.                                                                                                                              |

## Please cite our paper if you use our code.
- The following citation is under construction.
```
Lin, Y.-J.; Chen, K.-Y.; Kao, H.-Y. LAD: Layer-Wise Adaptive Distillation for BERT Model Compression. Sensors 2023, 1, 0. https://doi.org/
```

## Any questions?
If you have any questions related to the code or the paper, feel free to submit a GitHub issue!👌

## Acknowledgment

Thanks [france5289](https://github.com/france5289) and [ProFatXuanAll](https://github.com/ProFatXuanAll) for their contributions of this paper and the repository.