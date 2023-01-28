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

## Please cite our paper if you use our code.
- The following citation is under construction.
```
Lin, Y.-J.; Chen, K.-Y.; Kao, H.-Y. LAD: Layer-Wise Adaptive Distillation for BERT Model Compression. Sensors 2023, 1, 0. https://doi.org/
```

## Any questions?
If you have any questions related to the code or the paper, feel free to submit a GitHub issue!👌

## Acknowledgment

Thanks [france5289](https://github.com/france5289) and [ProFatXuanAll](https://github.com/ProFatXuanAll) for their contributions of this paper and the repository.