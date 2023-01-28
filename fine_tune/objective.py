r"""Fine-tune distillation objective.

Usage:
    loss = soft_target_loss(...)
    loss = distill(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd party modules

import torch
import torch.nn
import torch.nn.functional as F

def soft_target_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        is_regression=False
) -> torch.Tensor:
    r"""Soft-target distillation loss (KL divergence) function.

    Args:
        student_logits:
            Student model's output unnormalized logits with numeric type
            `torch.float32` and size (B, C).
        teacher_logits:
            Teacher model's output unnormalized logits with numeric type
            `torch.float32` and size (B, C).

    Returns:
        Soft-target KL divergence loss.
        Hinton, G. (2014). Distilling the Knowledge in a Neural Network use cross entropy.
        Here we follow recent KD paper, use KL divergence loss.
    """
    if is_regression:
        soft_loss = F.mse_loss(student_logits, teacher_logits)
    else:
        soft_loss = torch.nn.KLDivLoss(reduction="batchmean")(
                        F.log_softmax(student_logits, dim=1),
                        F.softmax(teacher_logits, dim=1)
                    )
    return soft_loss


def distill_loss(
        hard_target: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        gamma: float = 0.8,
        alpha: float = 0.2,
        softmax_temp: float = 1.0,
        is_regression=False
) -> torch.Tensor:
    r"""Knowledge distillation loss function.

    We use the following notation for the rest of the context.
        - B: batch size.
        - C: number of class.
        - P: teacher model's output softmax probability.
        - P_i: teacher model's output softmax probability on class i.
        - p: teacher model's output unnormalized logits.
        - p_i: teacher model's output unnormalized logits on class i.
        - Q: student model's output softmax probability.
        - Q_i: student model's output softmax probability on class i.
        - q: student model's output unnormalized logits.
        - q_i: student model's output unnormalized logits on class i.
        - y: Ground truth class index.

    We first convert logits into prediction using softmax normalization:
        $$
        P_i = \text{softmax}(p_i) = \frac{e^{p_i}}{\sum_{j=1}^C e^{p_j}} \\
        Q_i = \text{softmax}(q_i) = \frac{e^{q_i}}{\sum_{j=1}^C e^{q_j}}
        $$

    Then we calculate soft-target cross-entropy using following formula:
        $$
        -P * \log Q = -\sum_{i = 1}^c P_i * \log Q_i
        $$

    Then we calculate hard-target cross-entropy using following formula:
        $$
        -\log Q_y
        $$

    Finally we add hard-target and soft-target together as total loss.

    Args:
        hard_target:
            Actual label for cross-entropy loss with numeric type `torch.int64`
            and size (B).
        student_logits:
            Student model's output unnormalized logits with numeric type
            `torch.float32` and size (B, C).
        teacher_logits:
            Teacher model's output unnormalized logits with numeric type
            `torch.float32` and size (B, C).
        gamma (optional):
            loss weight of hard target cross entropy, default 0.8
        alpha (optional):
            loss weight of soft target cross entropy.
        softmax_temp (optional):
            softmax temperature, it will apply to both student and teacher logits.
    Returns:
        Hard-target + soft-target cross-entropy loss. See Hinton, G. (2014).
        Distilling the Knowledge in a Neural Network.
    """
    if is_regression:
        hard_target_loss = F.mse_loss(student_logits.squeeze(-1), hard_target)
        return (
            gamma * hard_target_loss + \
            alpha * soft_target_loss(student_logits, teacher_logits, is_regression)
        )
    else:
        return (
            gamma * F.cross_entropy(student_logits, hard_target) +
            alpha * soft_target_loss(student_logits / softmax_temp, teacher_logits / softmax_temp) * pow(softmax_temp, 2)
        )

def hidden_MSE_loss(
        teacher_hidden: torch.Tensor,
        student_hidden: torch.Tensor,
        mu: int = 100,
) -> torch.Tensor:
    r""" MSE loss between teacher's and student's hidden states.
    We use the following notation for the reset of the context.
        - B: batch size.
        - S: sequence size.
        - H: hidden size.
    Args:
        teacher_hidden:
            hidden state from one of teacher layer with numeric type
            `torch.float32` and size (B, S, H)
        student_hidden:
            hidden state from one of student layer with numeric type
            `torch.float32` and size (B, S, H)
        mu (optional):
            scaling factor of hidden MSE loss.
    Returns:
        MSE loss between teacher and student hidden states.
    """
    return mu * F.mse_loss(student_hidden, teacher_hidden)
