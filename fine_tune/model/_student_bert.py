r"""Fine-tune BERT student models.

Usage:
    import fine_tune

    model = fine_tune.model.StudentBert(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from typing import List

# 3rd party modules

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertModel

# Get main logger.
logger = logging.getLogger('fine_tune.model.student_bert')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO
)

# Filter out message not begin with name 'fine_tune'.
for handler in logging.getLogger().handlers:
    handler.addFilter(logging.Filter('fine_tune'))

class StudentBert(nn.Module):
    r"""Fine-tune distillation student model based on BERT.

    Args:
        dropout:
            Dropout probability.
        d_ff:
            BERT layers feed forward dimension.
        d_model:
            BERT layers hidden dimension.
        max_seq_len:
            Maximum input sequence length.
        num_attention_heads:
            Number of attention heads in BERT layers.
        num_class:
            Number of classes to classify.
        num_hidden_layers:
            Number of BERT layers.
        type_vocab_size:
            BERT's token type embedding range.
        vocab_size:
            Vocabulary size for BERT's embedding range.
        init_from_pretrained:
            Use pre-trained teacher weight to init student model.
    """

    def __init__(
            self,
            d_ff: int,
            d_model: int,
            dropout: float,
            max_seq_len: int,
            num_attention_heads: int,
            num_class: int,
            num_hidden_layers: int,
            type_vocab_size: int,
            vocab_size: int
    ):
        super().__init__()

        # Construct BERT model.
        self.encoder = BertModel(BertConfig(
            attention_probs_dropout_prob=dropout,
            gradient_checkpointing=False,
            hidden_dropout_prob=dropout,
            hidden_size=d_model,
            initializer_range=0.02,
            intermediate_size=d_ff,
            layer_norm_eps=1e-12,
            max_position_embeddings=max_seq_len,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            type_vocab_size=type_vocab_size,
            vocab_size=vocab_size,
            return_dict=True
        ))

        # Dropout layer between encoder and linear layer.
        self.dropout = nn.Dropout(dropout)

        # Linear layer project from `d_model` into `num_class`.
        self.linear_layer = nn.Linear(
            in_features=d_model,
            out_features=num_class
        )

        # Linear layer initialization.
        with torch.no_grad():
            nn.init.normal_(
                self.linear_layer.weight,
                mean=0.0,
                std=0.02
            )
            nn.init.zeros_(self.linear_layer.bias)

    def init_from_pre_trained(
        self,
        teacher_indices: List[int],
        student_indices: List[int] = None,
        pretrain_ver: str = 'bert-base-uncased'
    ):
        """Given pre-trained model version and
        indices of the Transformer layers of the teacher and the student.
        Initialize model parameters from the given pre-trained model.

        Parameters
        ----------
        teacher_indices : List[int]
            Specify teacher layer to load pre-trained weight for initialization.
        student_indices : List[int], optional
            Specify which student layer will be initialized.
            If not given, the even student layer will be initialized, by default None
        pretrain_ver : str, optional
            pre-trained version string, by default 'bert-base-uncased'
        """
        pretrain_model = BertModel.from_pretrained(pretrain_ver)

        new_state_dict = {}

        keys = [
            'attention.self.query.weight',
            'attention.self.query.bias',
            'attention.self.key.weight',
            'attention.self.key.bias',
            'attention.self.value.weight',
            'attention.self.value.bias',
            'attention.output.dense.weight',
            'attention.output.dense.bias',
            'attention.output.LayerNorm.weight',
            'attention.output.LayerNorm.bias',
            'intermediate.dense.weight',
            'intermediate.dense.bias',
            'output.dense.weight',
            'output.dense.bias',
            'output.LayerNorm.weight',
            'output.LayerNorm.bias'
        ]

        if student_indices is None:
            for i, t_index in enumerate(teacher_indices):
                for key in keys:
                    new_state_dict.update(
                        {
                            f'encoder.layer.{i}.{key}':
                            pretrain_model.state_dict()[f'encoder.layer.{t_index}.{key}']
                        }
                    )

        else:
            logger.info("Use student and teacher indicies to init relative student layers")
            for s_index, t_index in zip(student_indices, teacher_indices):
                for key in keys:
                    new_state_dict.update(
                        {
                            f'encoder.layer.{s_index}.{key}':
                            pretrain_model.state_dict()[f'encoder.layer.{t_index}.{key}']
                        }
                    )

        new_state_dict.update(
            {
                'pooler.dense.weight':pretrain_model.state_dict()['pooler.dense.weight'],
                'pooler.dense.bias':pretrain_model.state_dict()['pooler.dense.bias']
            }
        )

        del pretrain_model
        logger.info("Load model state dict from pre-trained model")
        self.encoder.load_state_dict(new_state_dict, strict=False)
        logger.info("Finish initialization from pre-trained model")

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            return_hidden_and_attn: bool = False
    ):
        r"""Forward pass

        We use the following notation for the rest of the context.
            - A: num of attention heads.
            - B: batch size.
            - S: sequence length.
            - C: number of class.
            - H: hidden state size.

        Args:
            input_ids:
                Batch of input token ids. `input_ids` is a `torch.Tensor` with
                numeric type `torch.int64` and size (B, S).
            attention_mask:
                Batch of input attention masks. `attention_mask` is a
                `torch.Tensor` with numeric type `torch.float32` and size
                (B, S).
            token_type_ids:
                Batch of input token type ids. `token_type_ids` is a
                `torch.Tensor` with numeric type `torch.int64` and size (B, S).
            return_hidden_and_attn:
                A boolean flag to indicate whether return hidden states and attention heads
                of model. It should be true if you want to get hidden states of a fine-tuned model.
                Default: `False`

        Returns:
            If `return_hidden_and_attn` is `False`:
                1. Unnormalized logits with numeric type `torch.float32` and size
                (B, C).
                2. [CLS] hidden state of last layer
            Else:
                Return three values:
                1. Unnormalized logits with numeric type `torch.float32` and size
                (B, C).
                2. Hidden states: Tuple of torch.FloatTensor with shape: (B, S, H).
                (One for the output of the embeddings + one for the output of each layer.)
                3. Attentions: Tuple of torch.FloatTensor with shape: (B, A, S, S).
                (One for each layer).
        """
        # Return logits, hidden states and attention heads.
        if return_hidden_and_attn:
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                output_attentions=True
            )

            pooled_output = output.pooler_output
            hidden_states = output.hidden_states
            attentions = output.attentions

            pooled_output = self.dropout(pooled_output)
            return self.linear_layer(pooled_output), hidden_states, attentions

        # Return logits and [CLS].
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = output.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.linear_layer(pooled_output), pooled_output

    @torch.no_grad()
    def predict(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor,
            token_type_ids: torch.LongTensor,
            do_softmax=True
    ):
        r"""Perform prediction on batch of inputs without gradient.

        This method will neither contruct computational graph not calculate
        gradient. We use the following notation for the rest of the context.
            - B: batch size.
            - S: sequence length.
            - C: number of class.

        Args:
            input_ids:
                Batch of input token ids. `input_ids` is a `torch.Tensor` with
                numeric type `torch.int64` and size (B, S).
            attention_mask:
                Batch of input attention masks. `attention_mask` is a
                `torch.Tensor` with numeric type `torch.float32` and size
                (B, S).
            token_type_ids:
                Batch of input token type ids. `token_type_ids` is a
                `torch.Tensor` with numeric type `torch.int64` and size (B, S).

        Returns:
            Softmax normalized logits with numeric type `torch.float32` and
            size (B, C).
        """
        logits, _ = self(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_hidden_and_attn=False
                )
        if do_softmax:
            return F.softmax(logits, dim=-1), logits
        else:
            return logits.squeeze(-1)
