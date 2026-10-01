"""SimCSE Model Architecture (Unsupervised & Supervised).
Supports contrastive InfoNCE loss, MLP projection layer, and flexible pooling strategies.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PreTrainedModel


class MLPLayer(nn.Module):
    """Projection MLP head for SimCSE training (linear + tanh).
    Paper observation: Using an MLP head during training and discarding it during test
    improves sentence representations.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(features))


class SimCSEModel(nn.Module):
    """SimCSE model implementation wrapping Hugging Face Transformer backbones."""

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        pooling: str = "cls",  # "cls", "cls_before_pooler", or "mean"
        temperature: float = 0.05,
        use_mlp: bool = True,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        if dropout_rate is not None:
            self.config.hidden_dropout_prob = dropout_rate
            self.config.attention_probs_dropout_prob = dropout_rate

        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.pooling = pooling
        self.temperature = temperature
        self.use_mlp = use_mlp
        self.mlp = MLPLayer(self.config.hidden_size) if use_mlp else nn.Identity()

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        apply_mlp: bool = False,
    ) -> torch.Tensor:
        """Encode tokenized input into normalized sentence representation."""
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)

        if self.pooling == "cls":
            # Use [CLS] representation (first token)
            rep = outputs.last_hidden_state[:, 0]
        elif self.pooling == "mean":
            # Mean pooling across valid tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            rep = sum_embeddings / sum_mask
        elif self.pooling == "pooler_output" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            rep = outputs.pooler_output
        else:
            rep = outputs.last_hidden_state[:, 0]

        if apply_mlp and self.use_mlp:
            rep = self.mlp(rep)

        return rep

    def forward_unsupervised(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        same_dropout_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unsupervised SimCSE forward pass.
        Runs input through encoder twice with independent dropout masks (or same mask if ablated).
        """
        z1 = self.encode(input_ids, attention_mask, token_type_ids, apply_mlp=True)

        if same_dropout_mask:
            # Ablation: identical view (same dropout mask / deterministic duplicate)
            z2 = z1.clone()
        else:
            # Second forward pass triggers independent dropout
            z2 = self.encode(input_ids, attention_mask, token_type_ids, apply_mlp=True)

        # Normalize representations
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        # Cosine similarity matrix: (batch_size, batch_size)
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature

        # Labels: diagonal entries are positive pairs
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device, dtype=torch.long)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss, sim_matrix

    def forward_supervised(
        self,
        premise_inputs: Dict[str, torch.Tensor],
        positive_inputs: Dict[str, torch.Tensor],
        negative_inputs: Optional[Dict[str, torch.Tensor]] = None,
        use_hard_negatives: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Supervised SimCSE forward pass.
        Computes representations for premise, positive (entailment), and optional negative (contradiction).
        """
        z_premise = self.encode(
            premise_inputs["input_ids"],
            premise_inputs["attention_mask"],
            premise_inputs.get("token_type_ids"),
            apply_mlp=True,
        )
        z_positive = self.encode(
            positive_inputs["input_ids"],
            positive_inputs["attention_mask"],
            positive_inputs.get("token_type_ids"),
            apply_mlp=True,
        )

        z_premise = F.normalize(z_premise, p=2, dim=-1)
        z_positive = F.normalize(z_positive, p=2, dim=-1)

        if use_hard_negatives and negative_inputs is not None:
            z_negative = self.encode(
                negative_inputs["input_ids"],
                negative_inputs["attention_mask"],
                negative_inputs.get("token_type_ids"),
                apply_mlp=True,
            )
            z_negative = F.normalize(z_negative, p=2, dim=-1)

            # Similarity against all positives and all hard negatives: shape (B, 2B)
            sim_pos = torch.matmul(z_premise, z_positive.T) / self.temperature
            sim_neg = torch.matmul(z_premise, z_negative.T) / self.temperature
            sim_matrix = torch.cat([sim_pos, sim_neg], dim=1)
        else:
            # Without hard negatives: shape (B, B)
            sim_matrix = torch.matmul(z_premise, z_positive.T) / self.temperature

        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device, dtype=torch.long)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss, sim_matrix

    @torch.no_grad()
    def get_sentence_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Inference mode sentence embedding extraction (without MLP head as recommended)."""
        self.eval()
        rep = self.encode(input_ids, attention_mask, token_type_ids, apply_mlp=False)
        if normalize:
            rep = F.normalize(rep, p=2, dim=-1)
        return rep
