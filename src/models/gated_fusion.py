"""
Context-Adaptive Gated Fusion Model for Health Misinformation Detection.

This module implements a gated fusion architecture that combines text and
contextual features using a learned, context-adaptive gating mechanism.

The model gracefully handles missing context by:
1. Zeroing out context features when context_mask = 0 (before encoding).
2. Forcing the gate value to 0 for samples without context, so the model
   falls back entirely to the text representation.
3. Never dropping samples or fabricating replacement values.

Input Contract:
    text_features:   (batch_size, text_dim)    — numerical text representations
    context_features:(batch_size, context_dim)  — numerical context representations
    context_mask:    (batch_size, 1) or (batch_size,) — 1=context available, 0=missing

Output:
    logits:      (batch_size, num_classes) — class prediction logits
    gate_values: (batch_size, 1)           — learned gate values (after masking)

Author: Member 4
"""

import torch
import torch.nn as nn


class GatedFusionModel(nn.Module):
    """
    Context-adaptive gated fusion model.

    Combines text and context representations through a learned gate:
        fused = (1 - gate) * text_repr + gate * context_repr

    When context is unavailable (context_mask=0), the gate is forced to 0
    so the model relies entirely on the text representation.

    Args:
        text_dim (int): Dimensionality of input text features.
        context_dim (int): Dimensionality of input context features.
        hidden_dim (int): Dimensionality of the shared hidden representation.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability. Default: 0.3.

    Example:
        >>> model = GatedFusionModel(text_dim=768, context_dim=32, hidden_dim=128, num_classes=2)
        >>> logits, gates = model(text_feats, ctx_feats, ctx_mask)
    """

    def __init__(
        self,
        text_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Store configuration for inspection / serialization
        self.text_dim = text_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # ── Text encoder ─────────────────────────────────────────────
        # Projects raw text features into the shared hidden space.
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Context encoder ──────────────────────────────────────────
        # Projects raw context features into the shared hidden space.
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Gating network ───────────────────────────────────────────
        # Learns how much to trust context vs. text.
        # Input: concatenation of [text_repr, context_repr, context_mask]
        # The mask is included so the gate can explicitly learn that
        # absent context should be ignored.
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # ── Classifier head ──────────────────────────────────────────
        # Operates on the fused representation.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        context_features: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the gated fusion model.

        Args:
            text_features: Tensor of shape (batch_size, text_dim).
            context_features: Tensor of shape (batch_size, context_dim).
            context_mask: Tensor of shape (batch_size,) or (batch_size, 1).
                          Values must be 0 or 1.

        Returns:
            logits: Tensor of shape (batch_size, num_classes).
            gate_values: Tensor of shape (batch_size, 1) — the effective gate
                         after context-mask enforcement.
        """
        # ── Validate inputs ──────────────────────────────────────────
        self._validate_inputs(text_features, context_features, context_mask)

        # Ensure context_mask is (batch_size, 1) for broadcasting
        if context_mask.dim() == 1:
            context_mask = context_mask.unsqueeze(1)
        context_mask = context_mask.float()

        # ── Step 1: Mask context BEFORE encoding ─────────────────────
        # When context_mask=0, replace context features with zeros so the
        # context encoder never sees fabricated or stale values.
        masked_context = context_features * context_mask

        # ── Step 2: Encode ───────────────────────────────────────────
        text_repr = self.text_encoder(text_features)        # (B, hidden_dim)
        context_repr = self.context_encoder(masked_context)  # (B, hidden_dim)

        # ── Step 3: Compute gate ─────────────────────────────────────
        gate_input = torch.cat(
            [text_repr, context_repr, context_mask], dim=1
        )  # (B, hidden_dim*2 + 1)
        raw_gate = self.gate_network(gate_input)  # (B, 1) in [0, 1]

        # Force gate to 0 when context is unavailable.
        # This guarantees the model falls back to text-only representation.
        gate = raw_gate * context_mask  # (B, 1)

        # ── Step 4: Fuse ─────────────────────────────────────────────
        fused = (1.0 - gate) * text_repr + gate * context_repr  # (B, hidden_dim)

        # ── Step 5: Classify ─────────────────────────────────────────
        logits = self.classifier(fused)  # (B, num_classes)

        return logits, gate

    def _validate_inputs(
        self,
        text_features: torch.Tensor,
        context_features: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> None:
        """Validate tensor shapes and raise informative errors."""
        if text_features.dim() != 2:
            raise ValueError(
                f"text_features must be 2D (batch_size, text_dim), "
                f"got shape {text_features.shape}"
            )
        if context_features.dim() != 2:
            raise ValueError(
                f"context_features must be 2D (batch_size, context_dim), "
                f"got shape {context_features.shape}"
            )
        if text_features.size(0) != context_features.size(0):
            raise ValueError(
                f"Batch size mismatch: text_features has {text_features.size(0)}, "
                f"context_features has {context_features.size(0)}"
            )
        if text_features.size(1) != self.text_dim:
            raise ValueError(
                f"text_features dim mismatch: expected {self.text_dim}, "
                f"got {text_features.size(1)}"
            )
        if context_features.size(1) != self.context_dim:
            raise ValueError(
                f"context_features dim mismatch: expected {self.context_dim}, "
                f"got {context_features.size(1)}"
            )
        mask_numel = context_mask.dim()
        if mask_numel == 1:
            expected_len = text_features.size(0)
            if context_mask.size(0) != expected_len:
                raise ValueError(
                    f"context_mask length {context_mask.size(0)} does not match "
                    f"batch size {expected_len}"
                )
        elif mask_numel == 2:
            if context_mask.size(0) != text_features.size(0) or context_mask.size(1) != 1:
                raise ValueError(
                    f"context_mask must be (batch_size, 1), got {context_mask.shape}"
                )
        else:
            raise ValueError(
                f"context_mask must be 1D or 2D, got {mask_numel}D"
            )

    def get_config(self) -> dict:
        """Return model configuration as a dictionary (useful for logging)."""
        return {
            "text_dim": self.text_dim,
            "context_dim": self.context_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
        }
