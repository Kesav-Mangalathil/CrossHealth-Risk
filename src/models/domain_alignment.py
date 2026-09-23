"""
Domain Alignment Module for Cross-Dataset Generalization.

This module provides adversarial domain alignment components that encourage
the feature extractor to produce domain-invariant representations.

Architecture overview:
    features ──► GradientReversal ──► DomainDiscriminator ──► domain_pred
                 (reverses gradient       (source vs target
                  during backprop)         classifier)

The gradient reversal trick ensures that the upstream feature extractor
learns representations that *confuse* the domain discriminator, thereby
becoming domain-invariant.

Components:
    - GradientReversalFunction : Custom autograd function for gradient reversal.
    - GradientReversalLayer    : nn.Module wrapper around GradientReversalFunction.
    - DomainDiscriminator      : Small classifier distinguishing source vs. target.
    - DomainAlignment          : Wrapper combining reversal + discriminator.
    - domain_alignment_loss()  : Standalone BCE loss function for domain labels.

Input Contract:
    features:      (batch_size, feature_dim) — representations from the encoder
    domain_labels: (batch_size,) or (batch_size, 1) — 0=source, 1=target

All dimensions are configurable through constructor arguments.

Author: Member 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ═══════════════════════════════════════════════════════════════════════
#  Gradient Reversal
# ═══════════════════════════════════════════════════════════════════════


class GradientReversalFunction(Function):
    """
    Custom autograd function that passes inputs unchanged during the
    forward pass and reverses (negates and scales) gradients during
    the backward pass.

    This is the core trick behind adversarial domain adaptation
    (Ganin & Lempitsky, 2015).

    Args:
        lambda_: Scaling factor for the reversed gradient.
                 Controls the strength of domain-adaptation regularization.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse and scale the gradient; no gradient w.r.t. lambda_
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Module wrapper for GradientReversalFunction.

    Args:
        lambda_ (float): Gradient reversal scaling factor. Default: 1.0.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        """Update the reversal strength (e.g., schedule it during training)."""
        self.lambda_ = lambda_


# ═══════════════════════════════════════════════════════════════════════
#  Domain Discriminator
# ═══════════════════════════════════════════════════════════════════════


class DomainDiscriminator(nn.Module):
    """
    Small neural network that classifies a representation as belonging
    to the source domain (0) or the target domain (1).

    Args:
        feature_dim (int): Dimensionality of input features.
        hidden_dim (int): Dimensionality of the hidden layer.
        dropout (float): Dropout probability. Default: 0.3.

    Example:
        >>> disc = DomainDiscriminator(feature_dim=128, hidden_dim=64)
        >>> domain_pred = disc(features)  # (batch_size, 1) in [0, 1]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feature_dim)

        Returns:
            domain_pred: (batch_size, 1) — probability of being target domain.
        """
        if features.dim() != 2 or features.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected features of shape (batch_size, {self.feature_dim}), "
                f"got {features.shape}"
            )
        return self.network(features)


# ═══════════════════════════════════════════════════════════════════════
#  Domain Alignment (full wrapper)
# ═══════════════════════════════════════════════════════════════════════


class DomainAlignment(nn.Module):
    """
    End-to-end domain alignment module.

    Chains GradientReversalLayer → DomainDiscriminator so that, when
    trained jointly with the main model, the upstream encoder learns
    domain-invariant representations.

    Args:
        feature_dim (int): Dimensionality of encoder output / hidden representations.
        hidden_dim (int): Hidden layer size inside the discriminator.
        lambda_ (float): Gradient reversal strength. Default: 1.0.
        dropout (float): Dropout probability. Default: 0.3.

    Example:
        >>> da = DomainAlignment(feature_dim=128, hidden_dim=64, lambda_=0.5)
        >>> domain_pred = da(fused_repr)
        >>> loss = domain_alignment_loss(domain_pred, domain_labels)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        lambda_: float = 1.0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gradient_reversal = GradientReversalLayer(lambda_=lambda_)
        self.discriminator = DomainDiscriminator(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feature_dim)

        Returns:
            domain_pred: (batch_size, 1) — source/target probability.
        """
        reversed_features = self.gradient_reversal(features)
        return self.discriminator(reversed_features)

    def set_lambda(self, lambda_: float) -> None:
        """Update gradient reversal strength during training."""
        self.gradient_reversal.set_lambda(lambda_)


# ═══════════════════════════════════════════════════════════════════════
#  Standalone Loss Function
# ═══════════════════════════════════════════════════════════════════════


def domain_alignment_loss(
    domain_pred: torch.Tensor,
    domain_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute binary cross-entropy loss for domain classification.

    Args:
        domain_pred: (batch_size, 1) — predicted domain probabilities.
        domain_labels: (batch_size,) or (batch_size, 1) — ground-truth
                       domain labels (0=source, 1=target).

    Returns:
        loss: Scalar tensor — BCE loss.
    """
    if domain_labels.dim() == 1:
        domain_labels = domain_labels.unsqueeze(1)
    domain_labels = domain_labels.float()
    return F.binary_cross_entropy(domain_pred, domain_labels)
