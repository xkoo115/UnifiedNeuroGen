"""Conditional Gaussian EEG-to-BOLD generator (M10)."""

# Derived from src/neurovascular_decoder/latent_decoder.py.
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from collections.abc import Mapping
from pathlib import Path

DEFAULT_CONFIG = {
    "feature_dim": 2928,
    "token_features": 366,
    "target_dim": 64,
    "condition_count": 4,
    "time_tokens": 8,
    "width": 128,
    "heads": 8,
    "depth": 4,
    "hrf_banks": 4,
    "query_tokens": 8,
    "activity_dim": 64,
    "context_dim": 64,
    "covariance_rank": 4,
    "dropout": 0.1,
    "activity_bottleneck_mean": True,
}


def build_model(config=None, baseline_weight=None, baseline_bias=None):
    """Build M10 from constructor settings and an optional fitted Ridge anchor."""
    settings = {**DEFAULT_CONFIG, **({} if config is None else dict(config))}
    if baseline_weight is not None:
        baseline_weight = torch.as_tensor(baseline_weight).to(dtype=torch.float32)
    if baseline_bias is not None:
        baseline_bias = torch.as_tensor(baseline_bias).to(dtype=torch.float32)
    return ConditionedFactorizedDecoder(
        **settings, baseline_weight=baseline_weight, baseline_bias=baseline_bias
    )


def load_checkpoint(
    path: str | Path, config: Mapping | None = None, *, map_location="cpu"
):
    """Strictly load a state dictionary, including its fitted anchor buffers."""
    state = torch.load(path, weights_only=True, map_location=map_location)
    weight = state.get("baseline_weight")
    bias = state.get("baseline_bias")
    if weight is not None and weight.numel() == 0:
        weight = bias = None
    model = build_model(config, baseline_weight=weight, baseline_bias=bias)
    model.load_state_dict(state, strict=True)
    return model.to(map_location).eval()


def m10_loss(
    model,
    output,
    target,
    condition,
    *,
    alignment_weight=0.10,
    classification_weight=0.20,
    orthogonality_weight=0.01,
):
    """Gaussian NLL with paired alignment, state supervision and orthogonality."""
    return low_rank_gaussian_nll(output, target) + factorized_condition_loss(
        output,
        model.encode_target_activity(target),
        condition,
        alignment_weight=alignment_weight,
        classification_weight=classification_weight,
        orthogonality_weight=orthogonality_weight,
    )


@dataclass(frozen=True)
class LatentDecoderOutput:
    """Predictive distribution and representation emitted by the decoder."""

    mean: Tensor
    log_std: Tensor
    covariance_factor: Tensor | None
    sample: Tensor
    eeg_representation: Tensor
    hrf_weights: Tensor
    context_representation: Tensor | None = None
    condition_logits: Tensor | None = None
    target_condition_logits: Tensor | None = None
    activity_subject_logits: Tensor | None = None
    context_subject_logits: Tensor | None = None


class _GradientReversalFunction(torch.autograd.Function):
    """Identity forward pass with a sign-reversed gradient for domain removal."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, value: Tensor, scale: float
    ) -> Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, gradient: Tensor
    ) -> tuple[Tensor, None]:
        return (-ctx.scale * gradient, None)


def gradient_reversal(value: Tensor, scale: float = 1.0) -> Tensor:
    """Return ``value`` while reversing downstream gradients by ``scale``."""
    if scale < 0:
        raise ValueError("gradient-reversal scale must be non-negative")
    return _GradientReversalFunction.apply(value, scale)


class _LatentCrossAttentionBlock(nn.Module):

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(width)
        hidden = width * 2
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )

    def forward(self, queries: Tensor, tokens: Tensor) -> Tensor:
        query = self.query_norm(queries)
        key_value = self.key_norm(tokens)
        attended, _ = self.cross_attention(
            query, key_value, key_value, need_weights=False
        )
        queries = queries + attended
        self_query = self.self_norm(queries)
        attended, _ = self.self_attention(
            self_query, self_query, self_query, need_weights=False
        )
        queries = queries + attended
        return queries + self.mlp(self.mlp_norm(queries))


class MultiResolutionCrossAttentionDecoder(nn.Module):
    """EEG token to target-latent decoder with explicit uncertainty."""

    def __init__(
        self,
        feature_dim: int,
        token_features: int,
        target_dim: int,
        *,
        time_tokens: int = 8,
        width: int = 64,
        heads: int = 4,
        depth: int = 2,
        hrf_banks: int = 4,
        query_tokens: int = 8,
        shared_representation_dim: int = 32,
        covariance_rank: int = 4,
        dropout: float = 0.05,
        baseline_weight: Tensor | None = None,
        baseline_bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        if (
            min(
                feature_dim,
                token_features,
                target_dim,
                time_tokens,
                width,
                heads,
                depth,
                hrf_banks,
                query_tokens,
            )
            <= 0
        ):
            raise ValueError("decoder dimensions must be positive")
        if feature_dim != token_features * time_tokens:
            raise ValueError("feature_dim must equal token_features * time_tokens")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if covariance_rank < 0:
            raise ValueError("covariance_rank must be non-negative")
        self.feature_dim = feature_dim
        self.token_features = token_features
        self.target_dim = target_dim
        self.time_tokens = time_tokens
        self.width = width
        self.hrf_banks = hrf_banks
        self.query_tokens = query_tokens
        self.shared_representation_dim = shared_representation_dim
        self.covariance_rank = covariance_rank
        self.baseline_enabled = baseline_weight is not None
        if baseline_weight is not None:
            if tuple(baseline_weight.shape) != (target_dim, feature_dim):
                raise ValueError(
                    "baseline_weight must have shape [target_dim, feature_dim]"
                )
            if baseline_bias is None or tuple(baseline_bias.shape) != (target_dim,):
                raise ValueError("baseline_bias must have shape [target_dim]")
            self.register_buffer("baseline_weight", baseline_weight.detach().clone())
            self.register_buffer("baseline_bias", baseline_bias.detach().clone())
        else:
            self.register_buffer("baseline_weight", torch.empty(0))
            self.register_buffer("baseline_bias", torch.empty(0))
        self.token_projection = nn.Sequential(
            nn.LayerNorm(token_features), nn.Linear(token_features, width), nn.GELU()
        )
        self.time_position = nn.Parameter(torch.zeros(1, time_tokens, width))
        nn.init.trunc_normal_(self.time_position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=2 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.hrf_logits = nn.Parameter(torch.zeros(hrf_banks, time_tokens))
        self.hrf_projection = nn.Linear(width, width)
        self.target_queries = nn.Parameter(torch.zeros(1, query_tokens, width))
        nn.init.trunc_normal_(self.target_queries, std=0.02)
        self.query_condition = nn.Linear(width, width)
        self.blocks = nn.ModuleList(
            [_LatentCrossAttentionBlock(width, heads, dropout) for _ in range(depth)]
        )
        self.query_norm = nn.LayerNorm(width)
        self.eeg_projector = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, shared_representation_dim),
        )
        self.target_projector = nn.Sequential(
            nn.Linear(target_dim, width),
            nn.GELU(),
            nn.Linear(width, shared_representation_dim),
        )
        flattened = query_tokens * width
        self.location = nn.Linear(flattened, target_dim)
        self.scale = nn.Linear(flattened, target_dim)
        self.factor = (
            nn.Linear(flattened, target_dim * covariance_rank)
            if covariance_rank
            else None
        )
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        if self.baseline_enabled:
            nn.init.zeros_(self.location.weight)
            nn.init.zeros_(self.location.bias)
        if self.factor is not None:
            nn.init.zeros_(self.factor.weight)
            nn.init.zeros_(self.factor.bias)

    def _check_features(self, features: Tensor) -> None:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected EEG features [batch, frames, {self.feature_dim}], got {tuple(features.shape)}"
            )

    def encode_eeg(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return query tokens, EEG representation, and normalized HRF banks."""
        self._check_features(features)
        batch = features.shape[0]
        tokens = features.reshape(batch, self.time_tokens, self.token_features)
        tokens = self.token_projection(tokens) + self.time_position
        tokens = self.temporal_encoder(tokens)
        hrf_weights = torch.softmax(self.hrf_logits, dim=-1)
        hrf_context = torch.einsum("btw,kt->bkw", tokens, hrf_weights)
        global_context = self.hrf_projection(hrf_context.mean(dim=1))
        queries = self.target_queries.expand(batch, -1, -1)
        queries = queries + self.query_condition(global_context).unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, tokens)
        queries = self.query_norm(queries)
        eeg_representation = F.normalize(
            self.eeg_projector(queries.mean(dim=1)), dim=-1
        )
        return (queries, eeg_representation, hrf_weights)

    def encode_target_representation(self, target: Tensor) -> Tensor:
        """Encode a paired target for training/evaluation-only alignment."""
        if target.ndim != 2 or target.shape[-1] != self.target_dim:
            raise ValueError(
                f"Expected target [batch, {self.target_dim}], got {tuple(target.shape)}"
            )
        return F.normalize(self.target_projector(target), dim=-1)

    def forward(self, features: Tensor, *, sample: bool = True) -> LatentDecoderOutput:
        queries, eeg_representation, hrf_weights = self.encode_eeg(features)
        flattened = queries.reshape(queries.shape[0], -1)
        residual = self.location(flattened)
        if self.baseline_enabled:
            baseline = (
                features[:, 0, :] @ self.baseline_weight.transpose(0, 1)
                + self.baseline_bias
            )
            mean = baseline + residual
        else:
            mean = residual
        log_std = self.scale(flattened).clamp(min=-5.0, max=2.0)
        factor = None
        if self.factor is not None:
            factor = 0.05 * self.factor(flattened).reshape(
                features.shape[0], self.target_dim, self.covariance_rank
            )
        generated = mean
        if sample:
            generated = mean + torch.exp(log_std) * torch.randn_like(mean)
            if factor is not None:
                noise = torch.randn(
                    features.shape[0],
                    self.covariance_rank,
                    device=features.device,
                    dtype=features.dtype,
                )
                generated = generated + torch.einsum("bdr,br->bd", factor, noise)
        return LatentDecoderOutput(
            mean=mean,
            log_std=log_std,
            covariance_factor=factor,
            sample=generated,
            eeg_representation=eeg_representation,
            hrf_weights=hrf_weights,
        )


class ConditionedFactorizedDecoder(nn.Module):
    """Decoder with a shared activity code and an EEG-derived context code."""

    def __init__(
        self,
        feature_dim: int,
        token_features: int,
        target_dim: int,
        condition_count: int,
        *,
        time_tokens: int = 8,
        width: int = 64,
        heads: int = 4,
        depth: int = 2,
        hrf_banks: int = 4,
        query_tokens: int = 8,
        activity_dim: int = 32,
        context_dim: int = 32,
        covariance_rank: int = 4,
        dropout: float = 0.05,
        subject_count: int | None = None,
        baseline_weight: Tensor | None = None,
        baseline_bias: Tensor | None = None,
        activity_bottleneck_mean: bool = False,
    ) -> None:
        super().__init__()
        if condition_count < 2 or activity_dim < 2 or context_dim < 2:
            raise ValueError(
                "condition_count, activity_dim and context_dim must be at least two"
            )
        if subject_count is not None and subject_count < 2:
            raise ValueError("subject_count must be at least two when supplied")
        self.backbone = MultiResolutionCrossAttentionDecoder(
            feature_dim=feature_dim,
            token_features=token_features,
            target_dim=target_dim,
            time_tokens=time_tokens,
            width=width,
            heads=heads,
            depth=depth,
            hrf_banks=hrf_banks,
            query_tokens=query_tokens,
            shared_representation_dim=activity_dim,
            covariance_rank=covariance_rank,
            dropout=dropout,
        )
        self.target_dim = target_dim
        self.activity_dim = activity_dim
        self.context_dim = context_dim
        self.subject_count = subject_count
        self.activity_bottleneck_mean = activity_bottleneck_mean
        self.baseline_enabled = baseline_weight is not None
        if baseline_weight is not None:
            if tuple(baseline_weight.shape) != (target_dim, feature_dim):
                raise ValueError(
                    "baseline_weight must have shape [target_dim, feature_dim]"
                )
            if baseline_bias is None or tuple(baseline_bias.shape) != (target_dim,):
                raise ValueError("baseline_bias must have shape [target_dim]")
            self.register_buffer("baseline_weight", baseline_weight.detach().clone())
            self.register_buffer("baseline_bias", baseline_bias.detach().clone())
        else:
            self.register_buffer("baseline_weight", torch.empty(0))
            self.register_buffer("baseline_bias", torch.empty(0))
        self.condition_head = nn.Linear(activity_dim, condition_count)
        self.target_condition_head = nn.Linear(activity_dim, condition_count)
        self.activity_subject_head = (
            nn.Linear(activity_dim, subject_count)
            if subject_count is not None
            else None
        )
        self.context_subject_head = (
            nn.Linear(context_dim, subject_count) if subject_count is not None else None
        )
        self.context_projector = nn.Sequential(
            nn.Linear(query_tokens * width, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        flattened = query_tokens * width
        self.location = (
            nn.Sequential(
                nn.LayerNorm(activity_dim + context_dim),
                nn.Linear(activity_dim + context_dim, width),
                nn.GELU(),
                nn.Linear(width, target_dim),
            )
            if activity_bottleneck_mean
            else nn.Linear(flattened + activity_dim + context_dim, target_dim)
        )
        self.scale = nn.Linear(flattened + activity_dim + context_dim, target_dim)
        self.factor = (
            nn.Linear(
                flattened + activity_dim + context_dim, target_dim * covariance_rank
            )
            if covariance_rank
            else None
        )
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        if self.baseline_enabled:
            final_location = (
                self.location[-1] if activity_bottleneck_mean else self.location
            )
            nn.init.zeros_(final_location.weight)
            nn.init.zeros_(final_location.bias)
        if self.factor is not None:
            nn.init.zeros_(self.factor.weight)
            nn.init.zeros_(self.factor.bias)

    def encode_target_activity(self, target: Tensor) -> Tensor:
        return self.backbone.encode_target_representation(target)

    def forward(
        self,
        features: Tensor,
        *,
        sample: bool = True,
        target_for_condition: Tensor | None = None,
        context_scale: float = 1.0,
        subject_adversarial_scale: float = 1.0,
    ) -> LatentDecoderOutput:
        queries, activity, hrf_weights = self.backbone.encode_eeg(features)
        flattened = queries.reshape(queries.shape[0], -1)
        context = self.context_projector(flattened)
        if context_scale < 0:
            raise ValueError("context_scale must be non-negative")
        if subject_adversarial_scale < 0:
            raise ValueError("subject_adversarial_scale must be non-negative")
        joint = torch.cat((flattened, activity, context * context_scale), dim=-1)
        mean_input = (
            torch.cat((activity, context * context_scale), dim=-1)
            if self.activity_bottleneck_mean
            else joint
        )
        residual = self.location(mean_input)
        if self.baseline_enabled:
            baseline = (
                features[:, 0, :] @ self.baseline_weight.transpose(0, 1)
                + self.baseline_bias
            )
            mean = baseline + residual
        else:
            mean = residual
        log_std = self.scale(joint).clamp(min=-5.0, max=2.0)
        factor = None
        if self.factor is not None:
            factor = 0.05 * self.factor(joint).reshape(
                features.shape[0], self.target_dim, self.backbone.covariance_rank
            )
        generated = mean
        if sample:
            generated = mean + torch.exp(log_std) * torch.randn_like(mean)
            if factor is not None:
                noise = torch.randn(
                    features.shape[0],
                    self.backbone.covariance_rank,
                    device=features.device,
                    dtype=features.dtype,
                )
                generated = generated + torch.einsum("bdr,br->bd", factor, noise)
        return LatentDecoderOutput(
            mean=mean,
            log_std=log_std,
            covariance_factor=factor,
            sample=generated,
            eeg_representation=activity,
            hrf_weights=hrf_weights,
            context_representation=context,
            condition_logits=self.condition_head(activity),
            target_condition_logits=(
                self.target_condition_head(
                    self.encode_target_activity(target_for_condition)
                )
                if target_for_condition is not None
                else None
            ),
            activity_subject_logits=(
                self.activity_subject_head(
                    gradient_reversal(activity, subject_adversarial_scale)
                )
                if self.activity_subject_head is not None
                else None
            ),
            context_subject_logits=(
                self.context_subject_head(context)
                if self.context_subject_head is not None
                else None
            ),
        )


def factorized_condition_loss(
    output: LatentDecoderOutput,
    target_activity: Tensor,
    condition: Tensor,
    *,
    alignment_weight: float = 0.1,
    classification_weight: float = 0.2,
    orthogonality_weight: float = 0.01,
    subject: Tensor | None = None,
    subject_adversarial_weight: float = 0.05,
    subject_residual_weight: float = 0.05,
) -> Tensor:
    """Combine paired activity alignment, condition supervision and factorization."""
    if output.condition_logits is None or output.context_representation is None:
        raise ValueError("output must come from ConditionedFactorizedDecoder")
    target_logits = output.target_condition_logits
    if target_logits is None:
        raise ValueError("target condition logits are required")
    loss = alignment_weight * symmetric_alignment_loss(
        output.eeg_representation, target_activity
    )
    loss = loss + classification_weight * (
        F.cross_entropy(output.condition_logits, condition)
        + F.cross_entropy(target_logits, condition)
    )
    activity = F.normalize(output.eeg_representation, dim=-1)
    context = F.normalize(output.context_representation, dim=-1)
    loss = (
        loss + orthogonality_weight * (activity * context).sum(dim=-1).square().mean()
    )
    if subject is not None:
        if (
            output.activity_subject_logits is None
            or output.context_subject_logits is None
        ):
            raise ValueError(
                "subject labels require a decoder constructed with subject_count"
            )
        if subject.ndim != 1 or len(subject) != len(activity):
            raise ValueError(
                "subject labels must be one-dimensional and match the batch"
            )
        if subject_adversarial_weight < 0 or subject_residual_weight < 0:
            raise ValueError("subject loss weights must be non-negative")
        loss = loss + subject_adversarial_weight * F.cross_entropy(
            output.activity_subject_logits, subject
        )
        loss = loss + subject_residual_weight * F.cross_entropy(
            output.context_subject_logits, subject
        )
    return loss


def low_rank_gaussian_nll(
    output: LatentDecoderOutput, target: Tensor, *, reduction: str = "mean"
) -> Tensor:
    """Negative log likelihood for diag(exp(2*log_std)) + factor factor.T."""
    if target.shape != output.mean.shape:
        raise ValueError("target and predictive mean must have identical shapes")
    residual = target - output.mean
    inv_diag = torch.exp(-2.0 * output.log_std)
    diagonal_maha = (residual.square() * inv_diag).sum(dim=-1)
    logdet = 2.0 * output.log_std.sum(dim=-1)
    maha = diagonal_maha
    if output.covariance_factor is not None:
        factor = output.covariance_factor
        weighted_factor = factor * inv_diag.unsqueeze(-1)
        system = torch.eye(
            factor.shape[-1], device=factor.device, dtype=factor.dtype
        ).unsqueeze(0) + torch.einsum("bdr,bds->brs", factor, weighted_factor)
        correction_rhs = torch.einsum("bdr,bd->br", weighted_factor, residual)
        correction = torch.linalg.solve(system, correction_rhs.unsqueeze(-1)).squeeze(
            -1
        )
        maha = maha - (correction_rhs * correction).sum(dim=-1)
        logdet = logdet + torch.linalg.slogdet(system).logabsdet
    values = 0.5 * (
        maha + logdet + target.shape[-1] * torch.log(target.new_tensor(2.0 * torch.pi))
    )
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError("reduction must be mean, sum, or none")


def symmetric_alignment_loss(
    eeg_representation: Tensor,
    target_representation: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """InfoNCE alignment with matched pairs on the diagonal."""
    if eeg_representation.shape != target_representation.shape:
        raise ValueError("EEG and target representations must have the same shape")
    if eeg_representation.ndim != 2 or eeg_representation.shape[0] < 2:
        raise ValueError("alignment requires at least two paired examples")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = eeg_representation @ target_representation.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )
