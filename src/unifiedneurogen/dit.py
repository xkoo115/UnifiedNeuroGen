# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Original source license: see the release LICENSE and third-party notices.
# Adapted by xkoo115 in UnifiedNeuroGen (MIT, Copyright (c) 2025 xkoo115).
# Source: https://github.com/xkoo115/UnifiedNeuroGen/blob/427a4f342ebd0494690a6b4c31186c2084b9b22a/models.py
# Upstream references: https://github.com/facebookresearch/DiT
# https://github.com/openai/glide-text2im
# https://github.com/facebookresearch/mae/blob/main/models_mae.py

"""Original checkpoint-compatible EEG-conditioned 1D diffusion transformer."""

import math
from typing import Callable, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=None,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if return_attention:
            return attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_qkv:
            return (x, qkv)
        return x


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class CrossAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        assert (
            self.head_dim * num_heads == hidden_size
        ), "hidden_size must be divisible by num_heads"
        self.scale = self.head_dim ** (-0.5)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c):
        batch_size, num_tokens, hidden_size = x.size()
        queries = (
            self.q_proj(x)
            .reshape(batch_size, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        keys_values = (
            self.k_proj(c)
            .reshape(batch_size, -1, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        keys, values = (keys_values[0], keys_values[1])
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_drop(attention_probs)
        weighted_values = torch.matmul(attention_probs, values)
        weighted_values = weighted_values.transpose(1, 2).reshape(
            batch_size, num_tokens, hidden_size
        )
        attended_values = self.out_proj(weighted_values)
        attended_values = self.resid_drop(attended_values)
        x = x + attended_values
        return x


class DiTBlockWithCrossAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-06)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-06)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, c):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), c)
        x = x + self.mlp(self.norm3(x))
        return x


def modulate(x, shift, scale):
    x = x * (1 + scale) + shift
    return x


class FinalLayer(nn.Module):

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Adaptor(nn.Module):

    def __init__(self, input_dim=75 * 40, output_dim=768, num_patches=64):
        super(Adaptor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_patches, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_patches, out_channels=num_patches, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        return x


class PatchEmbed1D(nn.Module):

    def __init__(
        self,
        length_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        bias: bool = True,
        strict_length_size: bool = True,
        dynamic_length_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = (patch_size,)
        if length_size is not None:
            self.length_size = length_size
            self.num_patches = self.length_size // self.patch_size[0]
        else:
            self.length_size = None
            self.num_patches = None
        self.flatten = flatten
        self.strict_length_size = strict_length_size
        self.dynamic_length_pad = dynamic_length_pad
        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        if self.length_size is not None:
            if self.strict_length_size:
                assert (
                    L == self.length_size
                ), f"Input length ({L}) doesn't match model ({self.length_size})."
            else:
                assert (
                    L % self.patch_size == 0
                ), f"Input length ({L}) should be divisible by patch size ({self.patch_size})."
        if self.dynamic_length_pad:
            pad_len = (self.patch_size - L % self.patch_size) % self.patch_size
            x = F.pad(x, (0, pad_len))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class DiT(nn.Module):

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed1D(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = Adaptor(
            input_dim=2560,
            output_dim=hidden_size,
            num_patches=int(input_size / patch_size),
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlockWithCrossAttention(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.x_embedder.num_patches
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        t = x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, p, c))
        imgs = x.permute(0, 3, 1, 2).reshape(shape=(x.shape[0], c, -1))
        return imgs

    def forward(self, x, t, c):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        c = self.c_embedder(c) + self.pos_embed
        all_conditioning = t.unsqueeze(1).expand(-1, x.size(1), -1) + c
        for block in self.blocks:
            x = block(x, all_conditioning)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t)
        eps, rest = (model_out[:, :3], model_out[:, 3:])
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def get_1d_sincos_pos_embed(embed_dim, length):
    pos_embed = np.zeros((length, embed_dim))
    position = np.arange(0, length, dtype=np.float32)
    div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
    pos_embed[:, 0::2] = np.sin(position[:, None] * div_term)
    pos_embed[:, 1::2] = np.cos(position[:, None] * div_term)
    return pos_embed


def DiT_fMRI(**kwargs):
    return DiT(depth=24, hidden_size=1200, patch_size=4, num_heads=12, **kwargs)


PRODUCTION_CONFIG = {
    "input_size": 2560,
    "in_channels": 1,
    "hidden_size": 1200,
    "depth": 24,
    "num_heads": 12,
    "patch_size": 4,
    "learn_sigma": True,
    "cond_channels": 2560,
    "cond_width": 1200,
}

DEMO_CONFIG = {
    "input_size": 32,
    "in_channels": 1,
    "hidden_size": 32,
    "depth": 1,
    "num_heads": 4,
    "patch_size": 4,
    "learn_sigma": True,
    "cond_channels": 2560,
    "cond_width": 32,
}


def build_model(config: Mapping | None = None) -> DiT:
    """Build the fMRI DiT from a configuration dictionary.

    Condition tensors have shape [batch, 2560, hidden_size]. The adaptor
    changes their channel count while preserving the last feature dimension.
    DEMO_CONFIG defines a small random model for software tests.
    """
    values = dict(PRODUCTION_CONFIG)
    supplied = {} if config is None else dict(config)
    unknown = set(supplied) - (
        set(PRODUCTION_CONFIG) | {"mlp_ratio", "class_dropout_prob", "num_classes"}
    )
    if unknown:
        raise ValueError(f"Unsupported DiT configuration fields: {sorted(unknown)}")
    values.update(supplied)
    for field in (
        "input_size",
        "in_channels",
        "hidden_size",
        "depth",
        "num_heads",
        "patch_size",
        "cond_channels",
        "cond_width",
    ):
        if (
            isinstance(values[field], bool)
            or not isinstance(values[field], int)
            or values[field] <= 0
        ):
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(values["learn_sigma"], bool):
        raise ValueError("learn_sigma must be a boolean")
    if values["cond_channels"] != 2560:
        raise ValueError("The original adaptor requires cond_channels=2560")
    if values["cond_width"] != values["hidden_size"]:
        raise ValueError(
            "cond_width must equal hidden_size; the original adaptor preserves condition feature width"
        )
    if values["hidden_size"] % values["num_heads"]:
        raise ValueError("hidden_size must be divisible by num_heads")
    if values["hidden_size"] % 2:
        raise ValueError(
            "hidden_size must be even for the original sine/cosine embedding"
        )
    if values["input_size"] % values["patch_size"]:
        raise ValueError("input_size must be divisible by patch_size")
    if (values["input_size"] // values["patch_size"]) % 2:
        raise ValueError(
            "The original cross-attention requires an even number of condition tokens"
        )
    values.pop("cond_channels")
    values.pop("cond_width")
    return DiT(**values)
