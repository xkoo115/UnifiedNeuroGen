# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Original source license: see the release LICENSE and third-party notices.
# Adapted by xkoo115 in the author-supplied gdit/models.py archive.
# Shared public source: https://github.com/xkoo115/UnifiedNeuroGen/tree/427a4f342ebd0494690a6b4c31186c2084b9b22a
# Upstream references: https://github.com/facebookresearch/DiT
# https://github.com/openai/glide-text2im
# https://github.com/facebookresearch/mae/blob/main/models_mae.py

"""Original Hb DiT architecture; pretrained Hb weights are not bundled."""

import numpy as np
import torch
from torch import nn
from typing import Mapping

from .dit import (
    Adaptor,
    DiTBlockWithCrossAttention,
    FinalLayer,
    PatchEmbed1D,
    TimestepEmbedder,
    get_1d_sincos_pos_embed,
)


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
            input_dim=144,
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


def DiT_fMRI(**kwargs):
    return DiT(depth=28, hidden_size=1000, patch_size=1, num_heads=10, **kwargs)


PRODUCTION_CONFIG = {
    "input_size": 128,
    "in_channels": 144,
    "hidden_size": 1000,
    "depth": 28,
    "num_heads": 10,
    "patch_size": 1,
    "learn_sigma": True,
    "cond_channels": 144,
    "cond_width": 1000,
}

DEMO_CONFIG = {
    "input_size": 8,
    "in_channels": 2,
    "hidden_size": 16,
    "depth": 1,
    "num_heads": 4,
    "patch_size": 1,
    "learn_sigma": True,
    "cond_channels": 144,
    "cond_width": 16,
}


def build_model(config: Mapping | None = None) -> DiT:
    """Build the Hb DiT for [batch, 144, hidden_size] conditions.

    The production target is [batch, 144, 128]. DEMO_CONFIG defines a small
    random model for software tests.
    """
    values = dict(PRODUCTION_CONFIG)
    supplied = {} if config is None else dict(config)
    unknown = set(supplied) - (
        set(PRODUCTION_CONFIG) | {"mlp_ratio", "class_dropout_prob", "num_classes"}
    )
    if unknown:
        raise ValueError(f"Unsupported Hb DiT configuration fields: {sorted(unknown)}")
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
    if values["cond_channels"] != 144:
        raise ValueError("The original Hb adaptor requires cond_channels=144")
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
