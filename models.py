# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.models.layers import drop_path, trunc_normal_
from timm.models.registry import register_model
from functools import partial
from einops import rearrange
from typing import Optional, Callable
from torch.nn import TransformerEncoder
from torchvision.models.video.mvit import PositionalEncoding


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

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
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
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
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
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
            return x, qkv

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is None:
            init_values = 0

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, in_chans=1, out_chans=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x


class NeuralTransformer(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200,
                 depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # To identify whether it is neural tokenizer or neural decoder.
        # For the neural decoder, use linear projection (PatchEmbed) to project codebook dimension to hidden dimension.
        # Otherwise, use TemporalConv to extract temporal features from EEG signals.
        self.patch_embed = TemporalConv(out_chans=out_chans) if in_chans == 1 else PatchEmbed(EEG_size=EEG_size,
                                                                                              patch_size=patch_size,
                                                                                              in_chans=in_chans,
                                                                                              embed_dim=embed_dim)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.time_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1,
                                                                                                                    2)
            pos_embed = torch.cat((pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            print(x.shape, pos_embed.shape)
            x = x + pos_embed
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(
                1, 2)
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            print(t.shape)
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                # return self.fc_norm(t.mean(1))
                return self.fc_norm(t)
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        '''
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens,
                                  return_all_tokens=return_all_tokens, **kwargs)
        # x = self.head(x)
        return x

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                # use last norm for all intermediate layers
                if l in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)

        return features

@register_model
def labram_base_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # print(c.shape)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert (
            self.head_dim * num_heads == hidden_size
        ), "hidden_size must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # 独立的线性层，分别用于计算Q、K和V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c):
        batch_size, num_tokens, hidden_size = x.size()

        queries = self.q_proj(x).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        keys_values = self.k_proj(c).reshape(batch_size, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        keys, values = keys_values[0], keys_values[1]

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_drop(attention_probs)

        weighted_values = torch.matmul(attention_probs, values)

        weighted_values = weighted_values.transpose(1, 2).reshape(batch_size, num_tokens, hidden_size)

        attended_values = self.out_proj(weighted_values)
        attended_values = self.resid_drop(attended_values)

        x = x + attended_values

        return x


class DiTBlockWithCrossAttention(nn.Module):
    """
    A DiT block with cross-attention conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), c)
        x = x + self.mlp(self.norm3(x))
        return x


def modulate(x, shift, scale):
    # x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = x * (1 + scale) + shift
    return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # print(c.shape)
        # print(x.shape)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)
        return x

class CtoPatchEmbed(nn.Module):
    def __init__(self, c_height=129, c_width=12, patch_size=16, in_chans=63, embed_dim=1152):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, c):
        c = c.permute(0, 3, 1, 2)
        c = self.proj(c)
        c = c.flatten(2).transpose(1, 2)
        return c


class Adaptor(nn.Module):
    def __init__(self, input_dim=75 * 40, output_dim=768, num_patches=64):
        super(Adaptor, self).__init__()

        # Conv1d layer to transform the 200 dimension to 64
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_patches, kernel_size=1),  # (N, 75*40, 200) -> (N, 75*40, 64)
            nn.ReLU(),
            nn.Conv1d(in_channels=num_patches, out_channels=num_patches, kernel_size=1),
            # (N, 75*40, 64) -> (N, 75*40, 64)
            nn.ReLU(),
        )

        # Linear layer to transform flattened (75*40) -> hidden_size
        # self.fc = nn.Linear(1200, 1200)

    def forward(self, x):
        batch_size = x.shape[0]

        # # Reshape the input from (N, 75, 40, 200) to (N, 75*40, 200)
        # x = x.view(batch_size, -1, x.shape[-1])
        # x = x.permute(0, 2, 1)

        # Apply convolution to reduce the 200 dimension to 64, resulting in (N, 75*40, 64)
        x = self.conv(x)

        # # Reshape to (N, 64, 75*40)
        # x = x.permute(0, 2, 1)

        # Transform to (N, 64, hidden_size)
        # x = self.fc(x)

        return x

class CReshapeAndTransform(nn.Module):
    def __init__(self, input_dim, output_dim, num_patches=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, num_patches, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(num_patches, num_patches, kernel_size=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1548, output_dim)

    def forward(self, c):
        batch_size = c.shape[0]
        c_flatten = c.reshape(batch_size, c.shape[1], -1)

        c_flatten = self.conv(c_flatten)
        c_transformed = self.fc(c_flatten)

        return c_transformed


class TimeSeriesPatchEmbedder(nn.Module):
    def __init__(self, input_size, patch_size, in_channels, hidden_size, time_length):
        super(TimeSeriesPatchEmbedder, self).__init__()

        # Create a list of PatchEmbed layers, one for each time step
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
            for _ in range(time_length)
        ])

        # Store references to all weights and biases
        self.weights = [pe.proj.weight.data for pe in self.patch_embeds]
        self.biases = [pe.proj.bias for pe in self.patch_embeds]

        # Total number of patches after embedding
        self.num_patches = self.patch_embeds[0].num_patches * time_length

    def forward(self, x):
        # x has shape (N, t, C, H, W)
        patches = []
        for t in range(x.shape[1]):  # Loop over time steps
            # Extract the slice at time step t
            x_t = x[:, t, :, :, :]  # (N, C, H, W)
            # Apply the corresponding PatchEmbed layer for time step t
            x_t_embed = self.patch_embeds[t](x_t)  # (N, T, D), where T = H * W / patch_size ** 2
            patches.append(x_t_embed)

        # Concatenate along the time dimension (which becomes T dimension)
        x = torch.cat(patches, dim=1)  # Concatenate patches along T dimension

        return x


class PatchEmbed1D(nn.Module):
    """ 1D Data to Patch Embedding
    """

    def __init__(
            self,
            length_size: Optional[int] = 224,  # 对应原来的img_size，改成长度
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

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape  # batch, channels, length
        if self.length_size is not None:
            if self.strict_length_size:
                assert L == self.length_size, f"Input length ({L}) doesn't match model ({self.length_size})."
            else:
                assert L % self.patch_size == 0, f"Input length ({L}) should be divisible by patch size ({self.patch_size})."

        if self.dynamic_length_pad:
            pad_len = (self.patch_size - L % self.patch_size) % self.patch_size
            x = F.pad(x, (0, pad_len))

        x = self.proj(x)  # Apply 1D convolution
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCL -> NLC
        x = self.norm(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,  # 28个DIT Block
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,  #?
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.x_embedder = PatchEmbed1D(input_size, patch_size, in_channels, hidden_size, bias=True)
        # self.x_embedder = TimeSeriesPatchEmbedder(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, time_length=32*30)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # self.c_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # self.c_embedder = CtoPatchEmbed()
        # self.c_embedder = CReshapeAndTransform(input_dim=63, output_dim=768, num_patches=1024)
        self.c_embedder = Adaptor(input_dim=2560, output_dim=hidden_size, num_patches=int(input_size/patch_size))
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([DiTBlockWithCrossAttention(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        # self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.num_patches)  # 使用1D的位置嵌入
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size * C)
        Returns:
        imgs: (N, C, L)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        t = x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, p, c))  # (N, T, patch_size, C)
        imgs = x.permute(0, 3, 1, 2).reshape(shape=(x.shape[0], c, -1))  # (N, C, L)

        return imgs

    # def unpatchify(self, x):
    #     """
    #     Reverse the patch embedding process to reconstruct the original images.
    #
    #     x: (N, T_total, D)  -> Input tensor after patch embedding, where T_total = t * num_patches
    #     Returns:
    #     imgs: (N, t, C, H, W)  -> Reconstructed images
    #     """
    #     # Get parameters
    #     # print(x.shape)
    #     # t = len(self.x_embedder.patch_embeds)  # Number of time steps
    #     t = 32 * 30
    #     p = self.x_embedder.patch_embeds[0].patch_size[0]  # Patch size (assuming square patches)
    #     # c = self.x_embedder.patch_embeds[0].proj.in_channels  # Number of input channels (C)
    #     c = self.out_channels
    #     h = w = int((x.shape[1] // t) ** 0.5)  # Height and width of the patch grid
    #
    #     assert h * w == x.shape[1] // t, "Error: x shape doesn't match the expected number of patches"
    #
    #     # Reshape x back to patches for each time step
    #     x = x.reshape(shape=(x.shape[0], t, h, w, p, p, c))  # (N, t, h, w, p, p, C)
    #
    #     # Reorder dimensions to get back the original image shape
    #     x = torch.einsum('nthwpqc->ntchpwq', x)  # (N, t, C, H, W)
    #
    #     # Combine patches into the original image size
    #     imgs = x.reshape(shape=(x.shape[0], t, c, h * p, w * p))  # (N, t, C, H, W)
    #     return imgs

    # def forward(self, x, t, y):
    #     """
    #     Forward pass of DiT.
    #     x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
    #     t: (N,) tensor of diffusion timesteps
    #     y: (N,) tensor of class labels 改
    #     """
    #     x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
    #     t = self.t_embedder(t)                   # (N, D)
    #     y = self.y_embedder(y, self.training)    # (N, D)
    #     c = t + y                                # (N, D)
    #     for block in self.blocks:
    #         x = block(x, c)                      # (N, T, D)
    #     x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
    #     x = self.unpatchify(x)                   # (N, out_channels, H, W)
    #     return x
    
    def forward(self, x, t, c):  #
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        c: (N, C, H, W) tensor of spatial inputs (images or latent representations of conditional images)
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2, 图片格式变为token格式
        # print(x.shape)
        t = self.t_embedder(t)                   # (N, D)
        # y = self.y_embedder(y, self.training)    # (N, D)
        c = self.c_embedder(c) + self.pos_embed  # (N, T, D)
        # c = c + self.pos_embed
        # all_conditioning = t + y + torch.mean(c, dim=1)  # (N, D)   
        # all_conditioning = t.unsqueeze(1).expand(-1, x.size(1), -1) + y.unsqueeze(1).expand(-1, x.size(1), -1) + c  # (N, T, D)
        all_conditioning = t.unsqueeze(1).expand(-1, x.size(1), -1) + c  # (N, T, D)
        for block in self.blocks:
            x = block(x, all_conditioning)
        # print(x.shape)# (N, T, D)
        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        # print(x.shape)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x
    

    def forward_with_cfg(self, x, t, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiT_Pretrained(nn.Module):
    """
    Diffusion model with a Pretrained ViT backbone.
    """
    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=16,  # 16个experts
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            learn_sigma=True,  # ?
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # self.c_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # self.c_embedder = CtoPatchEmbed()
        self.c_embedder = CReshapeAndTransform(input_dim=62, output_dim=1200, num_patches=2562)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [DiTBlockWithCrossAttention(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        # self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # elif isinstance(module, nn.LayerNorm):
            #     nn.init.constant_(module.bias, 0)
            #     nn.init.constant_(module.weight, 1.0)
            # elif isinstance(module, nn.Dropout):
            #     pass

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, p))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p))
        return imgs

    # def forward(self, x, t, y):
    #     """
    #     Forward pass of DiT.
    #     x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
    #     t: (N,) tensor of diffusion timesteps
    #     y: (N,) tensor of class labels 改
    #     """
    #     x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
    #     t = self.t_embedder(t)                   # (N, D)
    #     y = self.y_embedder(y, self.training)    # (N, D)
    #     c = t + y                                # (N, D)
    #     for block in self.blocks:
    #         x = block(x, c)                      # (N, T, D)
    #     x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
    #     x = self.unpatchify(x)                   # (N, out_channels, H, W)
    #     return x

    def forward(self, x, t, c):  #
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        c: (N, C, H, W) tensor of spatial inputs (images or latent representations of conditional images)
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2, 图片格式变为token格式
        t = self.t_embedder(t)  # (N, D)
        # y = self.y_embedder(y, self.training)    # (N, D)
        c = self.c_embedder(c) + self.pos_embed  # (N, T, D)
        # all_conditioning = t + y + torch.mean(c, dim=1)  # (N, D)
        # all_conditioning = t.unsqueeze(1).expand(-1, x.size(1), -1) + y.unsqueeze(1).expand(-1, x.size(1), -1) + c  # (N, T, D)
        all_conditioning = t.unsqueeze(1).expand(-1, x.size(1), -1) + c  # (N, T, D)
        for block in self.blocks:
            x = block(x, all_conditioning)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_pos_embed(embed_dim, seq_len, cls_token=False, extra_tokens=0):
    """
    seq_len: int of the sequence length
    return:
    pos_embed: [seq_len, embed_dim] or [1+seq_len, embed_dim] (w/ or w/o cls_token)
    """
    position = np.arange(seq_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_position(embed_dim, position)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_position(embed_dim, position):
    """
    embed_dim: embedding dimension
    position: [seq_len]
    return:
    pos_embed: [seq_len, embed_dim]
    """
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega = 1.0 / np.power(10000, omega / (embed_dim // 2))
    pos_embed = np.einsum('n,d->nd', position, omega)
    pos_embed = np.concatenate([np.sin(pos_embed), np.cos(pos_embed)], axis=1)
    return pos_embed

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, length):
    pos_embed = np.zeros((length, embed_dim))
    position = np.arange(0, length, dtype=np.float32)
    div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
    pos_embed[:, 0::2] = np.sin(position[:, None] * div_term)
    pos_embed[:, 1::2] = np.cos(position[:, None] * div_term)
    return pos_embed


def generate_combined_positional_embeddings(embed_dim, grid_size, time_length):
    # Get the 2D positional embedding for spatial dimensions
    pos_embed_2d = get_2d_sincos_pos_embed(embed_dim, grid_size)

    # Get the 1D positional embedding for the time dimension
    pos_embed_1d = get_1d_sincos_pos_embed(embed_dim, time_length)

    # Repeat the 2D embedding for each time step
    pos_embed_2d = pos_embed_2d.repeat(time_length, axis=0)  # shape: (time_length * grid_size * grid_size, embed_dim)

    # Add the time dimension embedding
    pos_embed_1d_repeated = np.repeat(pos_embed_1d, grid_size * grid_size, axis=0)
    combined_pos_embed = pos_embed_2d + pos_embed_1d_repeated

    return combined_pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


if __name__ == "__main__":
    latent_size = 32
    num_classes = 1000

    model = DiT(input_size=latent_size, num_classes=num_classes)

    N = 2
    C = 4
    H = W = latent_size
    T = 100
    x = torch.rand(N, C, H, W)
    t = torch.randint(0, T, (N,))
    y = torch.randint(0, num_classes, (N,))
    c = torch.rand(N, C, H, W)

    with torch.no_grad():
        output = model(x, t, y, c)

    print("Output shape:", output.shape)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

# Can be adjusted to the required modal parameters
def DiT_fMRI(**kwargs):
    return DiT(depth=24, hidden_size=1200, patch_size=4, num_heads=12, **kwargs)

DiT_models = {
    'DiT_fMRI': DiT_fMRI,
}

