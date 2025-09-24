import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.ops

from mmcv.ops import MultiScaleDeformableAttention


class MHA_Block(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_drop=0.1, mlp_drop=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.attn_drop = attn_drop

        self.mha = nn.MultiheadAttention(hidden_size, num_heads, attn_drop, batch_first=True)

        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.dropout = nn.Dropout(mlp_drop)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(mlp_drop)
        self.dropout2 = nn.Dropout(mlp_drop)

        norm_cls = nn.LayerNorm
        self.norm1 = norm_cls(hidden_size)
        self.norm2 = norm_cls(hidden_size)

        self.act = nn.ReLU(inplace=True)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (B, T, C), mask: (B, P), where True = ignore
        B, N, C = q.shape
        _, P, _ = k.shape

        if mask is not None:
            assert mask.shape == (B, P), f"Mask shape: {mask.shape} expected {(B, P)}"
            full_mask = mask.all(dim=1)  # (B,)

        q_orig = q
        attn_out, _ = self.mha(q, k, v, key_padding_mask=mask, need_weights=False)

        drop = self.dropout1(attn_out)

        if mask is not None and full_mask.any():
            drop[full_mask] = 0

        q = q_orig + drop
        q = self.norm1(q)

        ffn = self.linear2(self.dropout(self.act(self.linear1(q))))
        q = q + self.dropout2(ffn)
        q = self.norm2(q)


        return q
    
# Deformable Multi Scale Attention Block
class DMSMHA_Block(nn.Module):
    def __init__(self, hidden_size, num_heads, num_levels, p_drop=0.1):
        super().__init__()


        self.mha = MultiScaleDeformableAttention(embed_dims=hidden_size,
                                                 num_heads=num_heads,
                                                 num_levels=num_levels,
                                                 num_points=4,
                                                 dropout=p_drop,
                                                 batch_first=True)


        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)

        self.dropout1 = nn.Dropout(p_drop)
        self.dropout2 = nn.Dropout(p_drop)

        norm_cls = nn.LayerNorm
        self.norm1 = norm_cls(hidden_size)
        self.norm2 = norm_cls(hidden_size)

        self.act = nn.ReLU(inplace=True)


    def forward(self, q, k, v, reference_points, spatial_shapes, start_levels):
        # :args q: (B, N, C)
        # :args kv: (B, P, C)
        # :args reference_points: (B, N, num_level, 2)
        # :args spatial_shapes: (num_level, 2)
        # :args start_levels: (num_level,)
        

        B, N, C = q.shape
        B, P, C = k.shape
        device = q.device

        q = q + self.dropout1(self.mha(q, k, v, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=start_levels))
        q = self.norm1(q)
        q = q + self.dropout2(self.linear2(self.dropout(self.act(self.linear1(q)))))
        q = self.norm2(q)

        return q


class SimpleFPN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.D = args.D
        self.size = args.input_size
        self.stride = 4

        self.H = self.size[0]
        self.W = self.size[1]
        self.Hf = self.H // self.stride
        self.Wf = self.W // self.stride
        self.P = int(self.Hf * self.Wf)                     # Number of tokens

        def block(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, self.D, kernel_size=1),
                nn.GroupNorm(32, self.D),
                nn.ReLU()
            )

        self.lateral32 = block(self.D)
        self.lateral16 = block(self.D)
        self.lateral8 = block(self.D)

        self.output = nn.Sequential(nn.Conv2d(self.D, self.D, kernel_size=3, padding=1),
                                    nn.GroupNorm(32, self.D),
                                    nn.ReLU(),
                                    nn.Conv2d(self.D, self.D, kernel_size=3, padding=1))


    def forward(self, f4, f8, f16, f32):
        # :args f4: (B, T, P, D)
        # :args f8: (B, T, P // 4, D)
        # :args f16: (B, T, P // 16, D)
        # :args f32: (B, T, P // 64, D)
        #
        # :return out: (B, T, P, D)

        B, T, _, _ = f4.shape

        f4 = f4.permute(0, 1, 3, 2).reshape(B * T, self.D, self.Hf, self.Wf)               # (B * T, D, Hf, Wf)
        f8 = f8.permute(0, 1, 3, 2).reshape(B * T, self.D, self.Hf // 2, self.Wf // 2)     # (B * T, D, Hf // 2, Wf // 2)
        f16 = f16.permute(0, 1, 3, 2).reshape(B * T, self.D, self.Hf // 4, self.Wf // 4)   # (B * T, D, Hf // 4, Wf // 4)
        f32 = f32.permute(0, 1, 3, 2).reshape(B * T, self.D, self.Hf // 8, self.Wf // 8)   # (B * T, D, Hf // 8, Wf // 8)

        f32_lat = self.lateral32(f32)                                                                   # (B * T, D, Hf // 8, Wf // 8)
        f16_lat = self.lateral16(f16) + F.interpolate(f32_lat, size=f16.shape[-2:], mode="nearest")     # (B * T, D, Hf // 4, Wf // 4)
        f8_lat = self.lateral8(f8) + F.interpolate(f16_lat, size=f8.shape[-2:], mode="nearest")         # (B * T, D, Hf // 2, Wf // 2)
        f4_out = f4 + F.interpolate(f8_lat, size=f4.shape[-2:], mode="nearest")                         # (B * T, D, Hf, Wf)

        f4_out = self.output(f4_out)                                                                    # (B * T, D, Hf, Wf)
        f4_out = f4_out.permute(0, 2, 3, 1).reshape(B, T, self.P, self.D)                               # (B, T, P, D)

        return f4_out
