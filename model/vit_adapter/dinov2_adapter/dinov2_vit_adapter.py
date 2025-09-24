# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch.nn.init import normal_
from transformers import AutoModel

from functools import partial
from mmcv.ops import MultiScaleDeformableAttention


from model.vit_adapter.dinov2_adapter.dinov2_adapter_modules import (
    InteractionBlockWithCls,
    SpatialPriorModule,
    deform_inputs,
)

__all__ = ['ViTAdapter']

class ViTAdapter(nn.Module):
    def __init__(self, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, use_extra_extractor=True, with_cp=False, vit_upsample_factor=1.0, arch_type="s"):
        super().__init__()


        if arch_type == "s":
            pretrained_model_name = "facebook/dinov2-small"
        elif arch_type == "b":
            pretrained_model_name = "facebook/dinov2-base"
        else:
            raise ValueError(f"Unknown arch_type: {arch_type}")
            
        self.dinov2 = AutoModel.from_pretrained(pretrained_model_name, device_map=None)
        self.patch_size = 14

        # Freeze vit
        for param in self.dinov2.parameters():
            param.requires_grad = False

        self.vit_upsample_factor = vit_upsample_factor
        self.num_block = len(self.dinov2.encoder.layer)
        self.interaction_indexes = interaction_indexes
        self.drop_path_rate = self.dinov2.encoder.config.drop_path_rate
        embed_dim = self.dinov2.encoder.config.hidden_size
        self.embed_dim = embed_dim

        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, 
                                      embed_dim=embed_dim, 
                                      with_cp=False)
        
        self.interactions = nn.Sequential(*[
            InteractionBlockWithCls(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)

        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)


        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def init_weights(self) -> None:
        pass

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MultiScaleDeformableAttention):
            m.init_weights()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        x_adapter = x
        if self.vit_upsample_factor != 1.0:
            H_new = int(x.shape[2] * self.vit_upsample_factor)
            H_new = (H_new // self.patch_size) * self.patch_size
            W_new = int(x.shape[3] * self.vit_upsample_factor)
            W_new = (W_new // self.patch_size) * self.patch_size
            x_vit = F.interpolate(x, size=(H_new, W_new), mode='bilinear', align_corners=False)

        else:
            H_new = (x.shape[2] // self.patch_size) * self.patch_size
            W_new = (x.shape[3] // self.patch_size) * self.patch_size
            x_vit = F.interpolate(x, size=(H_new, W_new), mode='bilinear', align_corners=False)


        vit_size = (x_vit.shape[2], x_vit.shape[3])
        adapter_size = (x_adapter.shape[2], x_adapter.shape[3])
        device = x.device

        H_adapter = adapter_size[0] // 16
        W_adapter = adapter_size[1] // 16

        deform_inputs1, deform_inputs2 = deform_inputs(vit_size, adapter_size, device)

        c1, c2, c3, c4 = self.spm(x_adapter)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        embeddings = self.dinov2.embeddings(x_vit)    # [bs, n + 1, dim]
        cls_token = embeddings[:, 0, :].unsqueeze(1)  # [bs, 1, dim]
        patch_embeddings = embeddings[:, 1:, :]       # [bs, n, dim]
        
        bs, n, dim = patch_embeddings.shape

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            patch_embeddings, c, cls_token = layer(patch_embeddings, c, cls_token, 
                                                    self.dinov2.encoder.layer[indexes[0]:indexes[-1] + 1], 
                                                    deform_inputs1, deform_inputs2, H_adapter, W_adapter)
            
        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_adapter * 2, W_adapter * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_adapter, W_adapter).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_adapter // 2, W_adapter // 2).contiguous()
        c1 = self.up(c2) + c1

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


class Identity(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

    def forward(self, x):
        return x