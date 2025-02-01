# Copyright (c) Shanghai AI Lab. All rights reserved.
"""
More info:
- https://github.com/facebookresearch/dinov2/issues/84
- https://github.com/czczup/ViT-Adapter/tree/main/segmentation
- https://github.com/czczup/ViT-Adapter/blob/ae47f7b259a87517cffce9019ee8fd492874a9cc/segmentation/mmseg_custom/models/backbones/vit_adapter.py#L20
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch.nn.init import normal_

# from external.ms_deformable_attention.modules import MSDeformAttn
from mmcv.ops import MultiScaleDeformableAttention

from dino_adapter.vit.vision_transformer import DinoVisionTransformer
from dino_adapter.vit_adapter.adapter_modules import (
    InteractionBlock,
    InteractionBlockWithCls,
    SpatialPriorModule,
    deform_inputs,
)

_logger = logging.getLogger(__name__)

__all__ = ['ViTAdapter']

def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"

vit_kwargs = {"img_size":518,
             "patch_size": 14,
             "init_values": 1.00,
             "ffn_layer": "mlp",
             "block_chunks": 0,
            "embed_dim": 384,
             "depth": 12,
             "num_heads": 6,
             "mlp_ratio": 4}

class ViTAdapter(DinoVisionTransformer):
    def __init__(self, pretrain_size=518, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True,
                 use_extra_extractor=True, with_cp=False,
                 vit_arch_name="vit_small", vit_kwargs=vit_kwargs, vit_pretrained=True, upsample_dino_input=False):
        super().__init__(**vit_kwargs)

        model_name = _make_dinov2_model_name(vit_arch_name, vit_kwargs["patch_size"])
        if vit_pretrained:
            url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)

        self.upsample_dino_input = upsample_dino_input

        # Freeze vit
        for param in self.parameters():
            param.requires_grad = False

        # self.num_classes = 80
        self.mask_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

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

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 14, self.pretrain_size[1] // 14, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MultiScaleDeformableAttention):
            m.init_weights()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.upsample_dino_input)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        _, _, h, w = x.shape
        W_adapter = w // 16
        H_adapter = h // 16

        # Patch Embedding forward
        if self.upsample_dino_input:
            x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)

        _, _, h, w = x.shape

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            H_new = (h // self.patch_size) * self.patch_size
            W_new = (w // self.patch_size) * self.patch_size
            x = F.interpolate(x, size=(H_new, W_new), mode='bilinear', align_corners=False)

        x = self.patch_embed(x)
        W_vit = w // self.patch_size
        H_vit = h // self.patch_size
        
        bs, n, dim = x.shape

        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H_vit, W_vit)
        x = x + pos_embed
        cls = self.cls_token.expand(x.shape[0], -1, -1) + self.pos_embed[:, 0]

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(x, c, cls, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H_adapter, W_adapter)
            outs.append(x.transpose(1, 2).view(bs, dim, H_vit, W_vit).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_adapter * 2, W_adapter * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_adapter, W_adapter).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_adapter // 2, W_adapter // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, size=(c1.shape[2], c1.shape[3]), mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, size=(c2.shape[2], c2.shape[3]), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x3, size=(c3.shape[2], c3.shape[3]), mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, size=(c4.shape[2], c4.shape[3]), mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

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
