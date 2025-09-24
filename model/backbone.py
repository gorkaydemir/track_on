import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.D = args.D 
        self.size = args.input_size
        self.stride = 4

        self.with_cp = args.grad_checkpoint
        self.vit_upsample_factor = args.vit_upsample_factor

        model_name = args.vit_backbone[:6]  # "dinov2" or "dinov3"
        arch_type = args.vit_backbone[7:]   # "s", "b", or "s_plus"

        if model_name == "dinov3":
            from model.vit_adapter.dinov3_adapter.dinov3_vit_adapter import ViTAdapter
            self.vit_encoder = ViTAdapter(deform_num_heads=12 if arch_type == "b" else 6,
                                          with_cp=self.with_cp,
                                          vit_upsample_factor=self.vit_upsample_factor,
                                          arch_type=arch_type)
            
        else:
            from model.vit_adapter.dinov2_adapter.dinov2_vit_adapter import ViTAdapter
            self.vit_encoder = ViTAdapter(deform_num_heads=12 if arch_type == "b" else 6,
                                          with_cp=self.with_cp,
                                          vit_upsample_factor=self.vit_upsample_factor,
                                          arch_type=arch_type)
        
        self.adapter_dim = self.vit_encoder.embed_dim

        # modulelist
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(self.adapter_dim, self.D)),
            nn.Sequential(nn.Linear(self.adapter_dim, self.D)),
            nn.Sequential(nn.Linear(self.adapter_dim, self.D)),
            nn.Sequential(nn.Linear(self.adapter_dim, self.D))
        ])

        # === Positional embedding ===

        self.H = self.size[0]
        self.W = self.size[1]
        self.Hf = self.H // self.stride
        self.Wf = self.W // self.stride
        self.P = int(self.Hf * self.Wf)                     # Number of tokens

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.D, self.Hf, self.Wf))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 3, 1, 1),
                persistent=False,
            )

    def project_feat(self, feat, idx):
        # :args feat: (B * T, self.adapter_dim, Hs, Ws)
        # :args scale: 1, 2, 4, 8

        BT, _, Hs, Ws = feat.shape

        feat = feat.permute(0, 2, 3, 1).reshape(BT, -1, self.adapter_dim)                                  # (B * T, Hs * Ws, 384)
        feat = self.projections[idx](feat)                                                    # (B * T, Hs * Ws, D)
        feat = feat.permute(0, 2, 1).reshape(BT, self.D, Hs, Ws)                              # (B * T, D, Hs, Ws)
        return feat

        
    def forward(self, video):
        # :args video: (B, T, C, self.H, self.W) in range [0, 1]
        # :args queries: (B, N, 3) where 3 is (t, x, y)
        #
        # :return f4, f8, f16, f32: (B, T, P, D), (B, T, P // 4, D), (B, T, P // 16, D), (B, T, P // 64, D)

        B, T, _, H, W = video.shape
        D = self.D
        assert H == self.H and W == self.W, f"Input video size {H}x{W} does not match the expected size {self.H}x{self.W}"

        video_flat = video.view(B * T, 3, H, W) # * 2 - 1.0                           # to [-1, 1]
        video_flat = (video_flat - self._resnet_mean) / self._resnet_std              # Normalize to [-1, 1] range

        # === Extract features ===
        f4, f8, f16, f32 = self.vit_encoder(video_flat)               # (B * T, 384, H4, W4)

        pos4 = self.pos_embedding                                                                               # (1, D, H4, W4)
        pos8 = F.interpolate(self.pos_embedding, scale_factor=0.5, mode='bilinear', align_corners=False)        # (1, D, H8, W8)
        pos16 = F.interpolate(self.pos_embedding, scale_factor=0.25, mode='bilinear', align_corners=False)      # (1, D, H16, W16)
        pos32 = F.interpolate(self.pos_embedding, scale_factor=0.125, mode='bilinear', align_corners=False)     # (1, D, H32, W32)

        f4 = self.project_feat(f4, idx=0) + pos4                                                 # (B * T, D, H4, W4)
        f8 = self.project_feat(f8, idx=1) + pos8                                                 # (B * T, D, H8, W8)           
        f16 = self.project_feat(f16, idx=2) + pos16                                              # (B * T, D, H16, W16)
        f32 = self.project_feat(f32, idx=3) + pos32                                              # (B * T, D, H32, W32)
        # === === ===

        # === Reshape & Permute ===
        f4 = f4.permute(0, 2, 3, 1).view(B, T, self.P, D)                    # (B, T, P, D)
        f8 = f8.permute(0, 2, 3, 1).view(B, T, self.P // 4, D)               # (B, T, P // 4, D)
        f16 = f16.permute(0, 2, 3, 1).view(B, T, self.P // 16, D)            # (B, T, P // 16, D)
        f32 = f32.permute(0, 2, 3, 1).view(B, T, self.P // 64, D)            # (B, T, P // 64, D)
        
        return f4, f8, f16, f32

