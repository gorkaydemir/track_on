
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from model.modules import DMSMHA_Block

class Prediction_Head(nn.Module):
    def __init__(self, args, nhead):
        super().__init__()

        self.nhead = nhead
        self.layer_num = args.predicton_head_layer_num 
        self.size = args.input_size
        self.stride = 4
        self.Hf = self.size[0] // self.stride
        self.Wf = self.size[1] // self.stride
        self.P = self.Hf * self.Wf

        self.D = args.D

        # === Offset Transformer ===
        self.num_level = 4

        self.offset_transformer = []
        for _ in range(self.layer_num):
            decoder_layer = DMSMHA_Block(self.D, nhead, self.num_level)
            self.offset_transformer.append(decoder_layer)
        self.offset_transformer = nn.ModuleList(self.offset_transformer)

        self.offset_layer = nn.Sequential(nn.LayerNorm(self.D),
                                            nn.Linear(self.D, 2), 
                                            nn.Tanh())
        # === === ===

        # === Visibility Transformer ===
        self.vis_layer = nn.Sequential(nn.Linear(self.D, self.D), nn.ReLU(), nn.Linear(self.D, 1))
        self.unc_layer = nn.Sequential(nn.Linear(self.D, self.D), nn.ReLU(), nn.Linear(self.D, 1))
        # === === ===

        # Deformable inputs into buffers
        spatial_shapes = torch.tensor([(self.Hf, self.Wf), 
                                            (self.Hf // 2, self.Wf // 2), 
                                            (self.Hf // 4, self.Wf // 4), 
                                            (self.Hf // 8, self.Wf // 8)]) # (4, 2)
        self.register_buffer("spatial_shapes", spatial_shapes, persistent=False)

        
        start_levels = torch.tensor([0, 
                                     self.P, 
                                     self.P + self.P // 4,  
                                     self.P + self.P // 4 + self.P // 16]) # (4)
        self.register_buffer("start_levels", start_levels, persistent=False)


    def forward(self, q_t, h4, h8, h16, h32, target_coordinates):
        # :args q_t: (1, N, D)
        # :args h4: (1, P, D)
        # :args f8: (1, P // 4, D)
        # :args f16: (1, P // 16, D)
        # :args f32: (1, P // 32, D)
        # :args target_coordinates: (1, N, 2), in size range
        #
        # :return o_t: (#layers, N, 2)
        # :return v_logit: (N), visibility logits
        # :return u_logit: (N), uncertainty logits

        _, N, D = q_t.shape
        device = q_t.device
        layer_num = self.layer_num

        # Scale the target coordinates to [0, 1] range    
        target_coordinates = target_coordinates / torch.tensor([self.size[1], self.size[0]], device=device)  # (1, N, 2)
        target_coordinates = torch.clamp(target_coordinates, 0, 1)
        reference_points = target_coordinates.unsqueeze(2).expand(-1, -1, 4, -1)                            # (1, N, 4, 2)

        f_scales = torch.cat([h4, h8, h16, h32], dim=1)          # (1, P + P // 4 + P // 16 + P // 64, D)

        q_t_init = q_t.clone()                             # (1, N, D)
        q_t_offset = q_t.clone()                           # (1, N, D)

        o_t = torch.zeros((layer_num, N, 2), device=device) 

        for i in range(self.layer_num):
            q_t_offset = self.offset_transformer[i](q=q_t_offset, k=f_scales, v=f_scales, 
                                                    reference_points=reference_points, 
                                                    spatial_shapes=self.spatial_shapes, 
                                                    start_levels=self.start_levels)

            o_t_i = self.offset_layer(q_t_offset).squeeze(0) * self.stride      # (N, 2), in (-stride, stride), in pixel space
            o_t[i] = o_t_i

        v_logit = self.vis_layer(q_t_init).squeeze(-1).squeeze(0)   # (N), visibility logits
        u_logit = self.unc_layer(q_t_init).squeeze(-1).squeeze(0)   # (N), uncertainty logits


        return o_t, v_logit, u_logit


