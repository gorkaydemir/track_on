import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from model.modules import DMSMHA_Block, get_deformable_inputs



class Offset_Head(nn.Module):
    def __init__(self, args, nhead=8):
        super().__init__()

        self.nhead = nhead
        self.attn_layer_num = args.num_layers_offset_head
        self.size = args.input_size
        self.stride = args.stride

        self.H_prime = self.size[0] // self.stride
        self.W_prime = self.size[1] // self.stride
        self.P = self.H_prime * self.W_prime

        self.embedding_dim = args.transformer_embedding_dim

        self.offset_layer = nn.Sequential(nn.LayerNorm(self.embedding_dim),
                                            nn.Linear(self.embedding_dim, 2),
                                            nn.Tanh())

        self.local_transformer = []
        self.num_level = 4

        for _ in range(self.attn_layer_num):
            decoder_layer = DMSMHA_Block(self.embedding_dim, nhead, self.num_level)
            self.local_transformer.append(decoder_layer)
        self.local_transformer = nn.ModuleList(self.local_transformer)


    def forward(self, q_t, f_t, target_coordinates):
        # :args q_t: (B, N, C)
        # :args f_t: (B, P, C)
        # :args target_coordinates: (B, N, 2), in size range
        #
        # :return o_t: (B, layer_num, N, 2)

        B, N, C = q_t.shape
        device = q_t.device

        # <Target scaling>        
        target_coordinates = target_coordinates / torch.tensor([self.size[1], self.size[0]], device=device)  # (B, N, 2)
        target_coordinates = torch.clamp(target_coordinates, 0, 1)
        #   </Target scaling>

        f_scales, reference_points, spatial_shapes, start_levels = get_deformable_inputs(f_t, target_coordinates, self.H_prime, self.W_prime)

        o_t = torch.zeros(B, self.attn_layer_num, N, 2, device=device)
        for i in range(self.attn_layer_num):
            q_t = self.local_transformer[i](q=q_t, k=f_scales, v=f_scales, 
                                            reference_points=reference_points, 
                                            spatial_shapes=spatial_shapes, 
                                            start_levels=start_levels)
            o_t_i = self.offset_layer(q_t) * self.stride                     # (B, N, 2), in (-stride, stride), in pixel space

            o_t[:, i] = o_t_i

        return o_t
