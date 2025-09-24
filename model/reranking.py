import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


from utils.coord_utils import indices_to_coords
from model.modules import DMSMHA_Block, MHA_Block


class Rerank_Module(nn.Module):
    def __init__(self, args, nhead=8):
        super().__init__()

        self.K = args.K
        self.D = args.D
        self.size = args.input_size
        self.stride = 4
        self.nhead = nhead

        self.H = self.size[0]
        self.W = self.size[1]
        self.Hf = self.H // self.stride
        self.Wf = self.W // self.stride
        self.P = int(self.Hf * self.Wf)               

        self.reranking_layer_num = args.rerank_layer_num

        # === Top-K Point Decoder ===
        self.num_level = 4

        self.local_decoder = []
        for _ in range(self.reranking_layer_num):
            decoder_layer = DMSMHA_Block(self.D, nhead, self.num_level)
            self.local_decoder.append(decoder_layer)
        self.local_decoder = nn.ModuleList(self.local_decoder)
        # === === ===

        self.fusion = MHA_Block(self.D, nhead)

        self.fusion_layer = nn.Linear(2 * self.D, self.D)
        self.final_projection_layer = nn.Linear(2 * self.D, self.D)
        self.certainty_layer = nn.Linear(self.D, 1)
        self.score_layer = nn.Linear(self.D, 1)

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



    def forward(self, q_t, f4, f8, f16, f32, c_t):
        # :args q_t: (1, N_t, D)
        # :args f4: (1, P, D)
        # :args f8: (1, P // 4, D)
        # :args f16: (1, P // 16, D)
        # :args f32: (1, P // 32, D)
        # :args c_t: (1, N_t, P), in size range

        # :return q_t: (1, N_t, D)
        # :return p_patch_top_k: (1, N_t, K, 2)
        # :return u_logit_top_k: (1, N_t, K)
        # :return s_logit_top_k: (1, N_t, K)

        N_t = q_t.shape[1]
        _, P, D = f4.shape
        K = self.K
        device = q_t.device
        assert P == self.P, f"Expected P to be {self.P}, but got {P}"

        # === Top-k Indices ===
        top_k_indices = torch.topk(c_t, K, dim=-1)[1]                             # (1, N_t, K)
        p_patch_top_k = indices_to_coords(top_k_indices, self.size, self.stride)  # (1, N_t, K, 2), in [0, W] and [0, H] range
        # === === ===

        # === Local Decoder ===
        p_patch_top_k_norm = p_patch_top_k / torch.tensor([self.size[1], self.size[0]], device=device)  # (1, N_t, K, 2), in [0, 1] range
        p_patch_top_k_norm = torch.clamp(p_patch_top_k_norm, 0, 1)
        p_patch_top_k_norm = p_patch_top_k_norm.view(1, N_t * K, 1, 2)              # (1, N_t * K, 1, 2)
        p_patch_top_k_norm = p_patch_top_k_norm.expand(-1, -1, self.num_level, -1)  # (1, N_t * K, 4, 2)

        f_scales = torch.cat([f4, f8, f16, f32], dim=1)                          # (1, P + P // 4 + P // 16 + P // 64, D)

        q_top_k = q_t.unsqueeze(2).expand(-1, -1, K, -1).reshape(1, N_t * K, D)  # (1, N_t * K, D)

        for i in range(self.reranking_layer_num):
            q_top_k = self.local_decoder[i](q=q_top_k, k=f_scales, v=f_scales, 
                                            reference_points=p_patch_top_k_norm, 
                                            spatial_shapes=self.spatial_shapes, 
                                            start_levels=self.start_levels)
        q_top_k = q_top_k.view(1, N_t, K, D)
        q_top_k = torch.cat([q_top_k, q_t.unsqueeze(2).expand(-1, -1, K, -1)], dim=-1)                     # (1, N_t, K, 2 * D)
        q_top_k = self.fusion_layer(q_top_k)                                                               # (1, N_t, K, D)
        # === === ===

        # === Fusion and Prediction ===
        u_logit_top_k = self.certainty_layer(q_top_k).squeeze(-1) # (1, N_t, K)
        s_logit_top_k = self.score_layer(q_top_k).squeeze(-1)     # (1, N_t, K)

        q_top_k = q_top_k.view(N_t, K, D)  # (N_t, K, D)
        q_t = q_t.view(N_t, 1, D)          # (N_t, 1, D)
        q_t_init = q_t.clone()             # (N_t, 1, D)

        q_t = self.fusion(q_t, q_top_k, q_top_k)                               # (N_t, 1, D)
        q_t = self.final_projection_layer(torch.cat([q_t, q_t_init], dim=-1))  # (N_t, 1, D)
        q_t = q_t.view(1, N_t, D)  # (1, N_t, D)

        return q_t, p_patch_top_k, u_logit_top_k, s_logit_top_k




