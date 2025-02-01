import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


from utils.coord_utils import indices_to_coords
from model.modules import DMSMHA_Block, get_deformable_inputs, MHA_Block




class Query_Updater(nn.Module):
    def __init__(self, args, nhead=8):
        super().__init__()

        self.nhead = nhead
        self.size = args.input_size
        self.stride = args.stride
        self.H_prime = self.size[0] // self.stride
        self.W_prime = self.size[1] // self.stride
        self.P = self.H_prime * self.W_prime
        self.embedding_dim = args.transformer_embedding_dim
        self.num_layers = args.num_layers
        self.memory_size = args.memory_size

        self.num_level = 3 if self.stride == 8 else 4

        self.random_mask_ratio = 0.0


        self.K1 = args.num_layers_spatial_writer 
        self.K2 = args.num_layers_spatial_self
        self.K3 = args.num_layers_spatial_cross

        # <Query to Feature Map>
        if self.K1 > 0:
            self.query_to_feature = []
            for l in range(self.K1):
                decoder_layer = DMSMHA_Block(self.embedding_dim, nhead, self.num_level)
                self.query_to_feature.append(decoder_layer)
            self.query_to_feature = nn.ModuleList(self.query_to_feature)
        # </Query to Feature Map>

        # <Memory to Memory>
        if self.K2 > 0:
            self.memory_to_memory = []
            for l in range(self.K2):
                decoder_layer = MHA_Block(self.embedding_dim, nhead)
                self.memory_to_memory.append(decoder_layer)
            self.memory_to_memory = nn.ModuleList(self.memory_to_memory)
        # </Memory to Memory>

        # <Feature Map to Query>
        if self.K3 > 0:
            self.query_to_memory = []
            for l in range(self.K3):
                decoder_layer = MHA_Block(self.embedding_dim, nhead)
                self.query_to_memory.append(decoder_layer)
            self.query_to_memory = nn.ModuleList(self.query_to_memory)
        # </Feature Map to Query>

        self.augment_layer = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.query_residual = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.time_embedding = nn.Parameter(torch.zeros(1, self.memory_size + 1, self.embedding_dim))
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def get_augmented_memory(self, q_init, q_t, f_t, target_coordinates, query_times, t):
        # :args q_init              (q_init): (B, N, C)
        # :args q_t                 (q_t): (B, N, C)
        # :args f                   (F): (B, T, P, C)
        # :args target_coordinates (B, N, 2)
        # :args query_times        (B, N)
        # :args t                  (B, N)
        #
        # :return q_augmented: (B, N, C)

        B, N, C = q_init.shape
        device = q_init.device

        # use the updated memory for this mask
        queried_now_or_before = (query_times <= t).unsqueeze(-1).expand(-1, -1, C)  # (B, N, C)

        # q_aug = self.augment_layer(torch.cat([q_t, q_init, q_start_prev], dim=-1))  # (B, N, C)
        q_aug = self.augment_layer(torch.cat([q_t, q_init], dim=-1))  # (B, N, C)

        target_coordinates = target_coordinates / torch.tensor([self.size[1], self.size[0]], device=device)  # (B, N, 2)
        target_coordinates = torch.clamp(target_coordinates, 0, 1)
        f_scales, reference_points, spatial_shapes, start_levels = get_deformable_inputs(f_t, target_coordinates, self.H_prime, self.W_prime)
        
        if self.K1 > 0:
            for i in range(self.K1):
                q_aug = self.query_to_feature[i](q=q_aug, k=f_scales, v=f_scales,
                                                reference_points=reference_points, 
                                                spatial_shapes=spatial_shapes, 
                                                start_levels=start_levels)


        q_aug = torch.where(queried_now_or_before, q_aug, q_init.clone())  # (B, N, C)
        return q_aug

    def forward(self, q_init, past_aug_q, past_mask, past_vis, query_times, t):
        # :args q_init              (q_init): (B, N, C)
        #
        # :args q_prev              (q_t): (B, N, C)
        #
        # :args q_start_prev        (q^{start}_t): (B, N, C)
        # :args past_mask:          (B, N, memory_size), True if not queried yet at that frame
        # :args past_vis:           (B, N, memory_size), True if occluded at that frame
        #
        # :args past_aug_q          (q^{aug}_t): (B, N, memory_size, C)
        #
        # :args f                   (F): (B, T, P, C)
        # :args target_coordinates  (p_t): (B, N, 2), in pixel space
        # :args query_times             : (B, N)
        # :args t                       : (B, N)
        #
        # :return q_start: (B, N, C)


        B, N, C = q_init.shape
        device = q_init.device

        past_mask = past_mask[:, :, -self.memory_size:]     # (B, N, memory_size)
        past_vis = past_vis[:, :, -self.memory_size:]       # (B, N, memory_size)
        past_aug_q = past_aug_q[:, :, -self.memory_size:]   # (B, N, memory_size, C)

        queried_before = (query_times < t)
        queried_before = queried_before.unsqueeze(-1).expand(-1, -1, C)        # (B, N, C)

        if t == 0:
            return q_init.clone()

        else:

            ignore_mask = past_mask | past_vis                                    # (B, N, memory_size)
            if self.random_mask_ratio > 0:
                random_mask = torch.rand(B, N, self.memory_size, device=device) < self.random_mask_ratio
                ignore_mask = ignore_mask | random_mask

            fully_ignored = ignore_mask.all(dim=-1)                               # (B, N)

            if fully_ignored.all():
                return q_init.clone()

            q = q_init.view(B * N, 1, C)                # (B * N, 1, C)
            q_start = q.clone()

            kv = past_aug_q.view(B * N, -1, C)              # (B * N, memory_size, C)
            mask = ignore_mask.view(B * N, -1)               # (B * N, memory_size)
            fully_ignored = fully_ignored.view(B * N)        # (B * N)

            q = q[~fully_ignored]                        # (useful_query_num, 1, C)
            kv = kv[~fully_ignored]                      # (useful_query_num, memory_size, C)
            mask = mask[~fully_ignored]                  # (useful_query_num, memory_size)

            # <Memory to Memory>
            if self.K2 > 0:
                for i in range(self.K2):
                    kv_pos = kv + self.time_embedding[:, :-1]                                   # (useful_query_num, memory_size, C)
                    kv = self.memory_to_memory[i](q=kv_pos, k=kv_pos, v=kv, mask=mask)          # (useful_query_num, 1, C)
            # </Memory to Memory>


            # <Memory to Query>
            q = q + self.time_embedding[:, -1].unsqueeze(1)     # (useful_query_num, 1, C)
            k = kv + self.time_embedding[:, :-1]                # (useful_query_num, memory_size, C)
            v = kv

            if self.K3 > 0:
                for i in range(self.K3):
                    q = self.query_to_memory[i](q=q, k=k, v=v, mask=mask)

            # </Memory to Query>

            q_start[~fully_ignored] = q_start[~fully_ignored] + self.query_residual(q)  # (useful_query_num, 1, C)
            q_start = q_start.view(B, N, C)

            return q_start