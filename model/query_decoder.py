import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


from utils.coord_utils import indices_to_coords
from model.modules import DMSMHA_Block, get_deformable_inputs, MHA_Block



class Query_Decoder(nn.Module):
    def __init__(self, args, nhead=8):
        super().__init__()

        self.nhead = nhead
        self.num_layers = args.num_layers
        self.size = args.input_size
        self.stride = args.stride
        self.H_prime = self.size[0] // self.stride
        self.W_prime = self.size[1] // self.stride
        self.P = self.H_prime * self.W_prime

        self.embedding_dim = args.transformer_embedding_dim
        self.random_mask_ratio = 0.0

        # <Token to Query>
        self.token_to_query = []
        for _ in range(self.num_layers):
            decoder_layer = MHA_Block(self.embedding_dim, nhead)
            self.token_to_query.append(decoder_layer)
        self.token_to_query = nn.ModuleList(self.token_to_query)
        # </Token to Query>

        # <Query to Query>
        self.query_to_query = []
        for _ in range(self.num_layers):
            decoder_layer = MHA_Block(self.embedding_dim, nhead)
            self.query_to_query.append(decoder_layer)
        self.query_to_query = nn.ModuleList(self.query_to_query)
        # </Query to Query>

        # <Memory to Query>
        self.memory_size = args.memory_size

        self.memory_to_query = []
        for _ in range(self.num_layers):
            decoder_layer = MHA_Block(self.embedding_dim, nhead)
            self.memory_to_query.append(decoder_layer)
        self.memory_to_query = nn.ModuleList(self.memory_to_query)

        self.time_embedding = nn.Parameter(torch.zeros(1, self.memory_size + 1, self.embedding_dim))

        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        # </Memory to Query>


    def memory_forward(self, q_t, past_q, past_q_mask, iter_num):
        # :args q_t: (B, N, C)
        # :args past_q: (B, N, memory_size, C)
        # :args past_q_mask: (B, N, memory_size), True if wanted to be masked

        B, N, C = q_t.shape

        q_t = q_t.view(B * N, 1, C)                # (B * N, 1, C)
        past_q = past_q.view(B * N, -1, C)         # (B * N, memory_size, C)
        past_q_mask = past_q_mask.view(B * N, -1)  # (B * N, memory_size)

        # True if all masked
        all_masked = past_q_mask.all(dim=-1)       # (B * N)

        if all_masked.all():
            return q_t.view(B, N, C), past_q.view(B, N, -1, C)
        
        useful_query_num = (~all_masked).sum()
        qkv = torch.cat([past_q[~all_masked], q_t[~all_masked]], dim=1)                    # (useful_query_num, memory_size + 1, C)
        mask = torch.cat([past_q_mask[~all_masked], 
                          torch.zeros(useful_query_num, 1).bool().to(q_t.device)], dim=1)  # (useful_query_num, memory_size + 1)

        q = qkv + self.time_embedding
        k = qkv + self.time_embedding
        v = qkv

        qkv = self.memory_to_query[iter_num](q=q, k=k, v=v, mask=mask)  # (useful_query_num, memory_size + 1, C)

        q_t[~all_masked] = qkv[:, -1].unsqueeze(1)      # (useful_query_num, 1, C)
        past_q[~all_masked] = qkv[:, :-1]               # (useful_query_num, memory_size, C)

        return q_t.view(B, N, C), past_q.view(B, N, -1, C)


    def forward(self, q_start, f_t, past_q, past_q_mask, queried_now_or_before):
        # :args q_start_t: (B, N, C)
        # :args f_t: (B, P, C)
        # :args past_q: (B, N, memory_size, C)
        # :args past_q_mask: (B, N, memory_size), True if wanted to be masked
        # :args queried_now_or_before: (B, N),    True if at this time or before
        #
        # :return q_t: (B, N, C)

        B, N, C = q_start.shape
        P = f_t.size(1)
        device = q_start.device


        past_q = past_q[:, :, -self.memory_size:]               # (B, N, memory_size)
        past_q_mask = past_q_mask[:, :, -self.memory_size:]     # (B, N, memory_size)
        
        sa_mask = ~queried_now_or_before                                        # (B, N), True if will be queried later

        if self.random_mask_ratio > 0:
            random_mask = torch.rand(B, N, past_q_mask.size(-1), device=device) < self.random_mask_ratio
            past_q_mask = past_q_mask | random_mask

            random_mask = torch.rand(B, N, device=device) < self.random_mask_ratio
            sa_mask = sa_mask | random_mask
    

        memory = past_q
    
        q_t = q_start
        for i in range(self.num_layers):
            # <Token to Query>
            q = q_t
            k = f_t
            v = f_t
            q_t = self.token_to_query[i](q, k, v)     # (B, N, C)
            # </Token to Query>

            # <Query to Query>
            q = q_t
            k = q_t
            v = q_t
            q_t = self.query_to_query[i](q, k, v, mask=sa_mask)
            # </Query to Query>

            # <Context Memory to Query>
            q_t, memory = self.memory_forward(q_t, memory, past_q_mask, i)
            # </Context Memory to Query>


        return q_t
