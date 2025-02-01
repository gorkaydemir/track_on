import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from dino_adapter.dino_vit_adapter import ViTAdapter


class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding_dim = args.transformer_embedding_dim 
        encoder_dim = 384
        self.size = args.input_size
        self.stride = args.stride

        self.vit_encoder = ViTAdapter(add_vit_feature=False, 
                                    use_extra_extractor=True)      # initially, False, True
        self.token_projection = nn.Conv2d(encoder_dim, self.embedding_dim, kernel_size=1, stride=1, padding=0)


        # === Data Normalization ===
        self.normalization = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # === Positional embedding ===

        self.P = (self.size[0] // self.stride) * (self.size[1] // self.stride)
        self.H_prime = self.size[0] // self.stride
        self.W_prime = self.size[1] // self.stride

        self.frame_pos_embedding = nn.Parameter(torch.zeros(1, self.embedding_dim, self.H_prime, self.W_prime))  # (1, D, H, W)
        nn.init.trunc_normal_(self.frame_pos_embedding, std=0.02)


    def get_query_tokens(self, features, queries):
        # :args features: (B * T, C, H4, W4)
        # :args queries: (B, N, 3) where 3 is (t, y, x)
        #
        # :return queries: (B, N, C)

        B, N, _ = queries.shape
        C = features.shape[1]
        device = features.device

        assert C == self.embedding_dim, f"Token embedding dim: {C} expected: {self.embedding_dim}"

        H4, W4 = features.shape[-2], features.shape[-1]
        features = features.view(B, -1, C, H4, W4)              # (B, T, C, H4, W4)

        query_features = torch.zeros(B, N, C, device=device)    # (B, N, C)
        query_points_reshaped = queries.view(-1, 3)             # (B * N, 3)
        t, x, y = query_points_reshaped[:, 0].long(), query_points_reshaped[:, 1], query_points_reshaped[:, 2]       # (B * N)

        source_frame_f = features[torch.arange(B).repeat_interleave(N), t].view(-1, C, H4, W4)                       # (B * N, C, H4, W4)
        x_grid = (x / self.size[1]) * 2 - 1
        y_grid = (y / self.size[0]) * 2 - 1

        # assert (x_grid >= -1).all() and (x_grid <= 1).all(), f"x_grid: {x_grid}"
        # assert (y_grid >= -1).all() and (y_grid <= 1).all(), f"y_grid: {y_grid}"

        grid = torch.stack([x_grid, y_grid], dim=-1).view(B * N, 1, 1, 2).to(device)
        sampled = F.grid_sample(source_frame_f, grid, mode='bilinear', padding_mode='border', align_corners=False)
        query_features.view(-1, C)[:] = sampled.reshape(-1, C)      # (B * N, C)
        query_features = query_features.view(B, N, C)

        return query_features
    
    def forward(self, video, queries):
        # :args video: (B, T, C, H, W) in range [0, 255]
        # :args queries: (B, N, 3) where 3 is (t, y, x)
        #
        # :return tokens: (B, T, P, C)
        # :return q: (B, N, C)

        B, T, C, H, W = video.shape
        B, N, _ = queries.shape

        # === Normalize & Resize the video ===
        video_flat = video.view(B * T, C, H, W) / 255.0             # (B * T, C, H, W), to [0, 1]
        video_flat = F.interpolate(video_flat, size=self.size, mode="bilinear", align_corners=False)
        video_flat = self.normalization(video_flat)         # to [-1, 1]
        # === === ===

        f4, f8, _, _ = self.vit_encoder(video_flat)               # (B * T, 384, H4, W4)
        if self.stride == 4:
            f = f4
        elif self.stride == 8:
            f = f8

        C = f.shape[1]

        f = self.token_projection(f)                                          # (B * T, H4 * W4, C)
        f = f + self.frame_pos_embedding                                      # (B * T, C, H4, W4)

        q = self.get_query_tokens(f, queries)                                   # (B, N, C)

        f = f.permute(0, 2, 3, 1)                           # (B * T, H4, W4, C)
        tokens = f.view(B, T, self.P, self.embedding_dim)   # (B, T, P, C)

        assert tokens.shape == (B, T, self.P, self.embedding_dim), f"Tokens shape: {tokens.shape}, expected: {(B, T, self.P, self.embedding_dim)}"
        assert q.shape == (B, N, self.embedding_dim), f"Queries shape: {queries.shape}, expected: {(B, N, self.embedding_dim)}"

        return tokens, q
    
    # Online query sampling
    def sample_queries_online(self, video, queries):
        # :args video: (B, T, C, H, W) in range [0, 255]
        # :args queries: (B, N, 3) where 3 is (t, x, y)
        # :args query_features: (B, N, C)
        #
        # :return query_features: (B, N, C)
        

        B, T, C, H, W = video.shape
        B, N, _ = queries.shape
        device = video.device


        query_features = torch.zeros(B, N, self.embedding_dim, device=queries.device)    # (B, N, 3)
        for t in range(T):
            queries_of_this_time = queries[:, :, 0] == t                               # (B, N)
            query_positions = queries[queries_of_this_time].view(B, -1, 3)[:, :, 1:]   # (B, N', 2)
            N_prime = query_positions.shape[1]

            # No queries sampled at this time
            if N_prime == 0:
                continue

            x, y = query_positions[:, :, 0], query_positions[:, :, 1]   # (B, N')
            x_grid = (x / self.size[1]) * 2 - 1
            y_grid = (y / self.size[0]) * 2 - 1
            grid = torch.stack([x_grid, y_grid], dim=-1).view(B * N_prime, 1, 1, 2).to(device)  # (B * N', 1, 1, 2)


            video_frame = video[:, t] / 255.0          # (B, C, H, W), to [0, 1]
            video_frame = F.interpolate(video_frame, size=self.size, mode="bilinear", align_corners=False)
            video_frame = self.normalization(video_frame)         # to [-1, 1]


            f4, f8, _, _ = self.vit_encoder(video_frame)               # (B, 384, H4, W4)
            if self.stride == 4:
                f = f4
            elif self.stride == 8:
                f = f8

            H_prime, W_prime = f.shape[-2], f.shape[-1]
            assert H_prime == self.H_prime and W_prime == self.W_prime, f"Frame shape: {(H_prime, W_prime)}, expected: {(self.H_prime, self.W_prime)}"

            f = self.token_projection(f)                                                                    # (B, C, H4, W4)
            f = f + self.frame_pos_embedding                                                                # (B, C, H4, W4)
            C = f.shape[1]

            f = f.unsqueeze(1).expand(-1, N_prime, -1, -1, -1).reshape(B * N_prime, C, H_prime, W_prime)                # (B * N', C, H4, W4)

            sampled = F.grid_sample(f, grid, mode='bilinear', padding_mode='border', align_corners=False)   # (B * N', C, 1, 1)
            query_features[queries_of_this_time] = sampled.view(B, N_prime, C)                              # (B, N', C)


        return query_features
    
    def encode_frames_online(self, frames):
        # :args frame: (B, C, H, W) in range [0, 255]
        #
        # :return f: (B, P, C)

        frames = frames.clone() / 255.0                     # (B, C, H, W), to [0, 1]
        frames = F.interpolate(frames, size=self.size, mode="bilinear", align_corners=False)
        frames = self.normalization(frames)                 # to [-1, 1]

        f4, f8, _, _ = self.vit_encoder(frames)             # (B, 384, H4, W4)
        if self.stride == 4:
            f = f4
        elif self.stride == 8:
            f = f8


        f = self.token_projection(f)                                          # (B, C, H4, W4)
        f = f + self.frame_pos_embedding                                      # (B, C, H4, W4)
        C = f.shape[1]

        f = f.permute(0, 2, 3, 1)                           # (B, H4, W4, C)
        f = f.reshape(f.shape[0], -1, C)                       # (B, P, C)

        return f
    
