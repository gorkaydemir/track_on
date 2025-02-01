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

        self.nhead = nhead
        self.size = args.input_size
        self.stride = args.stride

        self.H_prime = self.size[0] // self.stride
        self.W_prime = self.size[1] // self.stride
        self.P = self.H_prime * self.W_prime

        self.random_mask_ratio = 0.0

        self.num_layers_rerank = args.num_layers_rerank
        self.num_layers_rerank_fusion = args.num_layers_rerank_fusion

        self.embedding_dim = args.transformer_embedding_dim
    
        self.top_k_regions = args.top_k_regions

        self.num_level = 4

        self.sample_query_fusion = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.sample_projection = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.sample_uncertainty = nn.Linear(self.embedding_dim, 1)

        # === Top-K Point Decoder ===
        self.local_patch_encoder = []
        for _ in range(self.num_layers_rerank):
            decoder_layer = DMSMHA_Block(self.embedding_dim, nhead, self.num_level)
            self.local_patch_encoder.append(decoder_layer)
        self.local_patch_encoder = nn.ModuleList(self.local_patch_encoder)

        # === Fuse Query and Top-K Points ===
        self.local_decoder = []
        for _ in range(self.num_layers_rerank_fusion):
            decoder_layer = MHA_Block(self.embedding_dim, nhead)
            self.local_decoder.append(decoder_layer)
        self.local_decoder = nn.ModuleList(self.local_decoder)

    def nms_topk(self, c_t, upper=512):
        # :args c_t: (B, N, P)
        #
        # :return patch_indices: (B, N, top_k_regions)

        B, N, P = c_t.shape
        c_t_dist = F.softmax(c_t / 0.05, dim=-1)     # (B, N, P)

        top_k_indices = torch.topk(c_t_dist.view(B * N, P), upper)[1]                # (B * N, upper)
        scores = torch.gather(c_t_dist.view(B * N, P), dim=-1, index=top_k_indices)  # (B * N, upper)

        x_coords = top_k_indices % self.W_prime
        y_coords = top_k_indices // self.W_prime

        points = torch.stack((x_coords, y_coords), dim=-1)  # (B * N, upper, 2)

        half_box_size = 3
        x1 = points[:, :, 0] - half_box_size    # (B * N, upper)
        y1 = points[:, :, 1] - half_box_size    # (B * N, upper)
        x2 = points[:, :, 0] + half_box_size    # (B * N, upper)
        y2 = points[:, :, 1] + half_box_size    # (B * N, upper)

        boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(B, N, upper, 4)  # (B, N, upper, 4)
        scores = scores.view(B, N, upper)                                   # (B, N, upper)

        top_k_indices_batch = top_k_indices.view(B, N, upper)       # (B, N, upper)
        patch_indices = torch.zeros(B, N, self.top_k_regions, device=c_t.device, dtype=torch.long)
        for b in range(B):
            boxes_flat = boxes[b].view(N * upper, 4).float()               # (N * upper, 4)
            scores_flat = scores[b].view(N * upper)                # (N * upper) 

            indices = torch.arange(N).repeat_interleave(upper).to(c_t.device)   # (N * upper), [[0]*upper, [1]*upper, ..., [N-1]*upper]
            kept_indices_flat = torchvision.ops.batched_nms(boxes_flat, scores_flat, indices, iou_threshold=0.5)        # kept_indices_flat: (kept)

            n_indices = kept_indices_flat // upper                # (kept)
            corresponding_patch_indices = kept_indices_flat % upper             # (kept)

            for n in range(N):
                patch_indices[b, n] = top_k_indices_batch[b, n][corresponding_patch_indices[n_indices == n][:self.top_k_regions]]

        return patch_indices
    
    def get_deformable_inputs(self, f_t, target_coordinates):
        # :args f_t: (B, P, C)
        # :args target_coordinates: (B, N, K, 2), in [0, 1]

        B, P, C = f_t.shape


        f1 = f_t.view(B, self.H_prime, self.W_prime, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        f2 =  F.avg_pool2d(f1, kernel_size=2, stride=2)                      # (B, C, H // 2, W // 2)
        f3 =  F.avg_pool2d(f1, kernel_size=4, stride=4)                      # (B, C, H // 4, W // 4)

        f1 = f1.view(B, C, self.H_prime * self.W_prime).permute(0, 2, 1)     # (B, P, C)
        f2 = f2.view(B, C, (self.H_prime // 2) * (self.W_prime // 2)).permute(0, 2, 1)  # (B, P // 4, C)
        f3 = f3.view(B, C, (self.H_prime // 4) * (self.W_prime // 4)).permute(0, 2, 1)  # (B, P // 16, C)
        f_scales = [f1, f2, f3]

        if self.num_level == 4:
            f4 =  F.avg_pool2d(f1, kernel_size=8, stride=8)                  # (B, C, H // 8, W // 8)
            f4 = f4.view(B, C, (self.H_prime // 8) * (self.W_prime // 8)).permute(0, 2, 1)
            f_scales.append(f4)

        # Features
        f_scales = torch.cat(f_scales, dim=1)   # (B, P + P // 4 + P // 16, C) or (B, P + P // 4 + P // 16 + P // 64, C)

        # Reference points
        reference_points = target_coordinates.unsqueeze(3).expand(-1, -1, -1, self.num_level, -1) # (B, N, K, num_level, 2)
        reference_points = reference_points.view(B, -1, self.num_level, 2)                        # (B, N * K, num_level, 2)

        # Spatial shapes
        spatial_shapes = [[self.H_prime, self.W_prime],
                            [self.H_prime // 2, self.W_prime // 2],
                            [self.H_prime // 4, self.W_prime // 4]]

        if self.num_level == 4:
            spatial_shapes.append([self.H_prime // 8, self.W_prime // 8])
            
        spatial_shapes = torch.tensor(spatial_shapes, device=f_t.device) # (num_level, 2)

        # Start levels
        start_levels = [0, self.P, self.P + self.P // 4]
        if self.num_level == 4:
            start_levels.append(self.P + self.P // 4 + self.P // 16)
        start_levels = torch.tensor(start_levels, device=f_t.device)     # (num_level)

        return f_scales, reference_points, spatial_shapes, start_levels
    
    def forward(self, q_t, h_t, c_t):
        # :args q_t: (B, N, C)
        # :args h_t: (B, P, C)
        # :args c_t: (B, N, P)
        #
        # :return q_t: (B, N, C)
        # :return top_patches_uncertainty: (B, N, top_k_regions)
        # :return target_patch_locations: (B, N, top_k_regions, 2)

        B, N, P = c_t.shape
        C = h_t.shape[2]
        device = c_t.device


        # === Get top k indices ===
        # top_k_indices = self.nms_topk(c_t)                                                               # (B, N, top_k_regions)
        # or
        top_k_indices = torch.topk(c_t, self.top_k_regions, dim=-1)[1]                                  # (B, N, top_k_regions)

        target_patch_locations = torch.zeros(B, N, self.top_k_regions, 2, device=device)                 # (B, N, top_k_regions, 2)
        for k in range(self.top_k_regions):
            ref_index = top_k_indices[:, :, k]                                                           # (B, N)
            ref_point = indices_to_coords(ref_index.unsqueeze(1), self.size, self.stride)[:, 0]          # (B, N, 2)
            target_patch_locations[:, :, k] = ref_point

        target_patch_locations_normalized = target_patch_locations / torch.tensor([self.size[1], self.size[0]], device=device)  # (B, N, K, 2)
        target_patch_locations_normalized = torch.clamp(target_patch_locations_normalized, 0, 1)

        f_scales, reference_points, spatial_shapes, start_levels = self.get_deformable_inputs(h_t, target_patch_locations_normalized)

        q_t_reshaped = q_t.unsqueeze(2).expand(-1, -1, self.top_k_regions, -1)      # (B, N, K, C)
        q_t_reshaped = q_t_reshaped.reshape(B, N * self.top_k_regions, self.embedding_dim)   # (B, N * K, C)
        q_local = q_t_reshaped

        # === Encode Top-K Points ===
        for i in range(self.num_layers_rerank):
            q_local = self.local_patch_encoder[i](q=q_local, k=f_scales, v=f_scales,
                                                reference_points=reference_points, 
                                                spatial_shapes=spatial_shapes, 
                                                start_levels=start_levels)              # (B, N * K, C)
        top_patches = q_local.reshape(B, N, self.top_k_regions, self.embedding_dim)     # (B, N, K, C)


        
        # cat q_t and top_patches
        q_t_expanded = q_t.unsqueeze(2).expand(-1, -1, self.top_k_regions, -1)                                 # (B, N, top_k_regions, C)
        q_t_top_patches = torch.cat([q_t_expanded, top_patches], dim=-1)                                       # (B, N, top_k_regions, 2 * C)
        q_t_top_patches = self.sample_projection(q_t_top_patches)                                              # (B, N, top_k_regions, C)

        top_patches_uncertainty = self.sample_uncertainty(q_t_top_patches).squeeze(-1)                         # (B, N, top_k_regions)

        # decode
        q_t_top_patches = q_t_top_patches.view(B * N, self.top_k_regions, C)                                   # (B * N, top_k_regions, C)
        q_t_query = q_t.view(B * N, 1, C)                                                                      # (B * N, top_k_regions, C)

        random_mask = torch.rand(B * N, self.top_k_regions, device=device) < self.random_mask_ratio            # (B * N, top_k_regions)
        for i in range(self.num_layers_rerank_fusion):
            q_t_query = self.local_decoder[i](q=q_t_query, k=q_t_top_patches, v=q_t_top_patches, mask=random_mask)    # (B * N, top_k_regions, C)

        # q_t_query to B, N, C
        q_t_query = q_t_query.view(B, N, C)

        q_t = self.sample_query_fusion(torch.cat([q_t, q_t_query], dim=-1))                                     # (B, N, C)

        return q_t, top_patches_uncertainty, target_patch_locations
  