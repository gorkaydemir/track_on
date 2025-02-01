import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.ops

from mmcv.ops import MultiScaleDeformableAttention



# Attention Block
class MHA_Block(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_drop=0.1, mlp_drop=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, attn_drop, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.dropout = nn.Dropout(mlp_drop)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(mlp_drop)
        self.dropout2 = nn.Dropout(mlp_drop)
    

    def forward(self, q, k=None, v=None, mask=None, full_attn_mask=None):
        # :args q: (B, N, C)
        # :args kv: (B, P, C)

        if k is None:
            k = q
            
        if v is None:
            v = k

        B, N, C = q.shape
        B, P, C = k.shape

        if mask is not None:
            assert mask.shape == (B, P), f"Mask shape: {mask.shape} expected: {(B, P)}"
            # assert full_attn_mask is None, "Cannot use both mask and full_attn_mask"

        if full_attn_mask is not None:
            assert full_attn_mask.shape == (B * self.num_heads, N, P), f"Full mask shape: {full_attn_mask.shape} expected: {(B * self.num_heads, N, P)}"
            # assert mask is None, "Cannot use both mask and full_attn_mask"

        q = q + self.dropout1(self.mha(q, k, v, key_padding_mask=mask, attn_mask=full_attn_mask, need_weights=False)[0])
        q = self.norm1(q)
        q = q + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(q)))))
        q = self.norm2(q)

        return q
    

# Deformable Multi Scale Attention Block
class DMSMHA_Block(nn.Module):
    def __init__(self, hidden_size, num_heads, num_levels, p_drop=0.1):
        super().__init__()


        self.mha = MultiScaleDeformableAttention(embed_dims=hidden_size,
                                                 num_heads=num_heads,
                                                 num_levels=num_levels,
                                                 num_points=4,
                                                 dropout=p_drop,
                                                 batch_first=True)


        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p_drop)
        self.dropout2 = nn.Dropout(p_drop)

    def forward(self, q, k, v, reference_points, spatial_shapes, start_levels):
        # :args q: (B, N, C)
        # :args kv: (B, P, C)
        # :args reference_points: (B, N, num_level, 2)
        # :args spatial_shapes: (num_level, 2)
        # :args start_levels: (num_level,)
        

        B, N, C = q.shape
        B, P, C = k.shape
        device = q.device

        q = q + self.dropout1(self.mha(q, k, v, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=start_levels))
        q = self.norm1(q)
        q = q + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(q)))))
        q = self.norm2(q)

        return q


class Token_Decoder(nn.Module):
    def __init__(self, args, use_norm=False):
        super().__init__()

        hidden_size = args.transformer_embedding_dim

        # linear - layer norm - relu - linear - layer norm - relu - linear
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        if use_norm:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.norm3 = nn.LayerNorm(hidden_size)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()
        
    def forward(self, x):
        # :args x: (B, P, C)
        #
        # :return x: (B, P, C)

        B, P, C = x.shape

        identity = x
        out = self.linear1(x)
        out = self.norm1(out)
        out = F.relu(out + identity)

        identity = out
        out = self.linear2(out)
        out = self.norm2(out)
        out = F.relu(out + identity)

        identity = out
        out = self.linear3(out)
        out = self.norm3(out)
        out = F.relu(out + identity)

        x = self.linear4(out)

        return x
    

def get_deformable_inputs(f_t, target_coordinates, H_prime, W_prime):
    # 4 level of features
    # :args f_t: (B, P, C)
    # :args target_coordinates: (B, N, 2), in [0, 1]
    # :args H_prime: int
    # :args W_prime: int

    B, P, C = f_t.shape
    num_level = 4

    # === Features ===
    f1 = f_t.view(B, H_prime, W_prime, C).permute(0, 3, 1, 2)  # (B, C, H, W)
    f2 =  F.avg_pool2d(f1, kernel_size=2, stride=2)                      # (B, C, H // 2, W // 2)
    f3 =  F.avg_pool2d(f1, kernel_size=4, stride=4)                      # (B, C, H // 4, W // 4)
    f4 =  F.avg_pool2d(f1, kernel_size=8, stride=8)                      # (B, C, H // 8, W // 8)

    f1 = f1.view(B, C, H_prime * W_prime).permute(0, 2, 1)     # (B, P, C)
    f2 = f2.view(B, C, (H_prime // 2) * (W_prime // 2)).permute(0, 2, 1)  # (B, P // 4, C)
    f3 = f3.view(B, C, (H_prime // 4) * (W_prime // 4)).permute(0, 2, 1)  # (B, P // 16, C)
    f4 = f4.view(B, C, (H_prime // 8) * (W_prime // 8)).permute(0, 2, 1)  # (B, P // 64, C)

    f_scales = [f1, f2, f3, f4]
    f_scales = torch.cat(f_scales, dim=1)   # (B, P + P // 4 + P // 16 + P // 64, C)    


    # === Reference Points ===
    reference_points = target_coordinates.unsqueeze(2).expand(-1, -1, num_level, -1) # (B, N, num_level, 2)

    # === Spatial Shapes ===
    spatial_shapes = [(H_prime, W_prime), 
                      (H_prime // 2, W_prime // 2), 
                      (H_prime // 4, W_prime // 4), 
                      (H_prime // 8, W_prime // 8)]
    
    spatial_shapes = torch.tensor(spatial_shapes, device=f_t.device) # (num_level, 2)

    # === Start Levels ===
    start_levels = torch.tensor([0, 
                                 P, 
                                 P + P // 4, 
                                 P + P // 4 + P // 16], device=f_t.device) # (num_level,)
    
    return f_scales, reference_points, spatial_shapes, start_levels


