import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import gc

from model.modules import DMSMHA_Block
from verifier.backbone import BasicEncoder, disp_sincos_embedding
from utils.coord_utils import sample_grid_points

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Transformer_Decoder_Layer(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_drop=0.1, mlp_drop=0.1):
        super().__init__()
        
        self.mha1 = nn.MultiheadAttention(hidden_size, num_heads, attn_drop, batch_first=True)  # spatial attention
        self.mha2 = nn.MultiheadAttention(hidden_size, num_heads, attn_drop, batch_first=True)  # temporal attention

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.dropout1 = nn.Dropout(attn_drop)
        self.dropout2 = nn.Dropout(attn_drop)
        self.dropout3 = nn.Dropout(attn_drop)

        # FFN
        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(mlp_drop)

        self.act = nn.GELU(approximate='tanh')

    def forward(self, q, f, t_embedding, t_mask=None):
        # q:            (B, T, N, 1, D)  # q is initialized as the last candidate feature and then updated with attention outputs
        # f:            (B, T, N, M, D)  # q is initialized as the last candidate feature and then updated with attention outputs
        # t_embedding:  (1, T, D)
        # t_mask:       (B, N, T)

        B, T, N, M, D = f.shape

        # First Multi-Head Attention (Spatial Self-Attention within each frame)
        # Factorized attention: First spatial attention across ensemble members
        # qkv = f: (B * T * N, M, D)
        
        f_spatial = f.reshape(B * T * N, M, D)  # (B * T * N, M, D)
        q = q.reshape(B * T * N, 1, D)          # (B * T * N, 1, D) - query is the last candidate feature for each point, expanded to match attention input shape
        
        
        # Spatial self-attention across ensemble members
        attn_out1, _ = self.mha1(q, f_spatial, f_spatial)       # (B * T * N, 1, D)
        q = q + self.dropout1(attn_out1)  # Residual connection
        q = self.norm1(q)  # Layer normalization
        
        # Reshape back to (B, T, N, D)
        q = q.reshape(B, T, N, D).permute(0, 2, 1, 3)  # (B, N, T, D)
        q = q.reshape(B * N, T, D)  # (B * N, T, D)
        
        # Create temporal attention mask if provided
        key_padding_mask = None
        if t_mask is not None:
            # t_mask: (B, N, T) -> (B * N, T)
            key_padding_mask = t_mask.reshape(B * N, T)  # (B * N, T)

        # Create windowed attention mask for temporal attention
        attn_out2, _ = self.mha2(q + t_embedding, 
                                q + t_embedding, 
                                q, key_padding_mask=key_padding_mask)  # (B * N, T, D)

        q = q + self.dropout2(attn_out2)  # Residual connection
        q = self.norm2(q)  # Layer normalization
        
            # Reshape back to (B, T, N, M, D)
        q = q.reshape(B, N, T, D).permute(0, 2, 1, 3)   # (B, T, N, D)

            
        # Feed-Forward Network with residual connection
        ffn_input = q
        ffn_out = self.linear1(q)
        ffn_out = self.act(ffn_out)
        ffn_out = self.dropout(ffn_out)
        ffn_out = self.linear2(ffn_out)
        
        q = ffn_input + self.dropout3(ffn_out)  # Residual connection
        q = self.norm3(q)                       # Layer normalization

        q = q.unsqueeze(3)                      # (B, T, N, 1, D)

        return q  # (B, T, N, 1, D)



class Verifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.T = args.T

        self.num_layers = args.verifier_num_layers
        self.verifier_embedding_dim = args.verifier_embedding_dim
        self.pos_embed_dim = 32
        nhead = 4
        self.size = args.verifier_input_size

        self.query_critic = args.query_critic  # "pos_embedding", "feature", "none"

        # === CNN backbone ===
        self.cnn_encoder = BasicEncoder(input_dim=3, output_dim=128, stride=4)
        print(f"Loading pretrained backbone from CoTracker3...")
        cotracker3 = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self.cnn_encoder.load_state_dict(cotracker3.model.fnet.state_dict(), strict=False)
        backbone_projection_input = 128
        # === === ===

        self.backbone_projection = nn.Conv2d(backbone_projection_input, 
                                             self.verifier_embedding_dim, 
                                             kernel_size=1, stride=1, padding=0)

        self.stride = 4

        self.P = (self.size[0] // self.stride) * (self.size[1] // self.stride)
        self.H_prime = self.size[0] // self.stride
        self.W_prime = self.size[1] // self.stride
        # === === ===

        # - Pos Embeddings -
        self.time_emb = nn.Parameter(torch.zeros(1, self.T, self.verifier_embedding_dim))
        nn.init.trunc_normal_(self.time_emb, std=0.02)
        # === === ===

        # === === Local Encoder === ===

        # - Deformable Transformer -
        self.local_encoder = []
        self.local_encoder_num_layers = args.local_encoder_num_layers

        for _ in range(self.local_encoder_num_layers):
            layer = DMSMHA_Block(hidden_size=self.verifier_embedding_dim, num_heads=4, num_levels=4, p_drop=0.1)
            self.local_encoder.append(layer)
        self.local_encoder = nn.ModuleList(self.local_encoder)

        # Deformable inputs into buffers
        spatial_shapes = torch.tensor([(self.H_prime, self.W_prime), 
                                            (self.H_prime // 2, self.W_prime // 2), 
                                            (self.H_prime // 4, self.W_prime // 4), 
                                            (self.H_prime // 8, self.W_prime // 8)]) # (4, 2)
        self.register_buffer("spatial_shapes", spatial_shapes, persistent=False)

        start_levels = torch.tensor([0, 
                                    self.P, 
                                    self.P + self.P // 4,  
                                    self.P + self.P // 4 + self.P // 16]) # (4)
        self.register_buffer("start_levels", start_levels, persistent=False)


        if self.query_critic == "feature":
            self.candidate_emb = nn.Embedding(2, self.pos_embed_dim)

        elif self.query_critic == "none":
            self.candidate_emb = None
        
        self.displacement_scale = nn.Parameter(torch.tensor(16.0))           # Learnable scaling for displacement

        # - Local Projection -
        # Adjust dimension based on whether displacement is excluded
        displacement_dim = (self.pos_embed_dim + 2)
        # Add query critic feature dimension if using "feature" mode
        query_critic_dim = self.pos_embed_dim if self.query_critic == "feature" else 0
        self.local_descriptor_dim = self.verifier_embedding_dim + displacement_dim + query_critic_dim

        self.local_projection = nn.Sequential(nn.LayerNorm(self.local_descriptor_dim),
                                              nn.Linear(self.local_descriptor_dim, self.verifier_embedding_dim))
        # === === ===
        
        # === Transformer Encoder ===
        self.layers = []
        for _ in range(self.num_layers):
            layer = Transformer_Decoder_Layer(self.verifier_embedding_dim, nhead)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        # === === ===

        # === Final Projections ===
        
        self.sim_scale_raw = nn.Parameter(torch.tensor(10.0))   # Learnable scaling for similarities
        self.q_projection = nn.Linear(self.verifier_embedding_dim, self.verifier_embedding_dim, bias=False)
        self.f_projection = nn.Linear(self.verifier_embedding_dim, self.verifier_embedding_dim, bias=False)


        # === === ===

    def sim_scale(self):
        return (F.softplus(self.sim_scale_raw) + 1.0).clamp(1.0, 20.0)

    def interpolate_time_embedding(self, target_T):
        """
        Interpolate self.time_embedding (of shape (1, original_T, 1, D))
        to a new shape (1, target_T, D) using linear interpolation.
        """
        original_T = self.time_emb.shape[1]
        if target_T == original_T:
            return self.time_emb

        # Reshape to (1, D, original_T) for 1D interpolation
        time_emb = self.time_emb.permute(0, 2, 1)  # (1, D, original_T)
        
        # Interpolate along the last dimension (the time axis)
        time_emb = F.interpolate(
            time_emb, size=target_T, mode='linear', align_corners=False
        )  # (1, D, target_T)
        
        # Permute back and add the singleton dimension for broadcasting: (1, target_T, D)
        time_emb = time_emb.permute(0, 2, 1)
        return time_emb
    

    def extract_local_features(self, q_init, f_multiscale, grid):
        """
        Extract local features using deformable attention or simple sampling.
        
        Args:
            q_init: (B, N, M, D) - Initial query features
            f_multiscale: (B, P_total, D) - Multi-scale feature maps concatenated
            grid: (B, N, M, 2) - Sampling coordinates in [-1, 1] range
            
        Returns:
            h_t: (B, N, M, D) - Enhanced local features
        """
        B, N, M, D = q_init.shape
        
        # Original deformable attention approach
        # Flatten grid for deformable attention
        grid_flat = grid.view(B, N * M, 2)  # (B, N*M, 2)
        
        # Convert reference points from [-1, 1] to [0, 1] for deformable attention
        reference_points = ((grid_flat.clone() + 1) / 2)  # (B, N*M, 2)
        reference_points = torch.clamp(reference_points, 0, 1)
        
        # Expand reference points for multi-level attention (4 levels)
        reference_points = reference_points.unsqueeze(2).expand(-1, -1, 4, -1)  # (B, N*M, 4, 2)
        
        # Flatten q_init for processing
        h_t = q_init.reshape(B, N * M, D)  # (B, N*M, D)
        
        # Apply deformable attention layers
        for l in range(self.local_encoder_num_layers):
            h_t = self.local_encoder[l](
                q=h_t, 
                k=f_multiscale, 
                v=f_multiscale,
                reference_points=reference_points, 
                spatial_shapes=self.spatial_shapes, 
                start_levels=self.start_levels
            )  # (B, N*M, D)
        
        # Reshape back to (B, N, M, D)
        h_t = h_t.view(B, N, M, D)
        
        return h_t

    def forward(self, videos, tracks, queries):
        # :args videos: (B, T, 3, H, W)
        # :args tracks: (B, T, N, M, 2)
        # :args visibilities: (B, T, N, M)
        # :args queries: (B, N, 3)
        #
        # :return rank_logits: (B, T, N, M)
        # :return predicted_offsets: (B, T, N, M, 2)

        self.cnn_encoder.eval()


        B, T, _, H, W = videos.shape
        M = tracks.shape[3]
        N = tracks.shape[2]

        # === Resize Tracks ===
        # To [-1, 1]
        tracks[..., 1] = (tracks[..., 1] / H) * 2 - 1
        tracks[..., 0] = (tracks[..., 0] / W) * 2 - 1

        # To [-1, 1]
        queries[..., 2] = (queries[..., 2] / H) * 2 - 1
        queries[..., 1] = (queries[..., 1] / W) * 2 - 1
        # === === ===

        # === Extract features (chunked for long videos) ===
        # Normalize frames and prepare for per-chunk processing
        device = videos.device

        # === Extract features ===
        # Normalize and resize video
        videos = (videos / 255.0) * 2 - 1
        videos = F.interpolate(videos.reshape(B * T, 3, H, W), size=self.size, mode="bilinear", align_corners=False)

        f_vis = self.backbone_projection(self.cnn_encoder(videos))          # (B * T, D, H4, W4)
        f2_vis = F.avg_pool2d(f_vis, kernel_size=2, stride=2)               # (B * T, D, H4/2, W4/2)
        f3_vis = F.avg_pool2d(f_vis, kernel_size=4, stride=4)               # (B * T, D, H4/4, W4/4)
        f4_vis = F.avg_pool2d(f_vis, kernel_size=8, stride=8)               # (B * T, D, H4/8, W4/8)

        f_vis = f_vis.view(B, T, self.verifier_embedding_dim, self.H_prime, self.W_prime)
        f_multiscale = torch.cat([f_vis.reshape(B, T, self.verifier_embedding_dim, self.H_prime * self.W_prime).permute(0, 1, 3, 2),
                                    f2_vis.reshape(B, T, self.verifier_embedding_dim, (self.H_prime // 2) * (self.W_prime // 2)).permute(0, 1, 3, 2),
                                    f3_vis.reshape(B, T, self.verifier_embedding_dim, (self.H_prime // 4) * (self.W_prime // 4)).permute(0, 1, 3, 2),
                                    f4_vis.reshape(B, T, self.verifier_embedding_dim, (self.H_prime // 8) * (self.W_prime // 8)).permute(0, 1, 3, 2)], dim=2)  # (B, T, P + P/4 + P/16 + P/64, D)


        local_feature_dim = self.local_descriptor_dim

        q_init = torch.zeros(B, N, self.verifier_embedding_dim, device=videos.device)
        q = torch.zeros(B, N, local_feature_dim, device=videos.device)
        f = torch.zeros(B, T, N, M, local_feature_dim, device=videos.device)

 
        # === Sample q_init ===
        for t in range(T):
            mask = queries[:, :, 0].long() == t        # (B, N)
            if mask.sum() > 0:
                batch_indices, point_indices = torch.nonzero(mask, as_tuple=True)       # Get valid indices
                assert B == 1
                grid = queries[batch_indices, point_indices, 1:].unsqueeze(0)           # (1, N_t, 2)

                f_t = f_vis[:, t]

                # Sample initial features from feature map
                sampled = sample_grid_points(f_t, grid, L=0)                            # (1, N_t, D, 1, 1)
                sampled = sampled.squeeze(-1).squeeze(-1)                               # (1, N_t, D)

                q_init[batch_indices, point_indices] = sampled                          # (B, N_t, D)
                
                # === Extract local features for query points ===
                # Get multiscale features for this timestep
                f_t_multiscale = f_multiscale[:, t]  # (B, P_total, D)

                # Prepare sampled features for deformable attention
                sampled_expanded = sampled.unsqueeze(2)  # (B, N_t, 1, D)
                grid_expanded = grid.unsqueeze(2)  # (B, N_t, 1, 2)
                
    
                # Extract local features using deformable attention
                h_t_query = self.extract_local_features(
                    q_init=sampled_expanded,      # (B, N_t, 1, D)
                    f_multiscale=f_t_multiscale,  # (B, P_total, D)
                    grid=grid_expanded            # (B, N_t, 1, 2)
                )  # (B, N_t, 1, D)
                    
                h_t_query = h_t_query.squeeze(2)  # (B, N_t, D)

                # Displacement Embedding Sampling - all 0 xy_disp
                xy_disp = torch.zeros(B, point_indices.shape[0], 2, device=videos.device)  # (B, N_t, 2)
                g = disp_sincos_embedding(xy_disp * self.displacement_scale, 
                                            num_bands=self.pos_embed_dim // 4, max_freq=32.0, include_xy=True)  # (B, N_t, D_pos + 2)
                # Concatenate all features
                h_t_query = torch.cat([h_t_query, g], dim=-1)         # (B, N_t, D + D_pos + 2)
                
                # Add query critic embedding if using "feature" mode
                if self.query_critic == "feature":
                    query_emb = self.candidate_emb(torch.zeros(point_indices.shape[0], dtype=torch.long, device=videos.device))  # (N_t, D_pos)
                    query_emb = query_emb.unsqueeze(0).expand(B, -1, -1)  # (B, N_t, D_pos)
                    h_t_query = torch.cat([h_t_query, query_emb], dim=-1)  # (B, N_t, D + D_pos + 2 + D_pos) or (B, N_t, D + D_pos)
                
                q[batch_indices, point_indices] = h_t_query           
                # === ===

        q_init = q_init.unsqueeze(2).expand(-1, -1, M, -1)           # (B, N, M, D)
        # === === ===

        # === Sample candidate features ===
        for t in range(T):
            grid = tracks[:, t]  # (B, N, M, 2)
            
            # === Multiscale Feature Maps for Deformable Attention ===
            f_t_multiscale = f_multiscale[:, t]  # (B, P_total, D)

            # Extract local features using deformable attention
            h_t = self.extract_local_features(
                q_init=q_init,                # (B, N, M, D)
                f_multiscale=f_t_multiscale,  # (B, P_total, D)
                grid=grid.clone()             # (B, N, M, 2)
            )  # (B, N, M, D)

            # === Displacement Embedding Sampling ===
            grid_reshaped = grid.view(B, N, M, 2)
            xy_disp = grid_reshaped - queries[:, :, 1:].unsqueeze(2)  # (B, N, M, 2)
            g = disp_sincos_embedding(xy_disp * self.displacement_scale, 
                                        num_bands=self.pos_embed_dim // 4, max_freq=32.0, include_xy=True)  # (B, N, M, D_pos + 2)

            h_t = torch.cat([h_t, g], dim=-1)  # (B, N, M, D + D_pos + 2)
        
            # Add candidate critic embedding if using "feature" mode
            if self.query_critic == "feature":
                candidate_emb = self.candidate_emb(torch.ones(B * N * M, dtype=torch.long, device=videos.device))  # (B * N * M, D_pos)
                candidate_emb = candidate_emb.reshape(B, N, M, self.pos_embed_dim)  # (B, N, M, D_pos)
                h_t = torch.cat([h_t, candidate_emb], dim=-1)  # (B, N, M, D + D_pos + 2 + D_pos) or (B, N, M, D + D_pos)
            
            # Concatenate all features
            f[:, t] = h_t  # (B, N, M, D + D_pos + 2) or (B, N, M, D)
        # === === ===

        # === Project to common dimension ===
        f = self.local_projection(f)    # (B, T, N, M, D + D_pos + 2) or (B, T, N, M, D) -> (B, T, N, M, D)
        q = self.local_projection(q)    # (B, N, D + D_pos + 2) or (B, N, D) -> (B, N, D)
        q = q.unsqueeze(1).unsqueeze(3).expand(-1, T, -1, -1, -1) #  (B, N, D) -> (B, 1, N, 1, D) -> (B, T, N, 1, D)

        # === === ===

        # === Temporal Mask ===
        temporal_mask = torch.arange(T, device=videos.device).reshape(1, 1, T).expand(B, N, T)  
        temporal_mask = (temporal_mask < queries[:, :, 0].unsqueeze(2)).bool()      # (B, N, T)
        # === ===

        time_emb = self.interpolate_time_embedding(T)  # (1, T, D)

        # === ===

        for i in range(self.num_layers):
            q = self.layers[i](q, f, time_emb, t_mask=temporal_mask)    # (B, T, N, 1, D)

        q = q.expand(-1, -1, -1, M, -1)                 # (B, T, N, 1, D) -> (B, T, N, M, D)

        q_normalized = F.normalize(self.q_projection(q), p=2, dim=-1)                       # (B, T, N, M, D)
        f_normalized = F.normalize(self.f_projection(f), p=2, dim=-1)                       # (B, T, N, M, D)

        similarities = torch.einsum("btnmd,btnmd->btnm", q_normalized, f_normalized)        # (B, T, N, M)
        rank_logits = similarities * self.sim_scale()                                       # (B, T, N, M)

        return rank_logits

    def efficient_forward(self, videos, tracks, queries, max_frames_per_pass: int = 500):
        """
        Memory-efficient forward:
          - First extract local descriptors for query frames only (h_t_query).
          - Then process all frames in chunks: extract per-frame candidate descriptors and immediately aggregate them into f (no global CNN feature storage).
        Arguments and returns same shapes as forward().
        """
        self.cnn_encoder.eval()

        B, T, _, H, W = videos.shape
        M = tracks.shape[3]
        N = tracks.shape[2]
        device = videos.device

        if max_frames_per_pass <= 0:
            raise ValueError("max_frames_per_pass must be > 0")

        # currently extraction logic assumes B == 1 as in original forward sampling code
        assert B == 1, "efficient_forward currently assumes batch size 1 (same as original sampling code)."

        # --- Resize coords to [-1,1] (in-place clones to avoid changing caller tensors) ---
        tracks = tracks.clone()
        tracks[..., 1] = (tracks[..., 1] / H) * 2 - 1
        tracks[..., 0] = (tracks[..., 0] / W) * 2 - 1

        queries = queries.clone()
        queries[..., 2] = (queries[..., 2] / H) * 2 - 1
        queries[..., 1] = (queries[..., 1] / W) * 2 - 1
        # === === ===

        # Local descriptor shapes / containers (match forward)
        local_feature_dim = self.local_descriptor_dim
        q_init = torch.zeros(B, N, self.verifier_embedding_dim, device=device)      # (B,N,D)
        q_local = torch.zeros(B, N, local_feature_dim, device=device)               # (B,N,local_dim)
        f_local = torch.zeros(B, T, N, M, local_feature_dim, device=device)        # will fill per-frame

        # Pre-normalize and reshape frames for indexed access (B == 1)
        videos_norm = (videos / 255.0) * 2 - 1
        videos_flat = videos_norm.reshape(B * T, 3, H, W)  # (T,3,H,W) when B==1

        # --------------------------
        # 1) Extract query local descriptors (only for frames that have queries)
        # --------------------------
        # unique frame indices that have queries (B==1)
        query_frame_idxs = torch.unique(queries[0, :, 0].long())
        query_frame_idxs = query_frame_idxs[ (query_frame_idxs >= 0) & (query_frame_idxs < T) ]

        # process unique query frames in small groups for GPU memory control
        q_frame_list = query_frame_idxs.tolist()
        for i in range(0, len(q_frame_list), max_frames_per_pass):
            chunk_idxs = q_frame_list[i:i + max_frames_per_pass]
            if len(chunk_idxs) == 0:
                continue
            # gather frames for this chunk
            chunk_frames = videos_flat[chunk_idxs].to(device)               # (k,3,H,W)
            chunk_resized = F.interpolate(chunk_frames, size=self.size, mode="bilinear", align_corners=False)
            f_vis_chunk = self.backbone_projection(self.cnn_encoder(chunk_resized))  # (k,D,H',W')
            f2_chunk = F.avg_pool2d(f_vis_chunk, kernel_size=2, stride=2)
            f3_chunk = F.avg_pool2d(f_vis_chunk, kernel_size=4, stride=4)
            f4_chunk = F.avg_pool2d(f_vis_chunk, kernel_size=8, stride=8)

            # process each frame in chunk (there typically won't be many query frames)
            for local_idx, t in enumerate(chunk_idxs):
                # which query points correspond to this frame
                mask = (queries[0, :, 0].long() == int(t))
                if mask.sum() == 0:
                    continue
                point_indices = torch.nonzero(mask, as_tuple=True)[0]

                # build per-frame multiscale descriptor input for deformable attention
                f_vis_frame = f_vis_chunk[local_idx].unsqueeze(0)   # (1,D,H',W')
                f2_frame = f2_chunk[local_idx].unsqueeze(0)
                f3_frame = f3_chunk[local_idx].unsqueeze(0)
                f4_frame = f4_chunk[local_idx].unsqueeze(0)

                # convert to (1, P_total, D)
                p1 = f_vis_frame.reshape(1, self.verifier_embedding_dim, self.H_prime * self.W_prime).permute(0, 2, 1)
                p2 = f2_frame.reshape(1, self.verifier_embedding_dim, (self.H_prime // 2) * (self.W_prime // 2)).permute(0, 2, 1)
                p3 = f3_frame.reshape(1, self.verifier_embedding_dim, (self.H_prime // 4) * (self.W_prime // 4)).permute(0, 2, 1)
                p4 = f4_frame.reshape(1, self.verifier_embedding_dim, (self.H_prime // 8) * (self.W_prime // 8)).permute(0, 2, 1)
                f_multiscale_frame = torch.cat([p1, p2, p3, p4], dim=1)  # (1, P_total, D)

                # sample initial features (q_init) for the query points
                grid = queries[0, point_indices, 1:].unsqueeze(0)  # (1, N_t, 2)
                sampled = sample_grid_points(f_vis_frame, grid, L=0)  # (1, N_t, D,1,1)
                sampled = sampled.squeeze(-1).squeeze(-1)             # (1, N_t, D)

                # write into q_init
                q_init[0, point_indices] = sampled[0]

                # extract local features for query points using deformable attention
                sampled_expanded = sampled.unsqueeze(2)               # (1, N_t, 1, D)
                grid_expanded = grid.unsqueeze(2)                     # (1, N_t, 1, 2)
                h_t_query = self.extract_local_features(q_init=sampled_expanded,
                                                       f_multiscale=f_multiscale_frame,
                                                       grid=grid_expanded)          # (1, N_t, 1, D)
                h_t_query = h_t_query.squeeze(2)                     # (1, N_t, D)

                # displacement embedding (all zero at query sampling)
                xy_disp = torch.zeros(1, point_indices.shape[0], 2, device=device)
                g = disp_sincos_embedding(xy_disp * self.displacement_scale,
                                            num_bands=self.pos_embed_dim // 4,
                                            max_freq=32.0, include_xy=True)  # (1, N_t, D_pos+2)
                h_t_query = torch.cat([h_t_query, g], dim=-1)

                # query_critic feature mode
                if self.query_critic == "feature":
                    query_emb = self.candidate_emb(torch.zeros(point_indices.shape[0], dtype=torch.long, device=videos.device))  # (N_t, D_pos)
                    query_emb = query_emb.unsqueeze(0).expand(1, -1, -1)  # (1, N_t, D_pos)
                    h_t_query = torch.cat([h_t_query, query_emb], dim=-1)

                # write into q_local
                q_local[0, point_indices] = h_t_query[0]

                # free per-frame temporaries used in this inner loop
                del f_vis_frame, f2_frame, f3_frame, f4_frame
                del p1, p2, p3, p4, f_multiscale_frame, grid, sampled, sampled_expanded, grid_expanded, h_t_query
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # free chunk-level temporaries immediately after processing the chunk
            del chunk_frames, chunk_resized, f_vis_chunk, f2_chunk, f3_chunk, f4_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # expand q_init to match M and continue with candidate extraction
        q_init_exp = q_init.unsqueeze(2).expand(-1, -1, M, -1)  # (B,N,M,D)

        # --------------------------
        # 2) Extract candidate local descriptors for all frames in chunks
        # --------------------------
        for start in range(0, T, max_frames_per_pass):
            end = min(T, start + max_frames_per_pass)
            frames_idx = list(range(start, end))
            chunk_frames = videos_flat[frames_idx].to(device)             # (k,3,H,W)
            chunk_resized = F.interpolate(chunk_frames, size=self.size, mode="bilinear", align_corners=False)
            f_vis_chunk = self.backbone_projection(self.cnn_encoder(chunk_resized))  # (k,D,H',W')
            f2_chunk = F.avg_pool2d(f_vis_chunk, kernel_size=2, stride=2)
            f3_chunk = F.avg_pool2d(f_vis_chunk, kernel_size=4, stride=4)
            f4_chunk = F.avg_pool2d(f_vis_chunk, kernel_size=8, stride=8)

            # process each frame in this chunk and write candidate descriptors directly to f_local
            for local_idx, t in enumerate(frames_idx):
                # build per-frame multiscale descriptor input for deformable attention
                f_vis_frame = f_vis_chunk[local_idx].unsqueeze(0)   # (1,D,H',W')
                f2_frame = f2_chunk[local_idx].unsqueeze(0)
                f3_frame = f3_chunk[local_idx].unsqueeze(0)
                f4_frame = f4_chunk[local_idx].unsqueeze(0)

                p1 = f_vis_frame.reshape(1, self.verifier_embedding_dim, self.H_prime * self.W_prime).permute(0, 2, 1)
                p2 = f2_frame.reshape(1, self.verifier_embedding_dim, (self.H_prime // 2) * (self.W_prime // 2)).permute(0, 2, 1)
                p3 = f3_frame.reshape(1, self.verifier_embedding_dim, (self.H_prime // 4) * (self.W_prime // 4)).permute(0, 2, 1)
                p4 = f4_frame.reshape(1, self.verifier_embedding_dim, (self.H_prime // 8) * (self.W_prime // 8)).permute(0, 2, 1)
                f_multiscale_frame = torch.cat([p1, p2, p3, p4], dim=1)  # (1, P_total, D)

                # tracks grid for this frame
                grid = tracks[:, t]  # (B, N, M, 2)

                # extract local features using deformable attention
                h_t = self.extract_local_features(q_init=q_init_exp, f_multiscale=f_multiscale_frame, grid=grid.clone())  # (B,N,M,D)

                grid_reshaped = grid.view(B, N, M, 2)
                xy_disp = grid_reshaped - queries[:, :, 1:].unsqueeze(2)
                g = disp_sincos_embedding(xy_disp * self.displacement_scale,
                                            num_bands=self.pos_embed_dim // 4, max_freq=32.0, include_xy=True)  # (B,N,M,D_pos+2)
                h_t = torch.cat([h_t, g], dim=-1)

                if self.query_critic == "feature":
                    candidate_emb = self.candidate_emb(torch.ones(B * N * M, dtype=torch.long, device=videos.device))
                    candidate_emb = candidate_emb.reshape(B, N, M, self.pos_embed_dim)
                    h_t = torch.cat([h_t, candidate_emb], dim=-1)

                # write into f_local
                f_local[:, t] = h_t

                # free per-frame temporaries
                del f_vis_frame, f2_frame, f3_frame, f4_frame
                del p1, p2, p3, p4, f_multiscale_frame, grid, h_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # free chunk-level temporaries immediately after processing the chunk
            del chunk_frames, chunk_resized, f_vis_chunk, f2_chunk, f3_chunk, f4_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --------------------------
        # 3) Continue with same downstream processing as forward (projection, transformer, heads)
        # --------------------------
        # free large pre-normalized buffers not needed anymore
        try:
            del videos_flat, videos_norm
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Project to common dimension
        f = self.local_projection(f_local)    # (B, T, N, M, D)
        q = self.local_projection(q_local)    # (B, N, D)
        q = q.unsqueeze(1).unsqueeze(3).expand(-1, T, -1, -1, -1)  # (B, T, N, 1, D) -> (B, T, N, 1, D)

        # Temporal mask
        temporal_mask = torch.arange(T, device=device).reshape(1, 1, T).expand(B, N, T)
        temporal_mask = (temporal_mask < queries[:, :, 0].unsqueeze(2)).bool()

        time_emb = self.interpolate_time_embedding(T)  # (1, T, D)


        for i in range(self.num_layers):
            q = self.layers[i](q, f, time_emb, t_mask=temporal_mask)    # (B, T, N, 1, D)

        q = q.expand(-1, -1, -1, M, -1)                                 # (B, T, N, 1, D) -> (B, T, N, M, D)

        q_normalized = F.normalize(self.q_projection(q), p=2, dim=-1)                       # (B, T, N, M, D)
        f_normalized = F.normalize(self.f_projection(f), p=2, dim=-1)                       # (B, T, N, M, D)

        similarities = torch.einsum("btnmd,btnmd->btnm", q_normalized, f_normalized)        # (B, T, N, M)
        rank_logits = similarities * self.sim_scale()                                       # (B, T, N, M)


        return rank_logits