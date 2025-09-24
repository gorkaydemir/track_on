import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torch

from model.backbone import Backbone
from model.modules import MHA_Block, SimpleFPN
from model.prediction_head import Prediction_Head
from model.reranking import Rerank_Module

from utils.coord_utils import indices_to_coords

class Track_On2(nn.Module):
    def __init__(self, args, nhead=4):
        super().__init__()

        # === Hyper-parameters ===
        self.K = args.K                  # Top-K regions for re-ranking                 
        self.D = args.D                  # Feature dimension    
        self.M = args.M                  # Memory size
        
        self.decoder_layer_num = args.decoder_layer_num     # Decoder layer num
        # === === ===

        # === Size related ===
        self.input_size = args.input_size
        self.stride = 4
        self.H = self.input_size[0]         # Original image size
        self.W = self.input_size[1]
        self.Hf = self.H // self.stride     # Feature map size:    
        self.Wf = self.W // self.stride
        self.P = int(self.Hf * self.Wf)     # Number of patches in the feature map:
        # === === ===

        # === Modules ===
        self.backbone = Backbone(args)
        self.fpn = SimpleFPN(args)
        self.reranking_head = Rerank_Module(args, nhead)
        self.prediction_head = Prediction_Head(args, nhead)

        # Query Decoder Layers
        self.feature_attention = self._make_transformer_layer(self.decoder_layer_num, nhead)
        self.query_attention = self._make_transformer_layer(self.decoder_layer_num, nhead)
        self.memory_attention = self._make_transformer_layer(self.decoder_layer_num, nhead)

        self.projection1 = nn.Linear(args.D, args.D)
        self.projection2 = nn.Linear(args.D, args.D)
        self.ms_corr_proj = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)
        # === === ===

        # === Time Positional Embedding for memory ===
        self.t_embedding = nn.Parameter(torch.zeros(1, self.M + 1, self.D))
        nn.init.trunc_normal_(self.t_embedding, std=0.02)
        # === === ===

    def _make_transformer_layer(self, layer_num, nhead=4):
        layers = []
        for _ in range(layer_num):
            layer = MHA_Block(self.D, nhead)
            layers.append(layer)
        return nn.ModuleList(layers)


    def memory_extension(self, new_memory_size):
        pos_embedding = self.t_embedding.clone()  # (1, self.memory_size + 1, self.embedding_dim)
        pos_embedding_past = pos_embedding[:, :-1]
        pos_embedding_now = pos_embedding[:, -1].unsqueeze(dim=1)  # (1, 1, self.embedding_dim)
        
        pos_embedding_past = pos_embedding_past.permute(0, 2, 1)
        pos_embedding_interpolated = F.interpolate(pos_embedding_past, size=new_memory_size, mode='linear', align_corners=True)
        pos_embedding_interpolated = pos_embedding_interpolated.permute(0, 2, 1)
        self.t_embedding = nn.Parameter(torch.cat([pos_embedding_interpolated, pos_embedding_now], dim=1))

        self.M = new_memory_size
        
    def multiscale_correlation(self, q, h4, h8, h16, h32):
        # :args q: (1, N, D)
        # :args h4: (1, P, D)
        # :args f8: (1, P // 4, D)
        # :args f16: (1, P // 16, D)
        # :args f32: (1, P // 32, D)

        N = q.shape[1]                        # Number of queries
        _, P, D = h4.shape                    # Number of tokens

        q_normalized = F.normalize(q, p=2, dim=-1)    # (1, N, D)

        c4 = torch.einsum("bnd,bpd->bnp",  q_normalized, F.normalize(h4, p=2, dim=-1))    # (1, N, P)
        c8 = torch.einsum("bnd,bpd->bnp",  q_normalized, F.normalize(h8, p=2, dim=-1))    # (1, N, P // 4)
        c16 = torch.einsum("bnd,bpd->bnp",  q_normalized, F.normalize(h16, p=2, dim=-1))  # (1, N, P // 16)
        c32 = torch.einsum("bnd,bpd->bnp",  q_normalized, F.normalize(h32, p=2, dim=-1))  # (1, N, P // 32)

        c4 = c4.view(N, self.Hf, self.Wf)                      # (N, H4, W4)
        c8 = c8.view(N, self.Hf // 2, self.Wf // 2)            # (N, H8, W8)
        c16 = c16.view(N, self.Hf // 4, self.Wf // 4)          # (N, H16, W16)
        c32 = c32.view(N, self.Hf // 8, self.Wf // 8)          # (N, H32, W32)

        c4 = c4.unsqueeze(1) # (N, 1, H4, W4)
        c8 = F.interpolate(c8.unsqueeze(1), size=(self.Hf, self.Wf), mode="bilinear", align_corners=False)   # (N, 1, H4, W4)
        c16 = F.interpolate(c16.unsqueeze(1), size=(self.Hf, self.Wf), mode="bilinear", align_corners=False) # (N, 1, H4, W4)
        c32 = F.interpolate(c32.unsqueeze(1), size=(self.Hf, self.Wf), mode="bilinear", align_corners=False) # (N, 1, H4, W4)

        c = torch.cat([c4, c8, c16, c32], dim=1)     # (N, 4, H4, W)
        c = self.ms_corr_proj(c)                     # (N, 1, H4, W)

        c = c.view(1, N, self.P)                     # (N, P)

        return c
    
    # Multiscale ViT-Adapter
    def forward(self, video, queries, p_mask=0.1):
        # :args video: (B, T, 3, H_in, W_in) in range [0, 255]
        # :args gt_tracks: (B, T, N, 2) in range [0, H_in] and [0, W_in]
        # :args queries: (B, N, 3) where 3 is (t, x, y) in range [0, 255]
        B, T, _, H_in, W_in = video.shape
        M = self.M
        D = self.D
        N = queries.shape[1]                        # Number of queries
        K = self.K
        layer_num_pred_head = self.prediction_head.layer_num
        device = video.device

        # === Scaling ====
        # scale to [0, 1] range
        queries[..., 2] = (queries[..., 2] / H_in)
        queries[..., 1] = (queries[..., 1] / W_in)

        video = video.view(B * T, 3, H_in, W_in) / 255.0                        # (B * T, 3, H, W), to [0, 1]
        video = F.interpolate(video, size=self.input_size, mode="bilinear", align_corners=False)
        video = video.view(B, T, 3, self.H, self.W)                             # (B, T, 3, H, W)
        # === === ===

        # == Feature Projection ==
        f4, f8, f16, f32 = self.backbone(video)                           # (B, T, P, D), (B, T, P // 4, D), (B, T, P // 16, D), (B, T, P // 64, D)
        f_fused = self.fpn(f4, f8, f16, f32)                              # (B, T, P, D)
        # === === ===

        # === Sample query features from f_fused ===
        query_init = torch.zeros(B, N, D, device=device)            # empty tensor of shape B, N, D
        query_times = queries[:, :, 0].long()                       # (B, N)
        query_coords_norm = queries[..., 1:].clone()                # (B, N, 2)
        query_coords_norm  = query_coords_norm * 2 - 1              # (B, N, 2), in range [-1, 1]
        for b in range(B):

            for t in range(T):
                mask = (query_times[b] == t)        # (N,)
                if not mask.any():
                    continue

                f_b_t = f_fused[b, t].view(1, self.Hf, self.Wf, self.D)                     # (1, H4, W4, D)
                f_b_t = f_b_t.permute(0, 3, 1, 2)                                           # (1, D, H4, W4)
                idxs = mask.nonzero(as_tuple=True)[0]                           # (N_t,)
                pts = query_coords_norm[b, idxs].unsqueeze(0).unsqueeze(2)                                         # (1, N_t, 1, 2)

                # Sample Points
                centers = F.grid_sample(f_b_t, pts, mode='bilinear', padding_mode='border', align_corners=False)   # (1, D, N_t, 1)
                centers = centers.squeeze(-1).squeeze(0).permute(1, 0)                                             # (N_t, D)
                query_init[b, idxs] = centers                                                                      # (N_t, D)
            # === === ===

        # === Memory initialization ===
        point_memory = torch.zeros(B, N, M, D, device=device)                             # (B, N, M, D)
        temporal_mask = torch.ones(B, N, M, device=device, dtype=torch.bool)              # True if masked, (B, N, M)
        # === === ===

        # === Output initialization ===
        C1 = torch.zeros(B, T, N, self.P, device=device)                        # (B, T, N, P)
        C2 = torch.zeros(B, T, N, self.P, device=device)                        # (B, T, N, P)
        O = torch.zeros(B, T, layer_num_pred_head, N, 2, device=device)         # (B, layer_num_pred_head, T, N, 2)

        V_logit = torch.zeros(B, T, N, device=device)      # (B, T, N)
        U_logit = torch.zeros(B, T, N, device=device)      # (B, T, N)
        
        P_patch = torch.zeros(B, T, N, 2, device=device)                        # (B, T, N, 2)

        U_logit_top_k = torch.zeros(B, T, N, K, device=device)                  # (B, T, N, K)
        S_logit_top_k = torch.zeros(B, T, N, K, device=device)                  # (B, T, N, K)
        P_patch_top_k = torch.zeros(B, T, N, K, 2, device=device)               # (B, T, N, K, 2)
        # === === ===

        for t in range(T):
            for b in range(B):
                available_queries = (query_times[b] <= t)                        # (N,)
            
                N_t = available_queries.sum().item()                             # Number of available queries in this batch
                if N_t == 0:
                    print(f"Warning: No available queries at time {t} for batch {b}. Skipping.")
                    continue
                
                memory_mask_t = temporal_mask[b, available_queries]                # (N_t, M)
                memory = point_memory[b, available_queries].clone()              # (N_t, M, D)
                
                q_t = query_init[b, available_queries].clone().unsqueeze(0)      # (1, N_t, D)

                f_fused_t = f_fused[b, t].unsqueeze(0)                           # (1, P, D)
                f4_t = f4[b, t].unsqueeze(0)                                     # (1, P, D)
                f8_t = f8[b, t].unsqueeze(0)                                     # (1, P // 4, D)
                f16_t = f16[b, t].unsqueeze(0)                                   # (1, P // 16, D)
                f32_t = f32[b, t].unsqueeze(0)                                   # (1, P // 64, D)

                # === Query Decoder ===
                for i in range(self.decoder_layer_num):

                    # === Attention to frame features ===
                    q_t = self.feature_attention[i](q_t,  f_fused_t, f_fused_t)         # (1, N_t, D)
                    # === === ===

                    # === Attention to other queries ===
                    q_t = self.query_attention[i](q_t, q_t, q_t)        # (1, N_t, D)
                    # === === ===

                    # === Attention to memory ===
                    q_t = q_t.view(N_t, 1, D)                                      # (N, 1, D)

                    memory_mask_noised = memory_mask_t | (torch.rand_like(memory_mask_t.float(), device=device) < p_mask)  # random masking
                    qkv = torch.cat([memory, q_t], dim=1)                                                           # (N_t, M + 1, D)
                    mask = torch.cat([memory_mask_noised, torch.zeros(N_t, 1, device=device).bool()], dim=1)        # (N_t, M + 1)


                    qkv = self.memory_attention[i](qkv + self.t_embedding, qkv + self.t_embedding, qkv, mask)                           # (N_t, M + 1, D)

                    q_t = qkv[:, -1].unsqueeze(0)                                   # (1, N_t, D)
                    memory = qkv[:, :-1]                                            # (N_t, M, D)
                    # === === ===

                q_t = self.projection1(q_t)
                # === === ===

                # === Re-ranking ===
                c1 = self.multiscale_correlation(q_t, f4_t, f8_t, f16_t, f32_t)                             # (1, N_t, P)

                q_t, p_patch_top_k, u_logit_top_k, s_logit_top_k = self.reranking_head(q_t, f4_t, f8_t, f16_t, f32_t, c1)  # (1, N_t, D), (1, N_t, K, 2), (1, N_t, K), (1, N_t, K)
                q_t = self.projection2(q_t)                                                             # (1, N_t, D)
                c2 = self.multiscale_correlation(q_t, f4_t, f8_t, f16_t, f32_t)                                        # (1, N_t, P)

                p_patch = indices_to_coords(torch.argmax(c2.detach(), dim=-1).unsqueeze(1), self.input_size, self.stride)   # (1, 1, N_t, 2), in range [H, W]
                p_patch = p_patch.squeeze(1)                                                                               # (1, N_t, 2)
                # === === ===

                # === Prediction Head ===
                o, v_logit, u_logit = self.prediction_head(q_t, f4_t, f8_t, f16_t, f32_t, p_patch)        # (layer_num, N_t, 2), (N_t), (N_t)
                # === === ===

                C1[b, t, available_queries] = c1.squeeze(0).float()         # (1, N_t, P) 
                C2[b, t, available_queries] = c2.squeeze(0).float()         # (1, N_t, P)
                O[b, t, :, available_queries] = o.float()                                        # (N, 2)
                V_logit[b, t, available_queries] = v_logit.float()                # (N)
                U_logit[b, t, available_queries] = u_logit.float()                # (N)
                P_patch[b, t, available_queries] = p_patch.squeeze(1).float()                 # (N, 2)

                U_logit_top_k[b, t, available_queries] = u_logit_top_k.squeeze(0).float()      # (N, K)
                S_logit_top_k[b, t, available_queries] = s_logit_top_k.squeeze(0).float()      # (N, K)
                P_patch_top_k[b, t, available_queries] = p_patch_top_k.squeeze(0).float()      # (N, K, 2)

                # === Updates ===
                # Update Memory
                point_memory_slice = torch.zeros(N, 1, D, device=device, dtype=q_t.dtype)  # (N, 1, D)
                point_memory_slice[available_queries] = q_t.clone().view(N_t, 1, D)
                point_memory[b] = torch.cat([point_memory[b, :, 1:], point_memory_slice], dim=1)                # (N, M, D)

                # Update Temporal Mask
                temporal_mask[b] = torch.cat([temporal_mask[b, :, 1:], ~available_queries.view(N, 1)], dim=1)   # (N, M)

            
        # P = indices_to_coords(torch.argmax(C1.detach(), dim=-1), self.input_size, self.stride)   # (B, T, N, 2), in range [H, W]
        P = P_patch + O[:, :, -1]
        P[..., 0] = (P[..., 0] / self.W) * W_in
        P[..., 1] = (P[..., 1] / self.H) * H_in

        # === Output ===
        # disctionary
        out = {"P": P, 
                "C1": C1, "C2": C2, "O": O,
                "V_logit": V_logit, "U_logit": U_logit, 
                "P_patch": P_patch, 
                "U_logit_top_k": U_logit_top_k, "S_logit_top_k": S_logit_top_k, "P_patch_top_k": P_patch_top_k}
        

        return out
    

    def forward_online(self, video, queries):

        B, T, _, H_in, W_in = video.shape
        M = self.M
        D = self.D
        N = queries.shape[1]                        # Number of queries
        K = self.K
        device = video.device

        # === Scaling ====
        # scale to [0, 1] range
        queries[..., 2] = (queries[..., 2] / H_in)
        queries[..., 1] = (queries[..., 1] / W_in)

        video = video.view(B * T, 3, H_in, W_in) / 255.0                        # (B * T, 3, H, W), to [0, 1]
        video = F.interpolate(video, size=self.input_size, mode="bilinear", align_corners=False)
        video = video.view(B, T, 3, self.H, self.W)                             # (B, T, 3, H, W)
        # === === ===

        # === Sample query features from f_fused ===
        query_init = torch.zeros(B, N, D, device=device)            # empty tensor of shape B, N, D
        query_times = queries[:, :, 0].long()                       # (B, N)
        query_coords_norm = queries[..., 1:].clone()                # (B, N, 2)
        query_coords_norm  = query_coords_norm * 2 - 1              # (B, N, 2), in range [-1, 1]

        # === Memory initialization ===
        point_memory = torch.zeros(B, N, M, D, device=device)                             # (B, N, M, D)
        temporal_mask = torch.ones(B, N, M, device=device, dtype=torch.bool)              # True if masked, (B, N, M)
        # === === ===

        # === Output initialization ===
        V_logit = torch.zeros(B, T, N, device=device)      # (B, T, N)
        U_logit = torch.zeros(B, T, N, device=device)      # (B, T, N)
        P = torch.zeros(B, T, N, 2, device=device)                        # (B, T, N, 2)
        # === === ===

        for t in range(T):
            for b in range(B):

                available_queries = (query_times[b] <= t)                        # (N,)

                N_t = available_queries.sum().item()                             # Number of available queries in this batch
                if N_t == 0:
                    continue
                
                memory_mask_t = temporal_mask[b, available_queries]                # (N_t, M)
                memory = point_memory[b, available_queries].clone()                # (N_t, M, D)

                # === Visual Backbone Feedforward ===
                f4_t, f8_t, f16_t, f32_t = self.backbone(video[b, t].unsqueeze(0).unsqueeze(0))     # (1, 1, P, D), (1, 1, P // 4, D), (1, 1, P // 16, D), (1, 1, P // 64, D)
                f_fused_t = self.fpn(f4_t, f8_t, f16_t, f32_t)                                      # (1, 1, P, D)
                
                f4_t = f4_t.squeeze(0)                                                              # (1, P, D)
                f8_t = f8_t.squeeze(0)                                                              # (1, P // 4, D)
                f16_t = f16_t.squeeze(0)                                                            # (1, P // 16, D)
                f32_t = f32_t.squeeze(0)                                                            # (1, P // 64, D)
                f_fused_t = f_fused_t.squeeze(0)                                                    # (1, P, D)
                # === === ===

                # === Sample query features from f_fused ===
                queried_now = query_times[b] == t                                           # (N,)
                if queried_now.any():
                    f_b_t = f_fused_t.view(1, self.Hf, self.Wf, self.D)                     # (1, H4, W4, D)
                    f_b_t = f_b_t.permute(0, 3, 1, 2)                                       # (1, D, H4, W4)
                    idxs = queried_now.nonzero(as_tuple=True)[0]                            # (N_now_t,)
                    pts = query_coords_norm[b, idxs].unsqueeze(0).unsqueeze(2)                                         # (1, N_now_t, 1, 2)

                    # Sample Points
                    centers = F.grid_sample(f_b_t, pts, mode='bilinear', padding_mode='border', align_corners=False)   # (1, D, N_now_t, 1)
                    centers = centers.squeeze(-1).squeeze(0).permute(1, 0)                                             # (N_now_t, D)
                    query_init[b, idxs] = centers                                                                      # (N_now_t, D)
                # === === ===
                
                q_t = query_init[b, available_queries].clone().unsqueeze(0)      # (1, N_t, D)


                # === Query Decoder ===
                for i in range(self.decoder_layer_num):
                    # === Attention to frame features ===
                    q_t = self.feature_attention[i](q_t, 
                                                f_fused_t, 
                                                f_fused_t)         # (1, N_t, D)
                    # === === ===

                    # === Attention to other queries ===
                    q_t = self.query_attention[i](q_t, q_t, q_t)        # (1, N_t, D)
                    # === === ===

                    # === Attention to memory ===
                    q_t = q_t.view(N_t, 1, D)                                      # (N, 1, D)

                    qkv = torch.cat([memory, q_t], dim=1)                                                      # (N_t, M + 1, D)
                    mask = torch.cat([memory_mask_t, torch.zeros(N_t, 1, device=device).bool()], dim=1)        # (N_t, M + 1)
                    qkv = self.memory_attention[i](qkv + self.t_embedding,
                                                    qkv + self.t_embedding,
                                                    qkv,
                                                    mask)                           # (N_t, M + 1, D)
                    q_t = qkv[:, -1].unsqueeze(0)                                   # (1, N_t, D)
                    memory = qkv[:, :-1]                                            # (N_t, M, D)
                    # === === ===

                q_t = self.projection1(q_t)
                # === === ===

                # === Re-ranking ===
                c1 = self.multiscale_correlation(q_t, f4_t, f8_t, f16_t, f32_t)                             # (1, N_t, P)
                q_t, _, _, _ = self.reranking_head(q_t, f4_t, f8_t, f16_t, f32_t, c1)  # (1, N_t, D), (1, N_t, K, 2), (1, N_t, K), (1, N_t, K)
                q_t = self.projection2(q_t)                                                            # (1, N_t, D)

                c2 = self.multiscale_correlation(q_t, f4_t, f8_t, f16_t, f32_t)                                        # (1, N_t, P)
                p_patch = indices_to_coords(torch.argmax(c2.detach(), dim=-1).unsqueeze(1), self.input_size, self.stride)   # (1, 1, N_t, 2), in range [H, W]
                p_patch = p_patch.squeeze(1)                                                                               # (1, N_t, 2)
                # === === ===

                # === Prediction Head ===
                o, v_logit, u_logit = self.prediction_head(q_t, f4_t, f8_t, f16_t, f32_t, p_patch)        # (layer_num, N_t, 2), (N_t), (N_t)
                # === === ===

                P[b, t, available_queries] = p_patch[0] + o[-1]                   # (1, N_t, 2)
                V_logit[b, t, available_queries] = v_logit.float()                # (N)
                U_logit[b, t, available_queries] = u_logit.float()                # (N)

                # === Updates ===
                # Update Memory
                point_memory_slice = torch.zeros(N, 1, D, device=device, dtype=q_t.dtype)  # (N, 1, D)
                point_memory_slice[available_queries] = q_t.clone().view(N_t, 1, D)
                point_memory[b] = torch.cat([point_memory[b, :, 1:], point_memory_slice], dim=1)                # (N, M, D)

                # Update Temporal Mask
                temporal_mask[b] = torch.cat([temporal_mask[b, :, 1:], ~available_queries.view(N, 1)], dim=1)   # (N, M)

        # === Output ===
        P[..., 0] = (P[..., 0] / self.W) * W_in
        P[..., 1] = (P[..., 1] / self.H) * H_in
        out = {"P": P, "V_logit": V_logit, "U_logit": U_logit}

        return out



    def forward_frame(self, q_init, temporal_mask, point_memory, frame):
        # :args q_init: (N, D)
        # :args temporal_mask: (N, M) True if masked
        # :args point_memory: (N, M, D)
        # :args frame: (1, 3, H_in, W_in) in range [0, 255]

        B, N, D = q_init.shape
        M = point_memory.shape[2]
        device = frame.device

        q_t = q_init.unsqueeze(0).clone()                  # (1, N, D)
        memory_mask_t = temporal_mask.clone()              # (N, M)
        memory = point_memory.clone()                      # (N, M, D)

        # === Scaling ====
        frame = frame / 255.0                               # to [0, 1]
        frame = F.interpolate(frame, 
                              size=self.input_size, 
                              mode="bilinear", 
                              align_corners=False)          # (1, 3, H, W)
        frame = frame.unsqueeze(1)                          # (1, 1, 3, H, W)
        # === === ===

        # === Visual Backbone Feedforward ===
        f4_t, f8_t, f16_t, f32_t = self.backbone(frame)     # (1, 1, P, D), (1, 1, P // 4, D), (1, 1, P // 16, D), (1, 1, P // 64, D)
        f_fused_t = self.fpn(f4_t, f8_t, f16_t, f32_t)      # (1, 1, P, D)
        f4_t = f4_t.squeeze(0)                                                              # (1, P, D)
        f8_t = f8_t.squeeze(0)                                                              # (1, P // 4, D)
        f16_t = f16_t.squeeze(0)                                                            # (1, P // 16, D)
        f32_t = f32_t.squeeze(0)                                                            # (1, P // 64, D)
        f_fused_t = f_fused_t.squeeze(0)                                                    # (1, P, D)
        # === === ===

        # === Query Decoder ===
        for i in range(self.decoder_layer_num):
            # === Attention to frame features ===
            q_t = self.feature_attention[i](q_t, 
                                        f_fused_t, 
                                        f_fused_t)              # (1, N, D)
            # === === ===

            # === Attention to other queries ===
            q_t = self.query_attention[i](q_t, q_t, q_t)        # (1, N, D)
            # === === ===

            # === Attention to memory ===
            q_t = q_t.view(N, 1, D)                             # (N, 1, D)

            qkv = torch.cat([memory, q_t], dim=1)                                                    # (N, M + 1, D)
            mask = torch.cat([memory_mask_t, torch.zeros(N, 1, device=device).bool()], dim=1)        # (N, M + 1)
            qkv = self.memory_attention[i](qkv + self.t_embedding,
                                            qkv + self.t_embedding,
                                            qkv,
                                            mask)                           # (N, M + 1, D)
            q_t = qkv[:, -1].unsqueeze(0)                                   # (1, N, D)
            memory = qkv[:, :-1]                                            # (N, M, D)
            # === === ===

        q_t = self.projection1(q_t)
        # === === ===

        # === Re-ranking ===
        c1 = self.multiscale_correlation(q_t, f4_t, f8_t, f16_t, f32_t)                             # (1, N, P)
        q_t, _, _, _ = self.reranking_head(q_t, f4_t, f8_t, f16_t, f32_t, c1)  # (1, N, D), (1, N, K, 2), (1, N, K), (1, N, K)
        q_t = self.projection2(q_t)                                                            # (1, N, D)

        c2 = self.multiscale_correlation(q_t, f4_t, f8_t, f16_t, f32_t)                                        # (1, N, P)
        p_patch = indices_to_coords(torch.argmax(c2.detach(), dim=-1).unsqueeze(1), self.input_size, self.stride)   # (1, 1, N, 2), in range [H, W]
        p_patch = p_patch.squeeze(1)                                                                               # (1, N, 2)
        # === === ===

        # === Prediction Head ===
        o, v_logit, u_logit = self.prediction_head(q_t, f4_t, f8_t, f16_t, f32_t, p_patch)        # (layer_num, N, 2), (N), (N)
        # === === ===

        p = p_patch[0] + o[-1]                   # (1, N, 2)
        p = p[0]

        return p, v_logit, q_t.detach()