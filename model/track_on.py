
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


from model.backbone import Backbone
from model.spatial_memory import Query_Updater
from model.query_decoder import Query_Decoder
from model.modules import Token_Decoder
from model.offset_head import Offset_Head
from model.rerank_module import Rerank_Module
from model.visibility_head import Visibility_Head
from model.loss import Loss_Function

from utils.coord_utils import get_points_on_a_grid
from utils.coord_utils import indices_to_coords



class TrackOn(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.N = args.N                 # number of queries
        self.T = args.T                 # number of frames
        self.visibility_treshold = 0.8
        self.top_k_regions = args.top_k_regions

        self.size = args.input_size
        self.stride = args.stride
        self.embedding_dim = args.transformer_embedding_dim
        self.cnn_corr = args.cnn_corr

        self.H_prime = self.size[0] // self.stride
        self.W_prime = self.size[1] // self.stride
        self.P = self.H_prime * self.W_prime
        self.num_levels = 4

        self.backbone = Backbone(args)
        self.sm_query_updater = Query_Updater(args)
        self.query_decoder = Query_Decoder(args)
        self.feature_decoder = Token_Decoder(args, use_norm=False)
        self.offset_head = Offset_Head(args)
        self.rerank_module = Rerank_Module(args)
        self.visibility_head = Visibility_Head(args)
        self.loss = Loss_Function(args)

        self.projection1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.projection2 = nn.Linear(self.embedding_dim, self.embedding_dim)

        if self.cnn_corr:
            self.corr_layer = nn.Sequential(
                nn.Conv2d(self.num_levels, self.num_levels * 4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(self.num_levels * 4, 1, kernel_size=3, stride=1, padding=1, bias=False)
            )
        else:
            self.corr_layer = nn.Conv2d(self.num_levels, 1, kernel_size=1, stride=1, padding=0, bias=False)


        self.extend_queries = False


    def correlation(self, q_t, f_t):
        # :args q_t: (B, N, C)
        # :args f_t: (B, P, C)

        B, N, C = q_t.shape
        
        q_normalized = F.normalize(q_t, p=2, dim=-1)    # (B, N, C)

        f = f_t.view(B, self.H_prime, self.W_prime, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        f1 = f.view(B, C, self.H_prime * self.W_prime)                                                      # (B, C, P)
        f2 = F.avg_pool2d(f, kernel_size=2, stride=2).view(B, C, (self.H_prime // 2) * (self.W_prime // 2)) # (B, C, P // 4)
        f3 = F.avg_pool2d(f, kernel_size=4, stride=4).view(B, C, (self.H_prime // 4) * (self.W_prime // 4)) # (B, C, P // 16)
        f4 = F.avg_pool2d(f, kernel_size=8, stride=8).view(B, C, (self.H_prime // 8) * (self.W_prime // 8)) # (B, C, P // 64)

        c1 = torch.einsum("bnc,bcp->bnp",  q_normalized, F.normalize(f1, p=2, dim=1))    # (B, N, P)
        c2 = torch.einsum("bnc,bcp->bnp", q_normalized, F.normalize(f2, p=2, dim=1))    # (B, N, P // 4)
        c3 = torch.einsum("bnc,bcp->bnp", q_normalized, F.normalize(f3, p=2, dim=1))    # (B, N, P // 16)
        c4 = torch.einsum("bnc,bcp->bnp", q_normalized, F.normalize(f4, p=2, dim=1))    # (B, N, P // 64)

        c1 = c1.view(B * N, 1, self.H_prime, self.W_prime)            # (B * N, 1, H, W)
        c2 = c2.view(B * N, 1, self.H_prime // 2, self.W_prime // 2)  # (B * N, 1, H // 2, W // 2)
        c3 = c3.view(B * N, 1, self.H_prime // 4, self.W_prime // 4)  # (B * N, 1, H // 4, W // 4)
        c4 = c4.view(B * N, 1, self.H_prime // 8, self.W_prime // 8)    # (B * N, 1, H // 8, W // 8)

        c1 = F.interpolate(c1, size=(self.H_prime, self.W_prime), mode="bilinear", align_corners=False)  # (B * N, 1, H, W)
        c2 = F.interpolate(c2, size=(self.H_prime, self.W_prime), mode="bilinear", align_corners=False)  # (B * N, 1, H, W)
        c3 = F.interpolate(c3, size=(self.H_prime, self.W_prime), mode="bilinear", align_corners=False)  # (B * N, 1, H, W)
        c4 = F.interpolate(c4, size=(self.H_prime, self.W_prime), mode="bilinear", align_corners=False)  # (B * N, 1, H, W)

        c = [c1, c2, c3, c4]

        c = torch.cat(c, dim=1)  # (B * N, num_levels, H, W)

        c = self.corr_layer(c) + c[:, 0:1] if self.cnn_corr else self.corr_layer(c)
        c = c.view(B, N, self.P)

        return c
    
    def set_memory_mask_ratio(self, p):
        self.sm_query_updater.random_mask_ratio = p
        self.query_decoder.random_mask_ratio = p

    def set_memory_size(self, context_memory_size, spatial_memory_size):
        
        # context memory
        pos_embedding = self.query_decoder.time_embedding  # (1, self.memory_size + 1, self.embedding_dim)
        pos_embedding_past = pos_embedding[:, :-1]
        pos_embedding_now = pos_embedding[:, -1].unsqueeze(dim=1)  # (1, 1, self.embedding_dim)
        
        pos_embedding_past = pos_embedding_past.permute(0, 2, 1)
        pos_embedding_interpolated = F.interpolate(pos_embedding_past, size=context_memory_size, mode='linear', align_corners=True)
        pos_embedding_interpolated = pos_embedding_interpolated.permute(0, 2, 1)
        self.query_decoder.time_embedding = nn.Parameter(torch.cat([pos_embedding_interpolated, pos_embedding_now], dim=1))
        
        self.query_decoder.memory_size = context_memory_size

        # spatial memory
        pos_embedding = self.sm_query_updater.time_embedding  # (1, self.memory_size + 1, self.embedding_dim)
        pos_embedding_past = pos_embedding[:, :-1]
        pos_embedding_now = pos_embedding[:, -1].unsqueeze(dim=1)  # (1, 1, self.embedding_dim)
        
        pos_embedding_past = pos_embedding_past.permute(0, 2, 1)
        pos_embedding_interpolated = F.interpolate(pos_embedding_past, size=spatial_memory_size, mode='linear', align_corners=True)
        pos_embedding_interpolated = pos_embedding_interpolated.permute(0, 2, 1)
        self.sm_query_updater.time_embedding = nn.Parameter(torch.cat([pos_embedding_interpolated, pos_embedding_now], dim=1))
        self.sm_query_updater.memory_size = spatial_memory_size

    def scale_inputs(self, queries, gt_tracks, H, W):
        # :args queries: (B, N, 3) where 3 is (t, y, x)
        # :args gt_tracks: (B, T, N, 2)

        device = queries.device

        queries[:, :, 2] = (queries[:, :, 2] / H) * self.size[0]
        queries[:, :, 1] = (queries[:, :, 1] / W) * self.size[1]

        if self.extend_queries:
            K = 20
            extra_queries = get_points_on_a_grid(K, self.size, device)           # (1, K ** 2, 2)
            extra_queries = torch.cat([torch.zeros(1, int(K ** 2), 1, device=device), extra_queries], dim=-1).to(device)
            queries = torch.cat([queries, extra_queries], dim=1)
            N = queries.shape[1]

        gt_tracks_tmp = gt_tracks.clone()
        gt_tracks_tmp[:, :, :, 1] = (gt_tracks_tmp[:, :, :, 1] / H) * self.size[0]
        gt_tracks_tmp[:, :, :, 0] = (gt_tracks_tmp[:, :, :, 0] / W) * self.size[1]

        return queries, gt_tracks_tmp

    def forward(self, video, queries, gt_tracks, gt_visibility):
        # :args video:      (B, T, C, H, W) in range [0, 255]
        # :args queries:    (B, N, 3) where 3 is (t, y, x)
        # :args tracks:     (B, T, N, 2) in pixel space
        # :args visibility: (B, T, N), in [0, 1]

        B, T, C, H, W = video.shape
        N = queries.shape[1]
        device = video.device

        out = {}


        # ##### Scale inputs #####
        queries_scaled, gt_tracks_scaled = self.scale_inputs(queries, gt_tracks, H, W)          # (B, N, 3), (B, T, N, 2)
        query_times = queries_scaled[:, :, 0].long()                                            # (B, N)
        # ##### ##### #####



        # ##### Feature Encoder #####
        tokens, q_init = self.backbone(video, queries_scaled)       # (B, T, P, C), (B, N, C)
        C = tokens.shape[-1]
        # ##### ##### #####



        # ##### Memory initialization #####
        max_memory_size = max([self.query_decoder.memory_size, self.sm_query_updater.memory_size])
        query_num = q_init.shape[1]

        # Spatial Memory
        spatial_memory = torch.zeros(B, query_num, max_memory_size, C, device=device)             # (B, N, max_memory_size, C)
        
        # Context Memory
        context_memory = torch.zeros(B, query_num, max_memory_size, C, device=device)             # (B, N, max_memory_size, C)

        # Masking
        past_occ = torch.ones(B, query_num, max_memory_size, device=device, dtype=torch.bool)     # (B, N, max_memory_size)
        past_mask = torch.ones(B, query_num, max_memory_size, device=device, dtype=torch.bool)  # (B, N, memory_size)

        # ##### ##### #####


        c1 = []          # correlation
        c2 = []          # correlation 2
        p_patch = []           # predicted patch locations
        v_logit = []           # visibility logits
        u_logit = []           # uncertainty logits
        o = []           # offsets

        p_patch_top = []        # top k patch locations
        u_top_logit = []        # top k uncertainties

        q_init_t = q_init.clone()                          # (B, N, C)
        p_head_t = queries_scaled[:, :, 1:]             # (B, N, 2)
        K = self.top_k_regions
        
        for t in range(T):
            queried_now_or_before = (query_times <= t)        # (B, N), True if queried now or before

            # ##### Spatial Memory - Query Update #####
            q_init_t = self.sm_query_updater(q_init, 
                                            spatial_memory,
                                            past_mask,
                                            past_occ,
                                            query_times,
                                            t)
            # ##### ##### #####

            # ##### Visual Encoder #####
            f_t = tokens[:, t]                                # (B, P, C)
            h_t = self.feature_decoder(f_t)                   # (B, P, C)
            # ##### ##### #####

            # ##### Query Decoder #####
            q_t = self.query_decoder(q_init_t, f_t, context_memory.clone(), past_mask, queried_now_or_before)       # (B, N, C)
            q_t = self.projection1(q_t)                                                                             # (B, N, C)
            # ##### ##### #####

            # ##### Correlation - 1 #####
            c1_t = self.correlation(q_t, h_t)                                                         # (B, N, P)
            # ##### ##### #####
            
            # ##### Reranking #####
            q_t, top_k_u_logit, top_k_p = self.rerank_module(q_t, h_t, c1_t)                           # (B, N, C), (B, N, K), (B, N, K, 2) 
            q_t_corr = self.projection2(q_t)                                                           # (B, N, C)
            # ##### ##### #####

            # ##### Correlation - 2 #####
            c2_t = self.correlation(q_t_corr, h_t)                                                                     # (B, N, P)
            p_head_patch_t = indices_to_coords(torch.argmax(c2_t, dim=-1).unsqueeze(1), self.size, self.stride)[:, 0]  # (B, N, 2)
            # ##### ##### #####

            # ##### Offset Prediction #####
            o_t = self.offset_head(q_t_corr, h_t, p_head_patch_t)           # (B, #offset_layers, N, 2)
            p_head_t = p_head_patch_t + o_t[:, -1]       # (B, N, 2)
            # ##### ##### #####

            # ##### Visibility and Uncertainty Prediction #####
            v_t_logit, u_t_logit = self.visibility_head(q_t, h_t, p_head_t)  # (B, N), (B, N)
            # ##### ##### #####

            # ##### Memory Update #####
            # Spatial Memory Update
            q_aug = self.sm_query_updater.get_augmented_memory(q_init, q_t, f_t, p_head_t, query_times, t)   # (B, N, C)
            spatial_memory = torch.cat([spatial_memory[:, :, 1:], q_aug.unsqueeze(2)], dim=2)                # (B, N, max_memory_size, C)

            # Context Memory Update
            context_memory = torch.cat([context_memory[:, :, 1:], q_t.unsqueeze(2)], dim=2)                                     # (B, N, max_memory_size, C)

            # Masking Update
            past_mask = torch.cat([past_mask[:, :, 1:], ~queried_now_or_before.unsqueeze(-1)], dim=2)                           # (B, N, memory_size)
            past_occ = torch.cat([past_occ[:, :, 1:], (F.sigmoid(v_t_logit) < self.visibility_treshold).unsqueeze(-1)], dim=2)  # (B, N, memory_size)
            # ##### ##### #####

            # ##### Accumulate Outputs #####
            c1.append(c1_t.unsqueeze(1))                               # (B, 1, N, P)
            c2.append(c2_t.unsqueeze(1))                               # (B, 1, N, P)
            p_patch.append(p_head_patch_t.unsqueeze(1))                # (B, 1, N, 2)
            v_logit.append(v_t_logit.unsqueeze(1))          # (B, 1, N)
            u_logit.append(u_t_logit.unsqueeze(1))          # (B, 1, N)
            o.append(o_t.unsqueeze(1))                      # (B, 1, #offset_layers, N, 2)

            p_patch_top.append(top_k_p.unsqueeze(1))          # (B, 1, N, K, 2)
            u_top_logit.append(top_k_u_logit.unsqueeze(1))    # (B, 1, N, K)
            # ##### ##### #####

        # ##### Outputs #####
        c1 = torch.cat(c1, dim=1)                         # (B, T, N, P)
        c2 = torch.cat(c2, dim=1)                         # (B, T, N, P)
        p_patch = torch.cat(p_patch, dim=1)               # (B, T, N, 2)
        v_logit = torch.cat(v_logit, dim=1)               # (B, T, N)
        u_logit = torch.cat(u_logit, dim=1)               # (B, T, N)
        o = torch.cat(o, dim=1)                           # (B, T, #offset_layers, N, 2)

        p_patch_top = torch.cat(p_patch_top, dim=1)       # (B, T, N, K, 2)
        u_top_logit = torch.cat(u_top_logit, dim=1)       # (B, T, N, K)
        # ##### ##### #####

        if self.extend_queries:
            c1 = c1[:, :, :N]                         # (B, T, N, P)
            c2 = c2[:, :, :N]                         # (B, T, N, P)
            p_patch = p_patch[:, :, :N]               # (B, T, N, 2)
            v_logit = v_logit[:, :, :N]               # (B, T, N)
            u_logit = u_logit[:, :, :N]               # (B, T, N)
            o = o[:, :, :, :N]                        # (B, T, #offset_layers, N, 2)

            p_patch_top = p_patch_top[:, :, :N]                     # (B, T, N, K, 2)
            u_top_logit = u_top_logit[:, :, :N]                     # (B, T, N, K)
            query_times = query_times[:, :N]        # (B, N)


        # ##### Loss #####
        # Point Loss
        out["point_loss"] = self.loss.point_loss(c1, gt_tracks_scaled, gt_visibility, query_times)
        out["point_loss_local"] = self.loss.point_loss(c2, gt_tracks_scaled, gt_visibility, query_times)

        # Visibility Loss
        out["visibility_loss"] = self.loss.visibility_loss(v_logit, gt_visibility, query_times)

        # Offset Loss
        out["offset_loss"] = self.loss.offset_loss(o, p_patch, gt_tracks_scaled, gt_visibility, query_times)
        
        # Uncertainty Loss
        point_pred = p_patch + o[:, :, -1]  # (B, T, N, 2)
        if self.loss.lambda_uncertainty > 0:
            out["uncertainty_loss"] = self.loss.uncertainty_loss(u_logit, point_pred, gt_tracks_scaled, gt_visibility, query_times) * self.loss.lambda_uncertainty

        # Uncertainty Loss - Top K
        l_u_k = 0
        for k in range(self.top_k_regions):
            point_k = p_patch_top[:, :, :, k]           # (B, T, N, 2)
            uncertainty_k = u_top_logit[:, :, :, k]     # (B, T, N)
            l_u_k += (self.loss.uncertainty_loss(uncertainty_k, point_k, gt_tracks_scaled, gt_visibility, query_times) * self.loss.lambda_top_k)

        out["uncertainty_loss_topk"] = (l_u_k / self.top_k_regions)
        # ##### ##### #####


        # <Outputs>
        point_pred[:, :, :, 1] = (point_pred[:, :, :, 1] / self.size[0]) * H
        point_pred[:, :, :, 0] = (point_pred[:, :, :, 0] / self.size[1]) * W
        out["points"] = point_pred

        out["visibility"] = F.sigmoid(v_logit) > self.visibility_treshold

        return out
    
    def inference(self, video, queries, K=20):
        # :args video:      (B, T, C, H, W) in range [0, 255]
        # :args queries:    (B, N, 3) where 3 is (t, y, x)
        

        B, T, C, H, W = video.shape
        N = queries.shape[1]
        device = video.device

        out = {}

        # ##### Scale inputs #####
        queries[:, :, 2] = (queries[:, :, 2] / H) * self.size[0]
        queries[:, :, 1] = (queries[:, :, 1] / W) * self.size[1]

        if self.extend_queries:
            extra_queries = get_points_on_a_grid(K, self.size, device)           # (1, K ** 2, 2)
            extra_queries = torch.cat([torch.zeros(1, int(K ** 2), 1, device=device), extra_queries], dim=-1).to(device)
            queries = torch.cat([queries, extra_queries], dim=1)        # (B, N + K ** 2, 3)

        query_times = queries[:, :, 0].long()       
        
        # ##### ##### #####


        # ##### Feature Encoder #####
        # Extract Queries
        q_init = self.backbone.sample_queries_online(video, queries)        # (B, N, C)
        C = q_init.shape[-1]
        # ##### ##### #####

        # ##### Memory initialization #####
        max_memory_size = max([self.query_decoder.memory_size, self.sm_query_updater.memory_size])
        query_num = q_init.shape[1]

        # Spatial Memory
        spatial_memory = torch.zeros(B, query_num, max_memory_size, C, device=device)             # (B, N, max_memory_size, C)

        # Context Memory
        context_memory = torch.zeros(B, query_num, max_memory_size, C, device=device)             # (B, N, max_memory_size, C)

        # Masking
        past_occ = torch.ones(B, query_num, max_memory_size, device=device, dtype=torch.bool)     # (B, N, max_memory_size)
        past_mask = torch.ones(B, query_num, max_memory_size, device=device, dtype=torch.bool)

        # ##### ##### #####

        coord_pred = []          # predicted coordinates
        vis_pred = []            # visibility logits

        for t in range(T):
            queried_now_or_before = (query_times <= t)

            # ##### Spatial Memory - Query Update #####
            q_init_t = self.sm_query_updater(q_init, 
                                            spatial_memory,
                                            past_mask,
                                            past_occ,
                                            query_times,
                                            t)
            # ##### ##### #####

            # ##### Visual Encoder #####
            f_t = self.backbone.encode_frames_online(video[:, t])     # (B, P, C)
            h_t = self.feature_decoder(f_t)                           # (B, P, C)
            # ##### ##### #####

            # ##### Query Decoder #####
            q_t = self.query_decoder(q_init_t, f_t, context_memory.clone(), past_mask, queried_now_or_before)       # (B, N, C)
            q_t = self.projection1(q_t)                                                                             # (B, N, C)
            # ##### ##### #####

            # ##### Correlation - 1 #####
            c1_t = self.correlation(q_t, h_t)                                                         # (B, N, P)
            # ##### ##### #####
            
            # ##### Reranking #####
            q_t, top_k_u_logit, top_k_p = self.rerank_module(q_t, h_t, c1_t)                           # (B, N, C), (B, N, K), (B, N, K, 2) 
            q_t_corr = self.projection2(q_t)                                                           # (B, N, C)
            # ##### ##### #####

            # ##### Correlation - 2 #####
            c2_t = self.correlation(q_t_corr, h_t)                                                                     # (B, N, P)
            p_head_patch_t = indices_to_coords(torch.argmax(c2_t, dim=-1).unsqueeze(1), self.size, self.stride)[:, 0]  # (B, N, 2)
            # ##### ##### #####

            # ##### Offset Prediction #####
            o_t = self.offset_head(q_t_corr, h_t, p_head_patch_t)           # (B, #offset_layers, N, 2)
            p_head_t = p_head_patch_t + o_t[:, -1]       # (B, N, 2)
            # ##### ##### #####

            # ##### Visibility and Uncertainty Prediction #####
            v_t_logit, u_t_logit = self.visibility_head(q_t, h_t, p_head_t)  # (B, N), (B, N)
            # ##### ##### #####

            # ##### Memory Update #####
            # Spatial Memory Update
            q_aug = self.sm_query_updater.get_augmented_memory(q_init, q_t, f_t, p_head_t, query_times, t)   # (B, N, C)
            spatial_memory = torch.cat([spatial_memory[:, :, 1:], q_aug.unsqueeze(2)], dim=2)                # (B, N, max_memory_size, C)

            # Context Memory Update
            context_memory = torch.cat([context_memory[:, :, 1:], q_t.unsqueeze(2)], dim=2)                                     # (B, N, max_memory_size, C)

            # Masking Update
            past_mask = torch.cat([past_mask[:, :, 1:], ~queried_now_or_before.unsqueeze(-1)], dim=2)                           # (B, N, memory_size)
            past_occ = torch.cat([past_occ[:, :, 1:], (F.sigmoid(v_t_logit) < self.visibility_treshold).unsqueeze(-1)], dim=2)  # (B, N, memory_size)
            # ##### ##### #####

            coord_pred.append(p_head_t.unsqueeze(1))                    # (B, 1, N, 2)
            vis_pred.append(v_t_logit.unsqueeze(1))                     # (B, 1, N)

        # ##### Outputs #####
        coord_pred = torch.cat(coord_pred, dim=1)       # (B, T, N_prime, 2)
        vis_pred = torch.cat(vis_pred, dim=1)           # (B, T, N_prime)

        if self.extend_queries:
            coord_pred = coord_pred[:, :, :N]           # (B, T, N, 2)
            vis_pred = vis_pred[:, :, :N]               # (B, T, N)

        coord_pred[:, :, :, 1] = (coord_pred[:, :, :, 1] / self.size[0]) * H
        coord_pred[:, :, :, 0] = (coord_pred[:, :, :, 0] / self.size[1]) * W

        out["points"] = coord_pred
        out["visibility"] = F.sigmoid(vis_pred) > self.visibility_treshold

        return out





        

