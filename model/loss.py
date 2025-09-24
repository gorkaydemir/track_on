import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from utils.coord_utils import coords_to_indices, indices_to_coords

class Loss_Function(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.size = args.input_size
        self.stride = 4
        self.epsilon_smoothing = 0.0
        self.lambda_patch_cls = args.lambda_patch_cls
        self.temperature = 0.05


    def forward(self, out, gt_tracks, gt_visibility, query_times):
        # :args out: dict
        #        C1: (B, T, N, P)
        #        P: (B, T, N, 2)
        #        V_logit: (B, T, N)
        #        U_logit: (B, T, N)
        #        O: (B, T, #layers, N, 2)
        #        P_patch: (B, T, N, 2)
        #        U_logit_top_k: (B, T, N, K)
        #        S_logit_top_k: (B, T, N, K)
        #        P_patch_top_k: (B, T, N, K, 2)
        #
        # :args gt_tracks: (B, T, N, 2)
        # :args gt_visibility: (B, T, N)
        # :args query_times: (B, N)


        # :return loss:

        B, T, N, _ = gt_tracks.shape
        C1 = out["C1"]                                          # (B, T, N, P)
        C2 = out["C2"]                                          # (B, T, N, P)
        O = out["O"]                                            # (B, T, #layers, N, 2)
        V = out["V_logit"]                                   # (B, T, #layers, N)
        U = out["U_logit"]                                   # (B, T, #layers, N)
        P_patch = out["P_patch"]                             # (B, T, N, 2)

        layer_num_pred_head = O.shape[2]                    # Number of layers in prediction head

        # Top-K related
        U_top_k = out["U_logit_top_k"]                       # (B, T, N, K)
        S_top_k = out["S_logit_top_k"]                       # (B, T, N, K)
        P_patch_top_k = out["P_patch_top_k"]                 # (B, T, N, K, 2)
        K = P_patch_top_k.shape[-2]                        # Top-K

        # === Point Loss ===
        L_p = self.patch_classification_loss(C1, gt_tracks, gt_visibility, query_times)        # (B * T * N)
        L_p2 = self.patch_classification_loss(C2, gt_tracks, gt_visibility, query_times)       # (B * T * N)

        # == Offset Loss ===
        L_o = 0
        for i in range(layer_num_pred_head):
            L_o += self.offset_loss(O[:, :, i], P_patch, gt_tracks, gt_visibility, query_times)
        L_o /= layer_num_pred_head
                               
        # === Visibility Loss ===
        L_vis = self.visibility_loss(V, gt_visibility, query_times)

        # === Uncertainty Loss ===
        L_u = self.uncertainty_loss(U, P_patch, gt_tracks, gt_visibility, query_times)

        # === Top-K Uncertainty ===
        L_topk_u = 0
        for k in range(K):
            L_topk_u += self.uncertainty_loss(U_top_k[:, :, :, k], P_patch_top_k[:, :, :, k], gt_tracks, gt_visibility, query_times)
        L_topk_u /= K

        L_topk_rank = self.region_rank_loss(S_top_k, P_patch_top_k, gt_tracks, gt_visibility, query_times)

        return L_p, L_p2, L_vis, L_o, L_u, L_topk_u, L_topk_rank


    def get_gt_offset(self, gt_tracks, stride, p_t):

        point_pred = torch.argmax(p_t, dim=-1)                            # (B, T, N)
        point_pred = indices_to_coords(point_pred, self.size, stride)     # (B, T, N, 2)
        gt_offsets = gt_tracks - point_pred                               # (B, T, N, 2)

        return gt_offsets
    
    def get_masks(self, gt_visibility, query_times):
        # :args gt_visibility: (B, T, N)
        # :args query_times: (B, N)
        # :args device: torch.device
        #
        # :return mask_point: (B * T * N)
        # :return mask_visible: (B * T * N)

        B, T, N = gt_visibility.shape
        device = gt_visibility.device

        # === Masks ===
        # visibility mask
        vis_mask = gt_visibility.float().reshape(-1)                                         # (B * T * N)

        # time mask
        time_indices = torch.arange(T).reshape(1, T, 1).to(device)                      # (1, T, 1)
        query_times_expanded = query_times.unsqueeze(1)                                 # (B, 1, N)
        time_mask = (time_indices > query_times_expanded).float()                      # (B, T, N)
        time_mask = time_mask.view(B * T * N)                                           # (B * T * N)

        mask_point = vis_mask * time_mask                                              # (B * T * N)
        mask_visible = time_mask

        return mask_point, mask_visible


    def patch_classification_loss(self, p_t, gt_tracks, gt_visibility, query_times):
        # :args P_t: (B, T, N, P_prime)
        # :args gt_tracks: (B, T, N, 2)
        # :args gt_visibility: (B, T, N)
        # :args query_times: (B, N)
        #
        # :return loss:

        B, T, N, _ = gt_tracks.shape

        mask_point, _ = self.get_masks(gt_visibility, query_times)

        gt_indices = coords_to_indices(gt_tracks, self.size, self.stride).view(-1)         # (B * T * N)

        p_t = p_t.reshape(B * T * N, -1)                                                           # (B * T * N, P)
        L_p = F.cross_entropy(p_t / self.temperature, gt_indices.long(), reduction="none", 
                              label_smoothing=self.epsilon_smoothing)                # (B * T * N)
        L_p = L_p * mask_point                                                       # (B * T * N)
        L_p = L_p.sum() / mask_point.sum()

        L_p *= self.lambda_patch_cls

        return L_p
    

    def visibility_loss(self, V_t, gt_visibility, query_times):
        # :args V_t: (B, T, N)
        # :args gt_visibility: (B, T, N)
        # :args query_times: (B, N)
        #
        # :return loss:

        B, T, N = gt_visibility.shape

        _, mask_visible = self.get_masks(gt_visibility, query_times)

        L_vis = F.binary_cross_entropy_with_logits(V_t.float(), gt_visibility.float(), reduction="none")    # (B, T, N)
        L_vis = L_vis.reshape(B * T * N)                                                   # (B * T * N)
        L_vis = L_vis * mask_visible                                                   # (B * T * N)
        L_vis = L_vis.sum() / mask_visible.sum()

        return L_vis
    

    def offset_loss(self, O_t, ref_point, gt_tracks, gt_visibility, query_times, coordinate_pt=False):
        # :args O_t: (B, T, N, 2)
        # :args P_t: (B, T, N, 2)
        # :args gt_tracks: (B, T, N, 2)
        # :args gt_visibility: (B, T, N)
        # :args query_times: (B, N)
        #
        # :return loss:

        B, T, N, _ = gt_tracks.shape

        mask_point, _ = self.get_masks(gt_visibility, query_times)

        gt_offset = gt_tracks - ref_point

        L_offset = F.l1_loss(O_t, gt_offset, reduction="none")   # (B, T, N, 2)
        L_offset = L_offset.sum(dim=-1).view(-1)                          # (B * T * N)
        L_offset = L_offset * mask_point                                  # (B * T * N)
        L_offset = torch.clamp(L_offset, min=0, max=2 * self.stride)
        L_offset = L_offset.sum() / mask_point.sum()

        return L_offset

    
    def uncertainty_loss(self, U_t, point_pred, gt_tracks, gt_visibility, query_times):
        # :args U_t: (B, T, N)
        # :args point_pred: (B, T, N, 2)
        # :args gt_tracks: (B, T, N, 2)
        # :args gt_visibility: (B, T, N)
        # :args query_times: (B, N)
        
        B, T, N, _ = gt_tracks.shape

        _, loss_mask = self.get_masks(gt_visibility, query_times)   # all points after query times
        delta = 12

        # L2 difference between point_pred and gt_tracks
        uncertainty_loss = F.mse_loss(point_pred, gt_tracks, reduction="none")        # (B, T, N, 2)
        uncertainty_loss = uncertainty_loss.sum(dim=-1).sqrt()                        # (B, T, N)

        # (uncertainty_loss > 8.0) or (~gt_visibility)
        uncertains = (uncertainty_loss > delta) | (~gt_visibility)     # (B, T, N)

        L_unc = F.binary_cross_entropy_with_logits(U_t.float(), 
                                                   uncertains.float(), 
                                                   reduction="none")     # (B, T, N)
        L_unc = L_unc.view(B * T * N)                                               # (B * T * N)
        L_unc = L_unc * loss_mask                                                   # (B * T * N)
        L_unc = L_unc.sum() / loss_mask.sum()

        return L_unc
    
    
    def region_rank_loss(self, S_top_k, P_patch_top_k, gt_tracks, gt_visibility, query_times):
        # :args S_top_k: (B, T, N, K)
        # :args P_patch_top_k: (B, T, N, K, 2)
        # :args gt_tracks: (B, T, N, 2)
        # :args gt_visibility: (B, T, N)
        # :args query_times: (B, N)

        B, T, N = gt_visibility.shape
        K = S_top_k.shape[-1]

        mask_point, _ = self.get_masks(gt_visibility, query_times)                 # (B * T * N)

        gt_expanded = gt_tracks.unsqueeze(-2).expand(-1, -1, -1, K, -1)  # (B, T, N, K, 2)
    
        distances = torch.norm(P_patch_top_k - gt_expanded, dim=-1)      # (B, T, N, K)        
        best_indices = distances.argmin(dim=-1)  # (B, T, N)

        # apply cross entropy loss
        best_indices = best_indices.view(B * T * N)
        S_top_k = S_top_k.reshape(B * T * N, K)

        L_rank = F.cross_entropy(S_top_k.float(), best_indices.long(), reduction="none")  # (B * T * N)
        L_rank = L_rank * mask_point
        L_rank = L_rank.sum() / mask_point.sum()
        return L_rank