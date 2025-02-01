import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from utils.coord_utils import coords_to_indices, indices_to_coords

class Loss_Function(nn.Module):
    def __init__(self, args, tmp=0.05):
        super().__init__()

        self.gamma = 1.0

        self.size = args.input_size
        self.stride = args.stride

        self.lambda_point = args.lambda_point
        self.lambda_vis = args.lambda_vis
        self.lambda_offset = args.lambda_offset
        self.lambda_uncertainty = args.lambda_uncertainty
        self.lambda_top_k = args.lambda_top_k

        self.loss_after_query = args.loss_after_query


        self.temperature = tmp

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
        vis_mask = gt_visibility.float().view(-1)                                         # (B * T * N)

        # time mask
        if self.loss_after_query:
            time_indices = torch.arange(T).reshape(1, T, 1).to(device)                      # (1, T, 1)
            query_times_expanded = query_times.unsqueeze(1)                                 # (B, 1, N)
            time_mask = (time_indices > query_times_expanded).float()                      # (B, T, N)
            time_mask = time_mask.view(B * T * N)                                           # (B * T * N)
        else:
            time_mask = torch.ones(B * T * N, device=device).float()
        
        mask_point = vis_mask * time_mask                                              # (B * T * N)
        mask_visible = time_mask

        return mask_point, mask_visible


    def point_loss(self, p_t, gt_tracks, gt_visibility, query_times):
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
        L_p = F.cross_entropy(p_t / self.temperature, gt_indices.long(), reduction="none")  # (B * T * N)
        L_p = L_p * mask_point                                                       # (B * T * N)
        L_p = L_p.sum() / mask_point.sum()

        L_p *= self.lambda_point

        return L_p
    

    def visibility_loss(self, V_t, gt_visibility, query_times):
        # :args V_t: (B, T, N)
        # :args gt_visibility: (B, T, N)
        # :args query_times: (B, N)
        #
        # :return loss:

        B, T, N = gt_visibility.shape

        _, mask_visible = self.get_masks(gt_visibility, query_times)

        L_vis = F.binary_cross_entropy_with_logits(V_t, gt_visibility.float(), reduction="none")    # (B, T, N)
        L_vis = L_vis.view(B * T * N)                                                   # (B * T * N)
        L_vis = L_vis * mask_visible                                                   # (B * T * N)
        L_vis = L_vis.sum() / mask_visible.sum()

        L_vis *= self.lambda_vis

        return L_vis
    

    def offset_loss(self, O_t, ref_point, gt_tracks, gt_visibility, query_times, coordinate_pt=False):
        # :args O_t: (B, T, #offset_layers, N, 2)
        # :args P_t: (B, T, N, 2)
        # :args gt_tracks: (B, T, N, 2)
        # :args gt_visibility: (B, T, N)
        # :args query_times: (B, N)
        #
        # :return loss:

        B, T, N, _ = gt_tracks.shape

        mask_point, _ = self.get_masks(gt_visibility, query_times)

        offset_layer_num = O_t.size(2)

        gt_offset = gt_tracks - ref_point

        cnt_offset = 0
        L_offset = 0
        for l in range(offset_layer_num):

            o_l = O_t[:, :, l]                                          # (B, T, N, 2)
            offset_loss = F.l1_loss(o_l, gt_offset, reduction="none")   # (B, T, N, 2)
            offset_loss = offset_loss.sum(dim=-1).view(-1)                          # (B * T * N)
            offset_loss = offset_loss * mask_point                                  # (B * T * N)

            offset_loss = torch.clamp(offset_loss, min=0, max=2 * self.stride)

            offset_loss = offset_loss.sum() / mask_point.sum()
            offset_loss *= (self.gamma ** (offset_layer_num - l - 1))
            cnt_offset += 1

            L_offset += offset_loss

        L_offset *= self.lambda_offset
        L_offset /= cnt_offset

        return L_offset

    
    def uncertainty_loss(self, U_t, point_pred, gt_tracks, gt_visibility, query_times):
        # :args U_t: (B, T, N)
        # :args point_pred: (B, T, N, 2)
        # :args gt_tracks: (B, T, N, 2)
        # :args gt_visibility: (B, T, N)
        # :args query_times: (B, N)

        B, T, N, _ = gt_tracks.shape

        _, mask_visible = self.get_masks(gt_visibility, query_times)                 # (B * T * N)

        # L2 difference between point_pred and gt_tracks
        uncertainty_loss = F.mse_loss(point_pred, gt_tracks, reduction="none")        # (B, T, N, 2)
        uncertainty_loss = uncertainty_loss.sum(dim=-1).sqrt()                        # (B, T, N)

        # (uncertainty_loss > 8.0) or (~gt_visibility)
        uncertains = (uncertainty_loss > 8.0) | (~gt_visibility)      # (B, T, N)

        # uncertains = (uncertainty_loss < 8.0).float() * (~gt_visibility).float()      # (B, T, N)

        L_unc = F.binary_cross_entropy_with_logits(U_t, uncertains.float(), reduction="none")        # (B, T, N)
        L_unc = L_unc.view(B * T * N)                                                   # (B * T * N)
        L_unc = L_unc * mask_visible                                                   # (B * T * N)
        L_unc = L_unc.sum() / mask_visible.sum()

        return L_unc