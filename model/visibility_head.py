
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from model.modules import DMSMHA_Block, get_deformable_inputs


class Visibility_Head(nn.Module):
    def __init__(self, args, nhead=8):
        super().__init__()

        self.nhead = nhead
        self.size = args.input_size
        self.stride = args.stride

        self.linear_visibility = args.linear_visibility

        self.H_prime = self.size[0] // self.stride
        self.W_prime = self.size[1] // self.stride
        self.P = self.H_prime * self.W_prime

        self.embedding_dim = args.transformer_embedding_dim

        if self.linear_visibility:
            self.vis_layer = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(), nn.Linear(self.embedding_dim, 1))
            self.unc_layer = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(), nn.Linear(self.embedding_dim, 1))

        else:
            self.vis_layer = nn.Linear(2 * self.embedding_dim, 1)
            self.unc_layer = nn.Linear(2 * self.embedding_dim, 1)

            self.num_level = 4
            self.decoder_layer = DMSMHA_Block(self.embedding_dim, nhead, self.num_level)
    
    def forward(self, q_t, h_t, target_coordinates):
        # :args q_t: (B, N, C)
        # :args f_t: (B, P, C)
        # :args target_coordinates: (B, N, 2), in size range
        #
        # :return v_logit: (B, N)
        # :return u_logit: (B, N)

        B, N, C = q_t.shape
        device = q_t.device

        if self.linear_visibility:
            v_logit = self.vis_layer(q_t).squeeze(-1)                     # (B, N)
            u_logit = self.unc_layer(q_t).squeeze(-1)                     # (B, N)

        else:
            # <Target scaling>        
            target_coordinates = target_coordinates / torch.tensor([self.size[1], self.size[0]], device=device)  # (B, N, 2)
            target_coordinates = torch.clamp(target_coordinates, 0, 1)
            #   </Target scaling>

            f_scales, reference_points, spatial_shapes, start_levels = get_deformable_inputs(h_t, target_coordinates, self.H_prime, self.W_prime)

            q_t_local = self.decoder_layer(q=q_t, k=f_scales, v=f_scales, 
                                    reference_points=reference_points, 
                                    spatial_shapes=spatial_shapes, 
                                    start_levels=start_levels)
            
            q_t = torch.cat([q_t, q_t_local], dim=-1)  # (B, N, 2 * C)

            v_logit = self.vis_layer(q_t).squeeze(-1)                     # (B, N)
            u_logit = self.unc_layer(q_t).squeeze(-1)                     # (B, N)

        return v_logit, u_logit
