import torch
import torch.nn as nn
import torch.nn.functional as F
from ensemble.alltracker.alltracker import Net


class AllTrackerPredictor(nn.Module):
    def __init__(self, checkpoint_path=None, seqlen=16, inference_iters=4):
        super().__init__()
        
        self.model = Net(seqlen)
        self.inference_iters = inference_iters
        self.seqlen = seqlen
        self.interp_shape = (384, 512)  # Model expects this resolution
        
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in state_dict:
                self.model.load_state_dict(state_dict['model'], strict=True)
            else:
                self.model.load_state_dict(state_dict, strict=True)
        else:
            # Load from HuggingFace Hub
            url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
            self.model.load_state_dict(state_dict['model'], strict=True)
        
        self.model.eval()

    def forward(self, rgbs, queries=None):
        """
        Forward pass for AllTracker predictor.
        
        Args:
            rgbs: (B, T, C, H, W) tensor in [0, 255] range
            queries: (B, N, 3) tensor where each query is [time, x, y]
        
        Returns:
            tracks: (B, T, N, 2) predicted point locations
            visibility: (B, T, N) visibility mask
        """
        B, T, C, H, W = rgbs.shape
        device = rgbs.device
        
        assert queries is not None, "AllTracker requires queries"
        
        # Resize video to model's expected resolution
        rgbs_resized = rgbs.reshape(B * T, C, H, W)
        rgbs_resized = F.interpolate(rgbs_resized, self.interp_shape, mode="bilinear")
        rgbs_resized = rgbs_resized.reshape(B, T, C, self.interp_shape[0], self.interp_shape[1])
        
        H_resized, W_resized = self.interp_shape
        
        queries = queries.clone().float()
        B_q, N, D = queries.shape
        assert D == 3
        assert B_q == 1
        
        # Scale query coordinates to resized resolution
        queries_resized = queries.clone()
        queries_resized[:, :, 1] *= W_resized / W  # x coordinate
        queries_resized[:, :, 2] *= H_resized / H  # y coordinate
        
        # Create coordinate grid for converting flow to trajectories at resized resolution
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H_resized, device=device, dtype=torch.float32),
            torch.arange(W_resized, device=device, dtype=torch.float32),
            indexing='ij'
        )
        grid_xy = torch.stack([x_grid, y_grid], dim=0)[None, None]  # 1,1,2,H,W
        
        # Initialize output tensors
        trajs_e = torch.zeros([B, T, N, 2], device=device)
        visibility = torch.zeros([B, T, N], device=device)
        
        # Get unique query start times
        first_positive_inds = queries_resized[0, :, 0].long()
        
        # Process each unique start time
        for first_positive_ind in torch.unique(first_positive_inds):
            # Get points that start at this time
            chunk_pt_idxs = torch.nonzero(first_positive_inds == first_positive_ind, as_tuple=False)[:, 0]
            chunk_pts = queries_resized[:, chunk_pt_idxs, 1:]  # B, K, 2 (x, y coordinates at resized resolution)
            
            # Initialize trajectory maps at resized resolution
            traj_maps_e = grid_xy.repeat(1, T, 1, 1, 1)  # B,T,2,H_resized,W_resized
            visconf_maps_e = torch.zeros_like(traj_maps_e)
            
            if first_positive_ind < T - 1:
                # Run AllTracker from the query start time forward
                video_chunk = rgbs_resized[:, first_positive_ind:]
                T_chunk = video_chunk.shape[1]
                
                # Use forward_sliding for long videos, regular forward for short ones
                if T_chunk > 128:
                    forward_flow_e, forward_visconf_e, _, _ = self.model.forward_sliding(
                        video_chunk, 
                        iters=self.inference_iters, 
                        sw=None, 
                        is_training=False
                    )
                else:
                    forward_flow_e, forward_visconf_e, _, _ = self.model(
                        video_chunk, 
                        iters=self.inference_iters, 
                        sw=None, 
                        is_training=False
                    )
                
                # Convert flow to absolute trajectories
                forward_flow_e = forward_flow_e.to(device)
                forward_visconf_e = forward_visconf_e.to(device)
                
                # Handle different output shapes (flow might not have T dimension if T=2)
                if forward_flow_e.dim() == 4:  # B,2,H,W
                    forward_flow_e = forward_flow_e.unsqueeze(1)  # B,1,2,H,W
                    forward_visconf_e = forward_visconf_e.unsqueeze(1)
                
                forward_traj_maps_e = forward_flow_e + grid_xy  # B,T_chunk,2,H_resized,W_resized
                traj_maps_e[:, first_positive_ind:first_positive_ind + T_chunk] = forward_traj_maps_e
                visconf_maps_e[:, first_positive_ind:first_positive_ind + T_chunk] = forward_visconf_e
            
            # Sample trajectories at query points (at resized resolution)
            xyt = chunk_pts[0].round().long()  # K,2 (x, y)
            xyt[:, 0] = torch.clamp(xyt[:, 0], 0, W_resized - 1)
            xyt[:, 1] = torch.clamp(xyt[:, 1], 0, H_resized - 1)
            
            # Sample from maps: B,T,2,H,W -> B,T,2,K -> B,T,K,2
            trajs_e_chunk = traj_maps_e[:, :, :, xyt[:, 1], xyt[:, 0]].permute(0, 1, 3, 2)
            visconfs_e_chunk = visconf_maps_e[:, :, :, xyt[:, 1], xyt[:, 0]].permute(0, 1, 3, 2)
            
            # Scale trajectories back to original resolution
            trajs_e_chunk_scaled = trajs_e_chunk.clone()
            trajs_e_chunk_scaled[..., 0] *= W / W_resized  # x coordinate
            trajs_e_chunk_scaled[..., 1] *= H / H_resized  # y coordinate
            
            # Store in output tensor
            trajs_e[:, :, chunk_pt_idxs] = trajs_e_chunk_scaled
            
            # Compute visibility from visconf (product of two channels)
            vis_chunk = visconfs_e_chunk[..., 0] * visconfs_e_chunk[..., 1]  # B,T,K
            visibility[:, :, chunk_pt_idxs] = vis_chunk
        
        # Apply visibility threshold
        vis_thr = 0.6
        visibility = (visibility > vis_thr).float()
        
        # Rearrange to match output format: (B, T, N, 2) and (B, T, N)
        return trajs_e, visibility
