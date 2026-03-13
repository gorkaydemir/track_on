import torch
from model.trackon import Track_On2
import torch.nn.functional as F
from utils.coord_utils import get_points_on_a_grid
from utils.train_utils import load_pretrained_weights


class Predictor(torch.nn.Module):
    def __init__(self, model_args=None, checkpoint_path=None, support_grid_size=20):
        super().__init__()

        if model_args is None:
            model_args = self._default_args()

        self.model = Track_On2(model_args)

        if checkpoint_path is not None:
            load_pretrained_weights(self.model, checkpoint_path)

        # Optional inference-time memory extension
        ime_size = getattr(model_args, "M_i", 72)
        if ime_size != model_args.M:
            self.model.memory_extension(ime_size)

        self.model.eval()

        self.delta_v = model_args.delta_v
        self.support_grid_size = support_grid_size

        self.reset()

    def _default_args(self):
        class Args:
            def __init__(self):
                self.input_size = [384, 512]
                self.M = 24
                self.D = 256
                self.K = 16
                self.decoder_layer_num = 3
                self.predicton_head_layer_num = 3
                self.rerank_layer_num = 3
                self.vit_backbone = "dinov3_s_plus"
                self.vit_upsample_factor = 1.143
                self.grad_checkpoint = True

                self.M_i = 72       # Inference-time memory size (should be larger than training M for better performance)
                self.delta_v = 0.8  # Visibility threshold for inference

        return Args()

    def reset(self):
        """Reset all tracking state to start fresh."""
        self.t = 0
        self.point_memory = None
        self.temporal_mask = None
        self.q_init = None
        self.N = 0                      # Number of active queries
        self.capacity = 0               # Buffer capacity
        self.initial_capacity = 128     # Starting buffer size
        self.H = self.model.input_size[0]
        self.W = self.model.input_size[1]
        self.device = None

    def init_queries(self, frame_features, queries, H_in, W_in):
        """
        Initialize new queries and add them to the tracking state using pre-extracted features.
        Can be called multiple times to add queries at different frames.

        :args frame_features: Tuple of (f_fused_t, device) where f_fused_t is (1, P, D)
        :args queries: Tensor of shape (N_new, 2), where each query is (x, y), in pixel coordinates
        :args H_in: Original frame height
        :args W_in: Original frame width
        """
        if queries.shape[0] == 0:
            return  # No queries to add

        f_fused_t, device = frame_features

        N_new = queries.shape[0]

        # === Scale coordinates to [-1, 1] range ===
        query_coords_norm = queries.clone()
        query_coords_norm[:, 0] = query_coords_norm[:, 0] / W_in  # x
        query_coords_norm[:, 1] = query_coords_norm[:, 1] / H_in  # y
        query_coords_norm = query_coords_norm * 2 - 1  # to [-1, 1]
        # === === ===

        # === Sample query features (using pre-extracted features) ===
        f_b_t = f_fused_t.view(1, self.model.Hf, self.model.Wf, self.model.D)  # (1, H4, W4, D)
        f_b_t = f_b_t.permute(0, 3, 1, 2)  # (1, D, H4, W4)

        pts = query_coords_norm.unsqueeze(0).unsqueeze(2)  # (1, N_new, 1, 2)

        # Sample Points
        q_new = F.grid_sample(f_b_t, pts, mode='bilinear', padding_mode='border', align_corners=False)  # (1, D, N_new, 1)
        q_new = q_new.squeeze(-1).squeeze(0).permute(1, 0)  # (N_new, D)
        # === === ===

        # Initialize memory structures
        M = self.model.M
        D = self.model.D

        if self.q_init is None:
            # First initialization - allocate buffer with extra capacity
            self.capacity = max(self.initial_capacity, N_new)
            self.q_init = torch.zeros(self.capacity, D, device=device)
            self.point_memory = torch.zeros(self.capacity, M, D, device=device)
            self.temporal_mask = torch.ones(self.capacity, M, device=device, dtype=torch.bool)

            # Fill in the actual data
            self.q_init[:N_new] = q_new
            self.N = N_new
            self.device = device
        else:
            # Check if we need to expand the buffer
            if self.N + N_new > self.capacity:
                # Double the capacity (or more if needed)
                new_capacity = max(self.capacity * 2, self.N + N_new)

                # Allocate new buffers
                new_q_init = torch.zeros(new_capacity, D, device=device)
                new_point_memory = torch.zeros(new_capacity, M, D, device=device)
                new_temporal_mask = torch.ones(new_capacity, M, device=device, dtype=torch.bool)

                # Copy existing data
                new_q_init[:self.N] = self.q_init[:self.N]
                new_point_memory[:self.N] = self.point_memory[:self.N]
                new_temporal_mask[:self.N] = self.temporal_mask[:self.N]

                # Update references
                self.q_init = new_q_init
                self.point_memory = new_point_memory
                self.temporal_mask = new_temporal_mask
                self.capacity = new_capacity

            # Add new queries using in-place indexing (no concatenation needed)
            self.q_init[self.N:self.N + N_new] = q_new
            # point_memory and temporal_mask are already initialized with zeros/ones
            self.N += N_new

    @torch.no_grad()
    def forward_frame(self, frame, new_queries=None):
        """
        Track all currently active queries through one frame, optionally adding new queries.

        :args frame: Tensor of shape (1, 3, H, W) in range [0, 255]
        :args new_queries: Optional Tensor of shape (N_new, 2), where each query is (x, y), in pixel coordinates
        :return P_t: Tensor of shape (N, 2) - positions for all active queries
        :return V_t: Tensor of shape (N,) - visibility for all active queries
        """
        device = frame.device
        _, _, H, W = frame.shape

        # Extract features once
        f4_t, f8_t, f16_t, f32_t, f_fused_t = self.model.extract_frame_features(frame)

        # Add new queries if provided
        if new_queries is not None and new_queries.shape[0] > 0:
            self.init_queries((f_fused_t, device), new_queries, H, W)

        # If no queries exist after initialization, return empty
        if self.q_init is None or self.N == 0:
            return torch.empty(0, 2, device=device), torch.empty(0, dtype=torch.bool, device=device)

        # Track using pre-extracted features (use only active queries)
        frame_features = (f4_t, f8_t, f16_t, f32_t, f_fused_t)
        p, v_logit, q_new = self.model.track_frame(
            self.q_init[:self.N],
            self.temporal_mask[:self.N],
            self.point_memory[:self.N],
            frame_features,
            H,
            W
        )

        # Shift memory left and add new feature at the end (in-place using roll for efficiency)
        self.point_memory[:self.N] = torch.roll(self.point_memory[:self.N], shifts=-1, dims=1)
        self.point_memory[:self.N, -1] = q_new
        self.temporal_mask[:self.N] = torch.roll(self.temporal_mask[:self.N], shifts=-1, dims=1)
        self.temporal_mask[:self.N, -1] = False

        # Get visibility
        v_t = (v_logit.sigmoid() >= self.delta_v)  # (N,)

        self.t += 1

        # Clean up intermediate tensors before returning
        del frame_features, f4_t, f8_t, f16_t, f32_t, f_fused_t, v_logit, q_new

        return p, v_t

    @torch.no_grad()
    def forward(self, video, queries):
        """
        Track queries through a video, adding new queries as their start time arrives.

        :args video: Tensor of shape (1, T, 3, H, W)
        :args queries: Tensor of shape (1, N, 3), where each query is (t, x, y)
        :return pred_trajectory: Tensor of shape (1, T, N, 2)
        :return pred_visibility: Tensor of shape (1, T, N)
        """
        _, T, _, H, W = video.shape
        device = video.device

        queries = queries.squeeze(0)  # (N, 3)
        N_orig = queries.shape[0]
        query_times = queries[:, 0].long()  # (N,)
        query_coords = queries[:, 1:]  # (N, 2)

        # Add support grid if needed (support grid starts at t=0)
        if self.support_grid_size > 0:
            extra = get_points_on_a_grid(self.support_grid_size, (H, W), device)  # (1, S^2, 2)
            extra = extra.squeeze(0)  # (S^2, 2)
            extra_queries = torch.cat([torch.zeros(extra.shape[0], 1, device=device), extra], dim=1)  # (S^2, 3)
            queries = torch.cat([queries, extra_queries], dim=0)  # (N + S^2, 3)
            query_times = queries[:, 0].long()  # (N + S^2,)
            query_coords = queries[:, 1:]  # (N + S^2, 2)

        N_total = queries.shape[0]

        # Reset state
        self.reset()

        # Set capacity to accommodate all queries when the input is video
        self.initial_capacity = N_total

        # Allocate output tensors on CPU (only for original queries, not support grid)
        pred_trajectory = torch.zeros(1, T, N_orig, 2, device='cpu', dtype=torch.float32)
        pred_visibility = torch.zeros(1, T, N_orig, device='cpu', dtype=torch.bool)

        # Create a mapping from tracking order to original query indices
        tracking_to_original = []

        # Process each frame
        for t in range(T):
            frame = video[0, t].unsqueeze(0)  # (1, 3, H, W)

            # Check if any new queries start at this frame
            new_queries_mask = (query_times == t)  # (N_total,)
            new_queries_this_frame = None

            if new_queries_mask.any():
                # Get coordinates of new queries
                new_queries_this_frame = query_coords[new_queries_mask]  # (N_new, 2)
                new_indices = new_queries_mask.nonzero(as_tuple=True)[0]  # Original indices in the combined array

                # Update mapping: append the original indices in the order they were added
                tracking_to_original.extend(new_indices.tolist())

            # Track all queries (new queries will be automatically added inside forward_frame)
            if self.N > 0 or new_queries_this_frame is not None:
                p_t, v_t = self.forward_frame(frame, new_queries=new_queries_this_frame)  # (N_active, 2), (N_active,)

                if self.N > 0:  # Only store if we have active queries
                    # Vectorized assignment: create index tensors
                    tracking_to_original_tensor = torch.tensor(tracking_to_original, dtype=torch.long, device='cpu')

                    # Filter to only original queries (not support grid)
                    orig_mask = tracking_to_original_tensor < N_orig

                    if orig_mask.any():
                        # Get the original indices for queries that should be stored
                        orig_indices = tracking_to_original_tensor[orig_mask]

                        # Move predictions to CPU and store using advanced indexing
                        pred_trajectory[0, t, orig_indices] = p_t[orig_mask].cpu()
                        pred_visibility[0, t, orig_indices] = v_t[orig_mask].cpu()

                    # Clean up GPU tensors immediately after copying to CPU
                    del p_t, v_t, tracking_to_original_tensor, orig_mask
                    if new_queries_this_frame is not None:
                        del new_queries_this_frame

            # Clean up frame tensor
            del frame

        return pred_trajectory, pred_visibility
