import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random as py_random
from concurrent.futures import ThreadPoolExecutor


# === Ensemble Utils ===
def select_best_predictions_and_indices(predictions, gt_trajectories):
    """
    Selects the closest prediction to the GT trajectory for each frame and returns both the best indices and best predictions.
    
    Args:
        predictions (list of torch.Tensor): Shape (B, T, N, M, 2)
        gt_trajectories (torch.Tensor): Shape (B, T, N, 2)
        gt_visibility (torch.Tensor): Shape (B, T, N), 1 for visible, 0 for occluded
    
    Returns:
        tuple:
            - torch.Tensor: Best model indices of shape (B, T, N), containing values in [0, M-1]
            - torch.Tensor: Best predictions of shape (B, T, N, 2)
    """
    # Stack predictions along a new dimension to form (B, T, N, M, 2)
    B, T, N, M, _ = predictions.shape  # Extract batch, time, number of objects, ensemble size, and spatial dims
    
    # Expand GT trajectories to match the ensemble dimension
    gt_expanded = gt_trajectories.unsqueeze(-2).expand(-1, -1, -1, M, -1)  # (B, T, N, M, 2)
    
    # Compute Euclidean distance between GT and each ensemble prediction
    distances = torch.norm(predictions - gt_expanded, dim=-1)  # (B, T, N, M)
    
    # Find the index of the closest prediction in each frame
    best_indices = distances.argmin(dim=-1)  # (B, T, N)
    
    # Select the best predictions based on the indices
    best_predictions = torch.gather(predictions, dim=-2, index=best_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2))
    
    # Remove the ensemble dimension (M=1 now after selection)
    best_predictions = best_predictions.squeeze(-2)  # (B, T, N, 2)

    assert (best_indices >= 0).all() and (best_indices < M).all()
    
    return best_indices, best_predictions


def get_ensemble(predictions, verifier_output, max=True):
    """
    Selects the most probable prediction from the ensemble using the verifier output.
    
    Args:
        predictions (torch.Tensor): Shape (B, T, N, M, 2), stack of predictions
        verifier_output (torch.Tensor): Shape (B, T, N, M), probability distribution over ensemble members
    
    Returns:
        torch.Tensor: Selected predictions of shape (B, T, N, 2)
    """
    # Get the most probable index from the verifier output
    if max:
        best_indices = verifier_output.argmax(dim=-1, keepdim=True)
    else:
        best_indices = verifier_output.argmin(dim=-1, keepdim=True)
    
    # Select predictions based on the most probable index
    selected_predictions = torch.gather(predictions, dim=-2, index=best_indices.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
    
    # Remove the ensemble dimension
    selected_predictions = selected_predictions.squeeze(-2)  # (B, T, N, 2)
    
    return selected_predictions





def select_diverse_tracks(preds: torch.Tensor,
                          validity: torch.Tensor,
                          N_top: int,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    Selects the N_top point indices (per batch) whose M tracker predictions are most diverse.
    
    Args:
        preds:     (B, T, N, M, 2) predicted points.
        validity:  (B, T, N) mask in {0,1} (or bool) for whether point n is valid at time t.
        N_top:     number of indices to return per batch.
        eps:       small value to avoid division-by-zero.
        
    Returns:
        top_idx:   (B, N_top) long tensor of point indices, sorted by descending diversity.
    """
    B, T, N, M, _ = preds.shape
    assert validity.shape == (B, T, N), "validity must be (B, T, N)"

    # Mean across M trackers at each (b, t, n)
    mean_pred = preds.mean(dim=3, keepdim=True)                      # (B, T, N, 1, 2)

    # Deviation from mean for each tracker (L2)
    diffs = preds - mean_pred                                        # (B, T, N, M, 2)
    dev = torch.norm(diffs, dim=-1)                                  # (B, T, N, M)

    # Average deviation across trackers -> per (b, t, n)
    dev_mt = dev.mean(dim=3)                                         # (B, T, N)

    # Validity-weighted mean across time
    v = validity.to(dev_mt.dtype)                                    # (B, T, N)
    num = (dev_mt * v).sum(dim=1)                                    # (B, N)
    den = v.sum(dim=1)                                               # (B, N)

    # If a track is never valid across T, set its score to -inf so it's never selected
    diversity_score = torch.where(den > 0, num / (den + eps), torch.full_like(num, float("-inf")))

    # Top-K indices per batch
    top_idx = torch.topk(diversity_score, k=N_top, dim=1, largest=True, sorted=True).indices  # (B, N_top)
    return top_idx



def psi_perturbation(P, visibility, M):

    B, T, N, _ = P.shape
    device = P.device
    P_perturbed = torch.zeros(B, T, N, M, 2, device=device, dtype=P.dtype)

    error_level = np.random.choice(["normal", "high", "low"], p=[0.5, 0.25, 0.25])

    def px(val):
        return val

    PIXEL_NOISE = {
        'stable_clean': 2.0,
        'stable_with_spikes': 4.0,
        'mild_drift': 8.0,
        'drifty': 16.0,
        'catastrophic_jump': 32.0,
        'catastrophic_constant': 2.0,
        'catastrophic_growing': 64.0,
        'spiky': 16.0,
        'small_noise': 4.0,
        'global_noise': 1.0,
    }

    error_distribution = {
        'drifty': 0.30,                     # Group B - Gradual drift
        'mild_drift': 0.40,                 # Group B - Gradual drift
        'stable_clean': 0.50,               # Group A - Stable noise
        'stable_with_spikes': 0.40,         # Group A - Stable noise
        'catastrophic_constant': 0.10,      # Group C - Complete switch
        'spiky': 0.30,                      # Group A - Spiky noise
        'catastrophic_jump': 0.10,          # Group C - Abrupt jump
        'catastrophic_growing': 0.10        # Group B - Long-term drift
    }
    
    if error_level == "normal":
        pass  # use default
    elif error_level == "high":
        # increae prob to 1.3
        error_distribution = {k: v * 1.3 for k, v in error_distribution.items()}
    elif error_level == "low":
        # decrease prob to 0.7
        error_distribution = {k: v * 0.7 for k, v in error_distribution.items()}
    else:
        raise ValueError(f"Unknown error_level: {error_level}")



    switch_params = {
        'drifty': (1.0, px(32), px(PIXEL_NOISE['drifty'])),
        'mild_drift': (1.0, px(16), px(PIXEL_NOISE['mild_drift'])),
        'catastrophic_jump': (1.0, px(128), px(PIXEL_NOISE['catastrophic_jump'])),
    }

    def smooth_trajectory(x, kernel_size=3):
        B, T, N, D = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N * D, 1, T)
        kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
        x_padded = F.pad(x, (kernel_size // 2, kernel_size // 2), mode='replicate')
        x_smooth = F.conv1d(x_padded, kernel)
        return x_smooth.view(B, N, D, T).permute(0, 3, 1, 2)

    def apply_occlusion_triggered_switch_global(P_full, V_full, selected_indices, prob, d_thresh, noise_std, fade_interp=True):
        B, T, N, _ = P_full.shape
        P_switched = P_full.clone()
        for n in selected_indices:
            vis = V_full[0, :, n]
            occluded_times = (~vis).nonzero(as_tuple=False).squeeze(1)
            if occluded_times.numel() == 0:
                continue
            t_occ = occluded_times[torch.randint(0, occluded_times.numel(), (1,)).item()].item()
            vis_at_t = V_full[0, t_occ].clone()
            vis_at_t[n] = False
            if not vis_at_t.any():
                continue
            p_n = P_full[0, t_occ, n]
            p_others = P_full[0, t_occ, vis_at_t]
            dists = torch.norm(p_others - p_n[None], dim=-1)
            close_mask = dists < d_thresh
            if not close_mask.any() or torch.rand(1).item() > prob:
                continue
            idx_candidates = vis_at_t.nonzero(as_tuple=False).squeeze(1)[close_mask]
            target_j = idx_candidates[torch.randint(0, len(idx_candidates), (1,)).item()]
            original = P_full[0, t_occ:, n]
            switched = P_full[0, t_occ:, target_j]
            if fade_interp:
                alpha = torch.linspace(0, 1, switched.shape[0], device=P_full.device).view(-1, 1)
                P_switched[0, t_occ:, n] = (1 - alpha) * original + alpha * switched
            else:
                P_switched[0, t_occ:, n] = switched
            if t_occ < T - 1:
                P_switched[0, t_occ+1:, n] += torch.randn(T - t_occ - 1, 2, device=P_full.device) * noise_std
        return P_switched

    def sample_based_id_switch_vectorized(P_sub, V_sub, switch_prob, d_thresh, num_time_samples, noise_std):
        B, T, N_sub, _ = P_sub.shape
        out = P_sub.clone()
        for _ in range(num_time_samples):
            t = torch.randint(0, T, (1,)).item()
            visible = V_sub[:, t]
            mask = visible.unsqueeze(2) & visible.unsqueeze(1)
            dist = torch.norm(P_sub[:, t].unsqueeze(2) - P_sub[:, t].unsqueeze(1), dim=-1)
            eye = torch.eye(N_sub, device=P.device).bool().unsqueeze(0)
            dist.masked_fill_(eye | ~mask, float('inf'))
            can_switch = (torch.rand_like(dist) < switch_prob) & (dist < d_thresh)
            for b in range(B):
                i_idx, j_idx = torch.where(can_switch[b])
                used = set()
                for i, j in zip(i_idx.tolist(), j_idx.tolist()):
                    if i in used or j in used: continue
                    used.update([i, j])
                    tmp = out[b, t:, i].clone()
                    out[b, t:, i] = out[b, t:, j]
                    out[b, t:, j] = tmp
                    if t < T - 1:
                        out[b, t+1:, i] += torch.randn(T - t - 1, 2, device=P.device) * noise_std
                        out[b, t+1:, j] += torch.randn(T - t - 1, 2, device=P.device) * noise_std
        return out

    def perturb(P_full, V_full, indices, err_type):
        P_sub = P_full[:, :, indices]
        V_sub = V_full[:, :, indices]
        std = px(PIXEL_NOISE[err_type])
        if err_type == 'stable_clean':
            return smooth_trajectory(P_sub + torch.cumsum(torch.randn_like(P_sub) * std, dim=1), kernel_size=5)
        elif err_type == 'stable_with_spikes':
            out = P_sub + torch.cumsum(torch.randn_like(P_sub) * std, dim=1)
            spike_t = torch.randint(1, P_sub.shape[1] - 1, (2,))
            out[:, spike_t] += torch.randn_like(out[:, spike_t]) * px(8.0)
            return smooth_trajectory(out, kernel_size=3)
        elif err_type == 'catastrophic_constant':
            rand_idx = torch.randint(0, P_full.shape[2], (B, len(indices)), device=P.device)
            return torch.stack([P_full[b, :, rand_idx[b]] for b in range(B)], dim=0)
        elif err_type == 'catastrophic_growing':
            direction = torch.randn(P_sub.shape[0], 1, 1, 2, device=P.device)
            scale = torch.linspace(0, 1, P_sub.shape[1], device=P.device).view(1, -1, 1, 1)
            return smooth_trajectory(P_sub + direction * scale * std, kernel_size=5)
        elif err_type in switch_params:
            prob, d_thresh, noise_std = switch_params[err_type]
            return apply_occlusion_triggered_switch_global(P_full, V_full, indices, prob, d_thresh, noise_std, fade_interp=True)[:, :, indices]
        elif err_type == 'spiky':
            out = sample_based_id_switch_vectorized(P_sub, V_sub, 0.6, px(24), 3, std)
            return out + torch.randn_like(out) * px(2.0)
        else:
            raise ValueError(f"Unknown error type: {err_type}")

    for m in range(M):
        for b in range(B):
            P_chain = P[b:b+1].clone()  # (1, T, N, 2)
            V_chain = visibility[b:b+1]  # (1, T, N)

            P_chain += torch.randn_like(P_chain) * px(PIXEL_NOISE['global_noise'])

            for err_type, prob in error_distribution.items():
                if torch.rand(1).item() < prob:
                    P_chain = perturb(P_chain, V_chain, list(range(N)), err_type)            

            P_perturbed[b, :, :, m] = P_chain[0]

    return P_perturbed