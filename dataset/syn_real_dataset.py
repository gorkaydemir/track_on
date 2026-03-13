import torch
import numpy as np
from torch.utils.data import Dataset

from utils.coord_utils import get_queries


class SynRealDataset(Dataset):
    """
    A meta-dataset that combines synthetic (Movi-F) and real-world samples.

    Each item returned is a 6-tuple:
        video_aug    : (T, 3, H, W) float32  – photometrically augmented video
        video_clean  : (T, 3, H, W) float32  – crop-only video (no photo aug)
        queries      : (N, 3) float32         – query points in (t, x, y) format
        tracks_gt    : (T, N, 2) float32      – ground-truth trajectories (x, y);
                                                zeros for real-world samples
        visibility_gt: (T, N) bool            – ground-truth visibility;
                                                zeros for real-world samples
        is_synthetic : scalar bool tensor     – True when the sample is from Movi-F

    Indices [0, real_len)              → real-world samples
    Indices [real_len, real_len+syn_len) → synthetic (Movi-F) samples

    The DistributedSampler will shuffle the combined index space so both sources
    are interleaved throughout training.
    """

    def __init__(self, real_dataset, syn_dataset=None):
        self.real_dataset = real_dataset
        self.syn_dataset  = syn_dataset
        self.real_len = len(real_dataset)
        self.syn_len  = 0 if syn_dataset is None else len(syn_dataset)

    def __len__(self):
        return self.real_len + self.syn_len

    def __getitem__(self, idx):
        # ── Real-world sample ──────────────────────────────────────────────────
        if idx < self.real_len:
            video_aug_np, video_clean_np, queries_np = self.real_dataset[idx]

            video_aug   = torch.from_numpy(np.ascontiguousarray(video_aug_np))    # (T, 3, H, W)
            video_clean = torch.from_numpy(np.ascontiguousarray(video_clean_np))  # (T, 3, H, W)
            queries     = torch.from_numpy(np.ascontiguousarray(queries_np))      # (N, 3)

            T = video_aug.shape[0]
            N = queries.shape[0]

            tracks_gt    = torch.zeros(T, N, 2, dtype=torch.float32)
            visibility_gt = torch.zeros(T, N, dtype=torch.bool)
            is_synthetic  = torch.tensor(False)

            return video_aug, video_clean, queries, tracks_gt, visibility_gt, is_synthetic

        # ── Synthetic (Movi-F) sample ──────────────────────────────────────────
        else:
            syn_idx = idx - self.real_len
            rgbs, trajs, visibles, _ = self.syn_dataset[syn_idx]
            # rgbs:     (T, 3, H, W) float32 tensor  (already augmented by Movi-F pipeline)
            # trajs:    (T, N, 2) float32 tensor  – x, y pixel coords at crop_size
            # visibles: (T, N) bool tensor

            # get_queries expects (B, T, N, 2) / (B, T, N); add/remove batch dim
            queries = get_queries(trajs.unsqueeze(0), visibles.unsqueeze(0)).squeeze(0)  # (N, 3): frame_ind, x, y
            is_synthetic = torch.tensor(True)

            # Use the same augmented frames for both video_aug and video_clean;
            # ensemble is not run on synthetic samples, so video_clean is unused.
            return rgbs, rgbs.clone(), queries, trajs, visibles.bool(), is_synthetic
