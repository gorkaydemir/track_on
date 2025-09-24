import torch
import torch.nn.functional as F

import numpy as np
import cv2


def coords_to_indices(points, size, ps=4):
    # :arg points: (B, T, N, 2)
    # :arg H: int
    # :arg W: int
    # :arg ps: int, patch size
    #
    # :return indices: (B, T, N) in [0, H //ps * W // ps) range

    H, W = size
    P = (H // ps) * (W // ps)

    in_border_mask = (points[:, :, :, 0] >= 0) & (points[:, :, :, 0] < W) & (points[:, :, :, 1] >= 0) & (points[:, :, :, 1] < H)    # (B, T, N)

    scaled_points = torch.floor(points / ps).long()                             # (B, T, N, 2)
    indices = scaled_points[:, :, :, 1] * (W // ps) + scaled_points[:, :, :, 0] # (B, T, N)
    indices = indices * in_border_mask.long()                                  # (B, T, N)

    assert torch.all(indices < P), f"{indices.max()} >= {P}"
    assert torch.all(indices >= 0)

    # Each index explained with the upper left coordinate of the patch
    return indices


def indices_to_coords(indices, size, ps):
    # :arg indices: (B, T, N)
    # :arg H: int, original H of the input to the ViT
    # :arg W: int, original W of the input to the ViT
    # :arg ps: int, patch size
    #
    # :return coordinates: (B, T, N, 2) in [0, W] and [0, H] range

    # highest can be (H - 0.5, W - 0.5)

    B, T, N = indices.shape
    H, W = size

    num_columns = W // ps

    
    rows = indices // num_columns
    cols = indices % num_columns
    
    y_coords = rows * ps + 0.5 * ps
    x_coords = cols * ps + 0.5 * ps
    
    coordinates = torch.stack((x_coords, y_coords), dim=-1)

    assert coordinates.shape == (B, T, N, 2)
    assert torch.all(coordinates[:, :, :, 0] <= W)
    assert torch.all(coordinates[:, :, :, 1] <= H)
    
    return coordinates

def sample_grid_points(f, coords, L=7):
    # :arg f: (B, C, H, W)
    # :arg coords: (B, N, 2), in [-1, 1] range
    #
    # :return: (B, N, C, 2L + 1, 2L + 1)

    B, C, H, W = f.shape
    _, N, _    = coords.shape
    device, dtype = f.device, f.dtype

    if L != 0:
        half = (L - 1) // 2
        delta_x = torch.linspace(-half, half, L, device=device, dtype=dtype) * (2.0 / W)
        delta_y = torch.linspace(-half, half, L, device=device, dtype=dtype) * (2.0 / H)
        dy, dx = torch.meshgrid(delta_y, delta_x, indexing='ij')                 # (L, L)
        offsets = torch.stack((dx, dy), dim=-1)                                  # (L, L, 2)

        grid = coords[:, :, None, None, :] + offsets[None, None, ...]            # (B, N, L, L, 2)

        grid_flat = grid.view(B, N * L, L, 2)       # -> (B, N*L, L, 2)
        # now call grid_sample on (B, C, H, W) & (B, N*L, L, 2)
        sampled_flat = F.grid_sample(f, grid_flat, mode='bilinear',
                                    padding_mode='border',
                                    align_corners=False)
        # sampled_flat: (B, C, N*L, L)
        # reshape back to (B, N, C, L, L)
        sampled = (sampled_flat
                    .view(B, C, N, L, L)
                    .permute(0, 2, 1, 3, 4)
                    .contiguous())
        
    else:
        # just sample the single point per track
        # build a grid of shape (B, N, 1, 2)
        # then grid_sample returns (B, C, N, 1)
        grid = coords.view(B, N, 1, 2)
        sampled_pt = F.grid_sample(
            f, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # → (B, C, N, 1)
        # reorder to (B, N, C, 1, 1)
        sampled = sampled_pt.permute(0, 2, 1, 3) \
                            .unsqueeze(-1) \
                            .contiguous()

    return sampled


def get_queries(traj, vis, N_rand=None):
    # traj: (B, T, N, 2)
    # vis: (B, T, N)
    #
    # returns: (B, N, 3) where 3 is (frame_ind, x, y)

    # just before calling get_queries
    assert traj.size(2) > 0, "N==0 just entered the model"          # (1)
    assert (vis.sum(dim=1) != 0).all(), "Found a track that is never visible in any frame"

    B, T, N, D = traj.shape
    device = traj.device

    __, first_positive_inds = torch.max(vis, dim=1) # (B, N)
    
    if N_rand is None:
        N_rand = N // 4
    # inds of visible points in the 1st frame
    nonzero_inds = [[torch.nonzero(vis[b, :, i]) for i in range(N)] for b in range(B)]

    for b in range(B):
        rand_vis_inds = torch.cat(
            [
                nonzero_row[torch.randint(len(nonzero_row), size=(1,))]
                for nonzero_row in nonzero_inds[b]
            ],
            dim=1,
        )
        first_positive_inds[b] = torch.cat(
            [rand_vis_inds[:, :N_rand], first_positive_inds[b : b + 1, N_rand:]], dim=1
        )

    ind_array_ = torch.arange(T, device=device)
    ind_array_ = ind_array_[None, :, None].repeat(B, 1, N)
    assert torch.allclose(
        vis[ind_array_ == first_positive_inds[:, None, :]].float(),
        torch.ones(1, device=device),
    )
    gather = torch.gather(traj, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, D))
    xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)

    queries = torch.cat([first_positive_inds[:, :, None], xys[:, :, :2]], dim=2)    # (B, N, 3)

    return queries

# From https://github.com/facebookresearch/co-tracker/blob/9ed05317b794cd177674e681321780614a65e073/cotracker/models/core/model_utils.py#L20
def get_points_on_a_grid(size, extent, device):
    # :arg size: int
    # :arg extent: tuple of 2
    # :arg device: torch.device
    #
    # :return grid: (1, size ** 2, 2)


    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


