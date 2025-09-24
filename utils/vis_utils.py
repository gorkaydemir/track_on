import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2


# This file is adapted from TAPNet and CoTracker visualization utilities.
# For reference, see: https://github.com/google-deepmind/tapnet/blob/main/tapnet/utils/viz_utils.py
#                     https://github.com/facebookresearch/co-tracker/blob/main/cotracker/utils/visualizer.py


# ----------------------------
# Homography utilities (NumPy)
# ----------------------------

def _normalize_points(pts):
    """Hartley normalization: shift to zero-mean and scale so mean dist = sqrt(2)."""
    pts = np.asarray(pts, dtype=np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    mean_dist = np.mean(np.sqrt(np.sum(centered**2, axis=1))) + 1e-12
    s = np.sqrt(2.0) / mean_dist
    T = np.array([[s, 0, -s*mean[0]],
                  [0, s, -s*mean[1]],
                  [0, 0, 1]], dtype=np.float64)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T  # (3,N)
    npts_h = (T @ pts_h).T[:, :2]
    return npts_h, T

def estimate_homography(targ_pts, src_pts, mask=None):
    """Robust DLT with normalization; returns 3x3 H mapping src->targ."""
    targ_pts = np.asarray(targ_pts, dtype=np.float64)
    src_pts  = np.asarray(src_pts,  dtype=np.float64)

    if mask is None:
        mask = np.ones((targ_pts.shape[0],), dtype=np.float64)
    else:
        mask = np.asarray(mask, dtype=np.float64)

    # keep only masked rows
    keep = mask > 0.5
    targ = targ_pts[keep]
    src  = src_pts[keep]

    # Need at least 4 correspondences
    if src.shape[0] < 4:
        return np.eye(3, dtype=np.float64)

    # Normalize points
    src_n,  T_src  = _normalize_points(src)
    targ_n, T_targ = _normalize_points(targ)

    x, y = src_n[:, 0],  src_n[:, 1]
    u, v = targ_n[:, 0], targ_n[:, 1]
    one  = np.ones_like(x)
    zero = np.zeros_like(x)

    A1 = np.stack([x, y, one,  zero, zero, zero, -u*x, -u*y, -u], axis=1)
    A2 = np.stack([zero, zero, zero, x, y, one,  -v*x, -v*y, -v], axis=1)
    A  = np.concatenate([A1, A2], axis=0)

    # SVD
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    h = Vh[-1, :]
    Hn = h.reshape(3, 3)

    # Denormalize: targ = H * src  =>  H = T_targ^{-1} * Hn * T_src
    H = np.linalg.inv(T_targ) @ Hn @ T_src

    # Normalize scale (optional)
    if np.abs(H[2,2]) > 1e-12:
        H /= H[2,2]

    # Validity checks
    if not np.all(np.isfinite(H)):
        return np.eye(3, dtype=np.float64)
    # Very ill-conditioned? (heuristic)
    try:
        cond = np.linalg.cond(H)
        if cond > 1e12:
            return np.eye(3, dtype=np.float64)
    except Exception:
        pass

    return H


def compute_canonical_points(
    all_tformed, occ, err, inner_thresh, outer_thresh, required_inlier_frac, rng=None
):
    """
    NumPy version of canonical point computation.

    Args:
      all_tformed: (T, N, C) transformed points per frame (T), point (N), coords (C=2)
      occ:         (T, N) boolean or {0,1} occlusion mask (True/1 = occluded)
      err:         (T, N) reprojection error per frame/point
      inner_thresh: float, inlier threshold (squared-distance units consistent with `err`)
      outer_thresh: float, definite outlier threshold
      required_inlier_frac: float in [0,1], minimum fraction of visible frames that must be inliers
      rng:         optional np.random.Generator

    Returns:
      canonical_pts:     (N, C) canonical locations
      canonical_invalid: (N,) boolean mask for invalid canonical points
    """
    all_tformed = np.asarray(all_tformed, dtype=np.float64)  # (T, N, C)
    occ = np.asarray(occ)                                    # (T, N)
    err = np.asarray(err, dtype=np.float64)                  # (T, N)

    if occ.dtype != bool:
        occ = occ.astype(bool)

    T, N, C = all_tformed.shape

    definite_outliers = np.logical_or(occ, err > outer_thresh)          # (T, N)
    maybe_inliers     = np.logical_and(~occ, err < inner_thresh)        # (T, N)

    sum_inliers = np.sum(maybe_inliers, axis=0)                         # (N,)
    sum_vis     = np.sum(~occ, axis=0)                                  # (N,)
    frac_inliers = sum_inliers / np.maximum(1.0, sum_vis)               # (N,)

    canonical_invalid = frac_inliers < required_inlier_frac             # (N,)

    # Average over non-outlier frames (uniform weights over frames kept)
    keep_mask = (~definite_outliers).astype(np.float64)                 # (T, N)
    # einsum over time with mask: sum_t (all_tformed[t, n, :] * keep_mask[t, n])
    num = np.einsum('tnc,tn->nc', all_tformed, keep_mask)               # (N, C)
    den = np.maximum(1.0, np.sum(keep_mask, axis=0))[:, None]           # (N, 1)
    canonical_pts = num / den                                           # (N, C)

    # Re-initialize invalid canonical points with a random visible sample (defaults to t=0 if none visible)
    rng = np.random.default_rng() if rng is None else rng
    vis = (~occ).astype(np.int32)                                       # (T, N) in {0,1}
    sum_vis_int = vis.sum(axis=0)                                       # (N,)

    # Choose a random visible index per point: r_n in [0, sum_vis_n), else 0 if sum_vis_n=0
    random_choice = (rng.random(N) * sum_vis_int).astype(np.int32)      # (N,)

    # Build per-frame running counts for visible frames;
    # ids[t, n] = 1..k at visible frames, -1 at occluded frames
    cumsum_vis = np.cumsum(vis, axis=0)                                  # (T, N)
    ids = cumsum_vis * vis - occ.astype(np.int32)                        # (T, N); visible: 1..k, occluded: -1

    # Pick the frame where ids == random_choice (i.e., the r_n-th visible frame)
    idx_mask = (ids == random_choice[None, :])                           # (T, N) boolean
    # Convert mask to selected time index per point (defaults to 0 if no True)
    selected_t = (idx_mask * np.arange(T, dtype=np.int32)[:, None]).sum(axis=0)  # (N,)

    # Gather random points at selected_t for each point n
    gather_idx = selected_t[None, :, None]                               # (1, N, 1) for take_along_axis
    random_pts = np.take_along_axis(all_tformed, gather_idx, axis=0)[0]  # (N, C)

    # Replace invalid canonical points with these random visible points
    canonical_pts = np.where(canonical_invalid[:, None], random_pts, canonical_pts)

    return canonical_pts, canonical_invalid

def compute_inliers(homog, thresh, targ_pts=None, src_pts=None, src_pts_homog=None):
    """Compute inliers and errors (NumPy)."""
    homog = np.asarray(homog, dtype=np.float64)

    if src_pts_homog is None:
        src_pts = np.asarray(src_pts, dtype=np.float64)
        ones = np.ones((src_pts.shape[0], 1), dtype=src_pts.dtype)
        src_pts_homog = np.concatenate([src_pts, ones], axis=-1).T  # (3, N)
    else:
        src_pts_homog = np.asarray(src_pts_homog, dtype=np.float64)

    targ_pts = np.asarray(targ_pts, dtype=np.float64)

    tformed = (homog @ src_pts_homog).T  # (N, 3)
    w = tformed[:, 2:3]
    denom = np.maximum(1e-12, np.abs(w)) * np.sign(w)
    tformed_xy = tformed[:, :2] / denom

    err = np.sum((targ_pts - tformed_xy) ** 2, axis=-1)
    new_inliers = err < (thresh * thresh)
    return new_inliers, err, tformed_xy


def ransac_homography(targ_pts, src_pts, vis, thresh=4.0, targ_inlier_frac=0.5, rng=None):
    """Run RANSAC for homography with NumPy, using visibility as sampling probs."""
    targ_pts = np.asarray(targ_pts, dtype=np.float64)
    src_pts  = np.asarray(src_pts,  dtype=np.float64)
    vis      = np.asarray(vis,      dtype=np.float64)

    probs = vis / (vis.sum() + 1e-12)
    N = targ_pts.shape[0]

    # We'll attempt up to N trials (like the JAX version generated N draws),
    # stopping early if we exceed a decaying inlier threshold.
    n_trials = max(1, N)
    best_inliers_count = -1
    best_homog = np.eye(3, dtype=np.float64)

    rng = np.random.default_rng() if rng is None else rng

    # Precompute homogeneous src once for compute_inliers speed.
    src_pts_homog = np.concatenate([src_pts, np.ones((N, 1), dtype=src_pts.dtype)], axis=-1).T

    def threshold_for_iter(it):
        # Mimic the original JAX decay logic
        t1 = 1.0 - (it + 1) / max(1, n_trials)
        t2 = targ_inlier_frac * (0.99 ** float(it))
        thr = min(t1, t2) * float(n_trials)
        return thr

    for it in range(n_trials):
        # Weighted sample 4 distinct correspondences
        # If not enough visible points, fallback to uniform among visible
        if probs.sum() <= 0:
            idx = rng.choice(N, size=4, replace=False)
        else:
            # np.random.Generator.choice with p requires replace=False to
            # be valid only if all probabilities are positive for chosen support.
            # To be robust to zeros, sample more and then pick unique until 4.
            # Simple practical approach:
            support = np.where(probs > 0)[0]
            if support.size >= 4:
                idx = rng.choice(support, size=4, replace=False, p=probs[support] / probs[support].sum())
            else:
                idx = rng.choice(N, size=4, replace=False, p=probs)

        H = estimate_homography(targ_pts[idx], src_pts[idx])

        inliers_mask, _, _ = compute_inliers(H, thresh, targ_pts=targ_pts, src_pts_homog=src_pts_homog)
        inliers_count = int(inliers_mask.sum())

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_homog = H

        if float(best_inliers_count) >= threshold_for_iter(it):
            break

    # Final refine with all inliers
    inliers_mask, _, _ = compute_inliers(best_homog, thresh, targ_pts=targ_pts, src_pts_homog=src_pts_homog)
    final_H = estimate_homography(targ_pts, src_pts, mask=inliers_mask.astype(np.float64))
    return final_H, inliers_mask


def maybe_ransac_homography(*arg, thresh=4.0, targ_inlier_frac=0.5):
    """Run RANSAC if there are enough visible points (NumPy)."""
    targ_pts_all, targ_occ, src_pts_all, src_occ = arg
    targ_pts_all = np.asarray(targ_pts_all, dtype=np.float64)
    src_pts_all  = np.asarray(src_pts_all,  dtype=np.float64)
    targ_occ     = np.asarray(targ_occ,     dtype=bool)
    src_occ      = np.asarray(src_occ,      dtype=bool)

    vis = np.logical_and(~targ_occ, ~src_occ).astype(np.float64)

    if np.sum(vis) > 4:
        final_homog, _ = ransac_homography(
            targ_pts_all, src_pts_all, vis, thresh=thresh, targ_inlier_frac=targ_inlier_frac
        )
    else:
        final_homog = np.eye(3, dtype=np.float64)

    inliers, err, tformed = compute_inliers(final_homog, thresh, targ_pts=targ_pts_all, src_pts=src_pts_all)
    return final_homog, inliers, tformed, err


# ---------------------------------------------------
# get_homographies_wrt_frame (NumPy replacements only)
# ---------------------------------------------------

def get_homographies_wrt_frame(
    pts,
    occ,
    image_dimensions,
    reference_frame=None,
    thresh=0.07,
    outlier_point_threshold=0.95,
    targ_inlier_frac=0.7,
    num_refinement_passes=2,
):
    """
    NumPy version. Requires a NumPy `compute_canonical_points(all_tformed_pts, all_tformed_invalid, err, thresh, outer_thresh, outlier_point_threshold)`
    that returns (canonical_pts, canonical_invalid).
    """
    # Transpose to [num_frames, num_points, 2] and normalize to [0,1]
    pts = np.transpose(np.asarray(pts, dtype=np.float64), (1, 0, 2)) / np.asarray(image_dimensions, dtype=np.float64)
    occ = np.transpose(np.asarray(occ, dtype=bool))  # [num_frames, num_points]

    outer_thresh = thresh * 2.0
    if reference_frame is None:
        reference_frame = pts.shape[0] // 2

    canonical_pts = pts[reference_frame]
    canonical_invalid = occ[reference_frame]

    all_tformed_pts = np.zeros_like(pts)
    all_tformed_invalid = np.ones_like(occ, dtype=bool)
    all_err = np.zeros(occ.shape, dtype=np.float64)

    all_tformed_pts[reference_frame] = canonical_pts
    all_tformed_invalid[reference_frame] = canonical_invalid

    res_homog = [None] * pts.shape[0]
    res_homog[reference_frame] = np.eye(3, dtype=np.float64)

    after  = list(range(reference_frame + 1, pts.shape[0]))
    before = list(range(reference_frame - 1, -1, -1))

    # --- Initial pass ---
    for i in after + before:
        # print(f'Initial RANSAC frame {i}...')
        res, _, tformed, err = maybe_ransac_homography(
            canonical_pts, canonical_invalid, pts[i], occ[i],
            thresh=thresh, targ_inlier_frac=targ_inlier_frac,
        )
        all_tformed_pts[i] = tformed
        all_tformed_invalid[i] = occ[i]
        all_err[i] = err
        res_homog[i] = res

        # You must implement this function in NumPy:
        canonical_pts, canonical_invalid = compute_canonical_points(
            all_tformed_pts, all_tformed_invalid, err,
            thresh, outer_thresh, outlier_point_threshold
        )

    # --- Refinement passes ---
    for j in range(num_refinement_passes):
        for fr in [reference_frame] + after + before:
            # print(f'Refinement pass {j} frame {fr}...')
            _, err, _ = compute_inliers(res_homog[fr], thresh, canonical_pts, pts[fr])
            invalid = np.logical_or(canonical_invalid, err > (thresh * thresh))
            invalid = np.logical_or(occ[fr], invalid)

            valid_mask = (~invalid).astype(bool)

            homog = estimate_homography(canonical_pts, pts[fr], mask=(~invalid).astype(np.float64))
            

            if fr == reference_frame and j != num_refinement_passes - 1:
                # Re-anchor scale to the reference frame
                try:
                    inv_homog = np.linalg.inv(homog)
                except np.linalg.LinAlgError:
                    # Option A (recommended): skip re-anchoring this pass
                    inv_homog = None
                    # Option B (more aggressive): use pseudo-inverse
                    # inv_homog = np.linalg.pinv(homog)
                if inv_homog is not None:
                    for fr2 in range(pts.shape[0]):
                        res_homog[fr2] = inv_homog @ res_homog[fr2]
                        _, _, tformed = compute_inliers(res_homog[fr2], thresh, canonical_pts, pts[fr2])
                        all_tformed_pts[fr2] = tformed
                    homog = np.eye(3, dtype=np.float64)
                    
                for fr2 in range(pts.shape[0]):
                    res_homog[fr2] = inv_homog @ res_homog[fr2]
                    _, _, tformed2 = compute_inliers(res_homog[fr2], thresh, canonical_pts, pts[fr2])
                    all_tformed_pts[fr2] = tformed2
                homog = np.eye(3, dtype=np.float64)

                canonical_pts, _ = compute_canonical_points(
                    all_tformed_pts, all_tformed_invalid, all_err,
                    thresh, outer_thresh, outlier_point_threshold
                )

            _, err2, tformed = compute_inliers(homog, thresh, canonical_pts, pts[fr])
            all_tformed_pts[fr] = tformed
            all_err[fr] = err2
            res_homog[fr] = homog

            canonical_pts, canonical_invalid = compute_canonical_points(
                all_tformed_pts, all_tformed_invalid, err2,
                thresh, outer_thresh, outlier_point_threshold
            )

    all_err = np.transpose(all_err)  # [num_points, num_frames]

    scaler = np.asarray(list(image_dimensions) + [1.0], dtype=np.float64)
    res_homog = [H @ np.diag(1.0 / scaler) for H in res_homog]

    return np.stack(res_homog, axis=0), all_err, canonical_pts


# ----------------------
# Plot (already NumPy)
# ----------------------

def plot_tracks_tails(
    rgb,
    points,
    occluded,
    homogs,
    point_size=6,
    linewidth=0.4,
    query_frame: int = 0,
    cmap=plt.cm.jet,
    tail_len: int = 6,
):
    """Plot tracks with tails, using colors normalized by y at query_frame (like plot_tracks_wo_tail)."""
    import numpy as np
    import torch
    import matplotlib
    import matplotlib.pyplot as plt

    # --- accept torch tensor or numpy ---
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()

    # rgb expected (T, H, W, 3)
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"rgb must be (T,H,W,3), got {rgb.shape}"

    disp = []
    T, H, W = rgb.shape[0], rgb.shape[1], rgb.shape[2]

    # --- colors by y position at query_frame (normalized) ---
    y_vals = points[:, query_frame, 1]
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    if y_max == y_min:
        y_max = y_min + 1e-6  # avoid zero-range
    norm = plt.Normalize(y_min, y_max)
    base_colors = cmap(norm(y_vals))[:, :3]  # (N,3)

    dpi = 64
    figs = []
    for i in range(T):
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, frameon=False, facecolor='w')
        figs.append(fig)
        ax = fig.add_subplot()
        ax.axis('off')
        ax.imshow(rgb[i] / 255.0)

        pts_clamped = np.clip(points, [0.0, 0.0], [W - 1, H - 1])

        # Head (current frame) points with alpha from visibility
        alpha_head = (1 - occluded[:, i:i+1]).astype(float)  # (N,1)
        col_head = np.concatenate([base_colors, alpha_head], axis=1)  # (N,4)
        plt.scatter(pts_clamped[:, i, 0], pts_clamped[:, i, 1], s=point_size, c=col_head)

        # Tails: same color per track; alpha from visibility of both ends and in-frame mask
        reference = pts_clamped[:, i]
        reference_occ = occluded[:, i:i+1]

        j_start = max(i - tail_len, 0)
        for j in range(i - 1, j_start - 1, -1):
            # Warp points from frame j to frame i using homographies
            points_homo = np.concatenate([pts_clamped[:, j], np.ones_like(pts_clamped[:, j, 0:1])], axis=1)
            M = np.linalg.inv(homogs[i]) @ homogs[j]
            points_transf = (M @ points_homo.T).T
            w = points_transf[:, 2:3]
            denom = np.maximum(1e-12, np.abs(w)) * np.sign(w)
            points_transf = points_transf[:, :2] / denom

            segs = np.stack([points_transf, reference], axis=1)  # (N,2,2)

            # Out-of-frame check (allow slightly inside [0..W-1], [0..H-1])
            oof = np.logical_or(segs < 0.0, segs > np.array([W - 1, H - 1]))
            oof = np.logical_or(oof[:, 0], oof[:, 1])
            oof = np.logical_or(oof[:, 0:1], oof[:, 1:2])

            segs = np.clip(segs, [0.0, 0.0], [W - 1, H - 1])

            alpha_tail = (1 - occluded[:, j:j+1]).astype(float) * (1 - reference_occ).astype(float) * (1 - oof).astype(float)
            col_tail = np.concatenate([base_colors, alpha_tail], axis=1)  # (N,4)

            plt.gca().add_collection(
                matplotlib.collections.LineCollection(segs, color=col_tail, linewidth=linewidth)
            )

            reference_occ = occluded[:, j:j+1]
            reference = points_transf

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.canvas.draw()

        # reshape buffer using known (H, W) to avoid swap bug
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(H, W, 4)[..., :3]
        disp.append(img.copy())

    for fig in figs:
        plt.close(fig)

    return np.stack(disp, axis=0)

def plot_tracks_wo_tail(
    rgb,
    points: np.ndarray,
    occluded: np.ndarray,
    gt_points=None,
    gt_occluded=None,
    point_size: int = 4,
    query_frame: int = 0,
    cmap=plt.cm.jet,
) -> np.ndarray:
    # --- accept torch tensor or numpy ---
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    # rgb expected (T, H, W, 3)
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"rgb must be (T,H,W,3), got {rgb.shape}"

    disp = []
    T = rgb.shape[0]
    H, W = rgb.shape[1], rgb.shape[2]

    # --- assign colors by y position at query_frame ---
    y_min, y_max = points[:, query_frame, 1].min(), points[:, query_frame, 1].max()
    norm = plt.Normalize(y_min, y_max)
    base_colors = cmap(norm(points[:, query_frame, 1]))[:, :3]  # (N,3)

    dpi = 64
    for i in range(T):
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, frameon=False, facecolor='w')
        ax = fig.add_subplot()
        ax.axis('off')
        ax.imshow(rgb[i] / 255.0)

        alpha = (1 - occluded[:, i:i+1]).astype(float)
        colalpha = np.concatenate([base_colors, alpha], axis=1)

        pts_clamped = np.clip(points, [0, 0], [W - 1, H - 1])
        plt.scatter(pts_clamped[:, i, 0], pts_clamped[:, i, 1],
                    s=point_size, c=colalpha)

        if gt_points is not None:
            gt_pts_clamped = np.clip(gt_points, [0, 0], [W - 1, H - 1])
            alpha_gt = 1 - gt_occluded[:, i:i+1].astype(float)
            colalpha_gt = np.concatenate([base_colors, alpha_gt], axis=1)
            plt.scatter(gt_pts_clamped[:, i, 0], gt_pts_clamped[:, i, 1],
                        s=point_size + 6, c=colalpha_gt, marker='D')

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.canvas.draw()

        # --- reshape buffer using known (H, W) to avoid swap bug ---
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(H, W, 4)[..., :3]  # (H, W, 3)
        disp.append(img.copy())
        plt.close(fig)

    return np.stack(disp, axis=0)  # (T, H, W, 3)

def save_video(frames, path, fps=10, codec="mp4v"):
    """
    Save a numpy array of frames to a video file.

    Args:
        frames: numpy array of shape [num_frames, height, width, 3], dtype=uint8
        path:   output filepath (e.g. "output.mp4" or "results.avi")
        fps:    frames per second (default 30)
        codec:  fourcc codec string ("mp4v", "XVID", etc.)

    Returns:
        None
    """
    frames = np.asarray(frames)
    assert frames.ndim == 4 and frames.shape[-1] == 3, "Expected [T,H,W,3] frames array"
    assert frames.dtype == np.uint8, "Frames must be uint8 (0–255)"

    height, width = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for f in frames:
        writer.write(f[..., ::-1])  # convert RGB → BGR for OpenCV
    writer.release()
    print(f"Video saved to {path}")


def vis(video, tracks, visibles, N=None, calc_homogs=True):
    # :args video: (T, H, W, 3) in range [0, 255]
    # :args tracks: (T, N, 2) in range [0, W] and [0, H], torch
    # :args visibles: (T, N), bool, torch

    ransac_inlier_threshold = 0.07  # @param {type: "number"}
    # What fraction of points need to be inliers for RANSAC to consider a trajectory
    # to be trustworthy for estimating the homography.
    ransac_track_inlier_frac = 0.95  # @param {type: "number"}
    # After initial RANSAC, how many refinement passes to adjust the homographies
    # based on tracks that have been deemed trustworthy.
    num_refinement_passes = 2  # @param {type: "number"}
    # After homographies are estimated, consider points to be outliers if they are
    # further than this threshold.
    foreground_inlier_threshold = 0.07  # @param {type: "number"}
    # After homographies are estimated, consider tracks to be part of the foreground
    # if less than this fraction of its points are inliers.
    foreground_frac = 0.6  # @param {type: "number"}
    width, height = video.shape[2], video.shape[1]


    occluded = 1 - visibles

    if calc_homogs:
        homogs, err, canonical = get_homographies_wrt_frame(
            tracks,
            occluded,
            [width, height],
            thresh=ransac_inlier_threshold,
            outlier_point_threshold=ransac_track_inlier_frac,
            num_refinement_passes=num_refinement_passes,
        )

        inliers = (err < np.square(foreground_inlier_threshold)) * visibles
        inlier_ct = np.sum(inliers, axis=-1)
        ratio = inlier_ct / np.maximum(1.0, np.sum(visibles, axis=1))
        is_fg = ratio <= foreground_frac

    else:
        num_frames = tracks.shape[1]
        homogs = np.tile(np.eye(3)[None, :, :], (num_frames, 1, 1))
        is_fg = np.ones((tracks.shape[0],), dtype=bool)

    if N is not None:
        tracks = tracks[:N]
        occluded = occluded[:N]
        is_fg = is_fg[:N]


    video = plot_tracks_tails(
        video, tracks[is_fg], occluded[is_fg], homogs, point_size=64, linewidth=1.0
    )

    return video