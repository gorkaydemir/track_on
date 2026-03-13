import glob, os
from typing import Tuple, List, Optional
from tqdm import tqdm
import cv2
import numpy as np
from torch.utils.data import Dataset


def _clamp255(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 255.0, out=x)

def _maybe(p: float) -> bool:
    return np.random.rand() < p

def _rand(a: float, b: float) -> float:
    return float(np.random.uniform(a, b))

# ---- photometric helpers (unchanged) ----
def _apply_brightness_contrast(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return _clamp255(img * alpha + beta)

def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    out = 255.0 * np.power((img / 255.0), inv_gamma, dtype=np.float32)
    return _clamp255(out)

def _apply_saturation(img: np.ndarray, sat_scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

def _apply_hue(img: np.ndarray, hue_delta_deg: float) -> np.ndarray:
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_delta_deg / 2.0) % 180.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

def _apply_grayscale_blend(img: np.ndarray, gray_alpha: float) -> np.ndarray:
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray3 = np.repeat(gray[..., None], 3, axis=2)
    return _clamp255(gray_alpha * gray3 + (1.0 - gray_alpha) * img)

def _apply_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    return _clamp255(img + np.random.normal(0.0, sigma, img.shape).astype(np.float32))

def _apply_gaussian_blur(img: np.ndarray, k: int, sigma: float) -> np.ndarray:
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101).astype(np.float32)

def _apply_jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok: return img
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)


class EpicK(Dataset):
    """
    NumPy-first dataset with clip-consistent photometric + spatial aug.
    Returns:
      video: [T, 3, H, W] float32 in 0..255
      tracks: [T, Nc, 2]  after spatial transform
      visibility: [T, Nc] original vis AND in-bounds after transform
    """
    def __init__(
        self,
        dataset_location: str,
        image_size: Tuple[int, int] = (384, 512),  # (H, W)
        T: int = 24,
        N: Optional[int] = None,
        max_resample_tries: int = 10,
        # photometric
        enable_aug: bool = True,
        p_brightness_contrast: float = 0.7,
        brightness_delta: float = 32.0,
        contrast_range: Tuple[float, float] = (0.8, 1.25),
        p_gamma: float = 0.5,
        gamma_range: Tuple[float, float] = (0.8, 1.25),
        p_saturation: float = 0.6,
        saturation_range: Tuple[float, float] = (0.7, 1.4),
        p_hue: float = 0.4,
        hue_delta_deg: float = 10.0,
        p_grayscale: float = 0.15,
        grayscale_alpha_range: Tuple[float, float] = (0.1, 0.4),
        p_noise: float = 0.3,
        noise_sigma_range: Tuple[float, float] = (2.0, 8.0),
        p_blur: float = 0.25,
        blur_kernel_choices: Tuple[int, ...] = (3, 5),
        blur_sigma_range: Tuple[float, float] = (0.2, 1.0),
        p_jpeg: float = 0.3,
        jpeg_quality_range: Tuple[int, int] = (40, 90),
        # spatial aug
        enable_spatial_aug: bool = True,
        p_hflip: float = 0.5,
        rot_deg: float = 8.0,                 # random in [-rot_deg, +rot_deg]
        scale_range: Tuple[float, float] = (0.9, 1.1),
        trans_frac: Tuple[float, float] = (0.05, 0.05),  # fractions of (W,H)
    ):
        self.image_size = image_size
        self.T = int(T)
        self.N = None if N is None else int(N)
        self.max_resample_tries = int(max_resample_tries)

        # photometric cfg
        self.enable_aug = enable_aug
        self.p_brightness_contrast = p_brightness_contrast
        self.brightness_delta = float(brightness_delta)
        self.contrast_range = contrast_range
        self.p_gamma = p_gamma
        self.gamma_range = gamma_range
        self.p_saturation = p_saturation
        self.saturation_range = saturation_range
        self.p_hue = p_hue
        self.hue_delta_deg = float(hue_delta_deg)
        self.p_grayscale = p_grayscale
        self.grayscale_alpha_range = grayscale_alpha_range
        self.p_noise = p_noise
        self.noise_sigma_range = noise_sigma_range
        self.p_blur = p_blur
        self.blur_kernel_choices = blur_kernel_choices
        self.blur_sigma_range = blur_sigma_range
        self.p_jpeg = p_jpeg
        self.jpeg_quality_range = jpeg_quality_range

        # spatial cfg
        self.enable_spatial_aug = enable_spatial_aug
        self.p_hflip = p_hflip
        self.rot_deg = float(rot_deg)
        self.scale_range = scale_range
        self.trans_frac = trans_frac  # (tx_frac_w, ty_frac_h)

        # discover sequences
        all_ann_paths: List[str] = sorted(glob.glob(os.path.join(dataset_location, "*/annot.npz")))
        all_rgb_paths_per_seq: List[List[str]] = [
            sorted(glob.glob(os.path.join(os.path.dirname(p), "rgbs", "*.jpg"))) for p in all_ann_paths
        ]
        self.annotation_paths, self.rgb_paths_per_seq = [], []
        for ann_path, rgb_list in zip(all_ann_paths, all_rgb_paths_per_seq):

            self.annotation_paths.append(ann_path)
            self.rgb_paths_per_seq.append(rgb_list)

        if not self.annotation_paths:
            raise RuntimeError(f"No valid sequences with >=T={self.T} in {dataset_location}")

    def __len__(self): return len(self.annotation_paths)

    # --------- photometric (clip-level) ----------
    def _photometric_augment_clip(self, video_thwc: np.ndarray) -> np.ndarray:
        if not self.enable_aug:
            return video_thwc
        T, H, W, _ = video_thwc.shape
        out = video_thwc

        do_bc   = _maybe(self.p_brightness_contrast); alpha = _rand(*self.contrast_range) if do_bc else 1.0; beta = _rand(-self.brightness_delta, self.brightness_delta) if do_bc else 0.0
        do_gamma= _maybe(self.p_gamma); gamma = _rand(*self.gamma_range) if do_gamma else 1.0
        do_sat  = _maybe(self.p_saturation); sat_s = _rand(*self.saturation_range) if do_sat else 1.0
        do_hue  = _maybe(self.p_hue); hue_d = _rand(-self.hue_delta_deg, self.hue_delta_deg) if do_hue else 0.0
        do_gray = _maybe(self.p_grayscale); gray_a = _rand(*self.grayscale_alpha_range) if do_gray else 0.0
        do_noise= _maybe(self.p_noise); noise_s = _rand(*self.noise_sigma_range) if do_noise else 0.0
        do_blur = _maybe(self.p_blur); blur_k = int(np.random.choice(self.blur_kernel_choices)) if do_blur else 3; blur_s = _rand(*self.blur_sigma_range) if do_blur else 0.0
        do_jpeg = _maybe(self.p_jpeg); jpeg_q = int(np.random.randint(self.jpeg_quality_range[0], self.jpeg_quality_range[1] + 1)) if do_jpeg else 95

        for t in range(T):
            img = out[t]
            if do_bc:   img = _apply_brightness_contrast(img, alpha, beta)
            if do_gamma:img = _apply_gamma(img, gamma)
            if do_sat:  img = _apply_saturation(img, sat_s)
            if do_hue:  img = _apply_hue(img, hue_d)
            if do_gray: img = _apply_grayscale_blend(img, gray_a)
            if do_noise and noise_s > 0: img = _apply_gaussian_noise(img, noise_s)
            if do_blur: img = _apply_gaussian_blur(img, blur_k, blur_s)
            if do_jpeg: img = _apply_jpeg_compress(img, jpeg_q)
            out[t] = img
        return out

    # --------- spatial (clip-level affine) ----------
    def _sample_affine(self, H: int, W: int) -> np.ndarray:
        """
        Build a 2x3 affine M combining optional hflip, scale, rotation, translation,
        around the image center (W/2, H/2).
        """
        # base: identity around center
        cx, cy = W * 0.5, H * 0.5
        M = np.array([[1, 0, 0],
                      [0, 1, 0]], dtype=np.float32)

        # optional horizontal flip
        if self.enable_spatial_aug and _maybe(self.p_hflip):
            F = np.array([[-1, 0, 2*cx],
                          [ 0, 1,     0]], dtype=np.float32)
            M = F @ np.vstack([M, [0,0,1]])  # compose

        # scale + rotation around center
        if self.enable_spatial_aug:
            s = _rand(*self.scale_range)
            ang = _rand(-self.rot_deg, self.rot_deg)
        else:
            s, ang = 1.0, 0.0
        R = cv2.getRotationMatrix2D((cx, cy), ang, s).astype(np.float32)  # 2x3
        M = R @ np.vstack([M, [0,0,1]])  # compose

        # translation (fractions of W/H)
        if self.enable_spatial_aug:
            tx = _rand(-self.trans_frac[0]*W, self.trans_frac[0]*W)
            ty = _rand(-self.trans_frac[1]*H, self.trans_frac[1]*H)
        else:
            tx, ty = 0.0, 0.0
        M[:, 2] += np.array([tx, ty], dtype=np.float32)

        return M.astype(np.float32)  # 2x3

    def _apply_affine_clip(self, video_thwc: np.ndarray, tracks: np.ndarray, vis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply one affine M to all frames and points.
        Inputs:
          video_thwc: [T,H,W,3] float32
          tracks:     [T,N,2]  float32
          vis:        [T,N]    float32 (0/1)
        Returns transformed (video, tracks, vis_new).
        """
        T, H, W, _ = video_thwc.shape
        M = self._sample_affine(H, W)  # 2x3

        # warp frames
        for t in range(T):
            video_thwc[t] = cv2.warpAffine(
                video_thwc[t], M, (W, H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
            ).astype(np.float32)

        # transform all (x,y)
        xy = tracks.reshape(-1, 2)                                           # [T*N,2]
        ones = np.ones((xy.shape[0], 1), dtype=np.float32)
        xy_h = np.concatenate([xy, ones], axis=1)                            # [T*N,3]
        xy_t = xy_h @ M.T                                                    # [T*N,2]
        tracks_t = xy_t.reshape(tracks.shape).astype(np.float32)             # [T,N,2]

        # recompute in-bounds mask after transform
        x, y = tracks_t[..., 0], tracks_t[..., 1]
        inb = (x >= 0) & (x < (W - 1)) & (y >= 0) & (y < (H - 1))            # [T,N] bool
        vis_new = (vis > 0) & inb                                            # combine original vis with in-bounds
        vis_new = vis_new.astype(np.float32)

        return video_thwc, tracks_t, vis_new
    # --------- main ----------
    def __getitem__(self, index: int):
        annot_path = self.annotation_paths[index]
        rgb_paths  = self.rgb_paths_per_seq[index]
    
        # --- Load annotations ---
        ann = np.load(annot_path)
        trajs   = ann["trajs_2d"].astype(np.float32)  # [S, Ntot, 2]
        visibs  = ann["visibs"].astype(np.float32)    # [S, Ntot]
        valids  = ann["valids"].astype(np.float32)    # [S, Ntot]
    
        S, Ntot, _ = trajs.shape
        if S < self.T or S != len(rgb_paths):
            raise RuntimeError(f"Bad sequence: S={S}, frames={len(rgb_paths)}, T={self.T} @ {annot_path}")
    
        # --- Resize scales from first frame ---
        probe = cv2.imread(rgb_paths[0], cv2.IMREAD_COLOR)
        if probe is None:
            raise FileNotFoundError(f"Failed to read image: {rgb_paths[0]}")
        H_bak, W_bak = probe.shape[:2]
        H, W = self.image_size
        sy, sx = np.float32(H / H_bak), np.float32(W / W_bak)
    
        tries = 0
        while True:
            # 1) Sample a T-length window
            start = 0 if S == self.T else int(np.random.randint(0, S - self.T + 1))
            end = start + self.T
            idxs = np.arange(start, end, dtype=np.int32)
    
            # 2) Slice + rescale tracks to (H,W) BEFORE spatial aug
            tracks_win = trajs[idxs].copy()           # [T, Ntot, 2]
            vis_win    = visibs[idxs].copy()          # [T, Ntot]
            val_win    = valids[idxs].copy()          # [T, Ntot]
            tracks_win[..., 0] *= sx
            tracks_win[..., 1] *= sy
    
            # 3) Load & resize frames (THWC float32 in [0,255])
            video_thwc = np.empty((self.T, H, W, 3), dtype=np.uint8)
            for i, t in enumerate(idxs):
                img = cv2.imread(rgb_paths[t], cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"Failed to read image: {rgb_paths[t]}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                video_thwc[i] = img
            video_thwc = video_thwc.astype(np.float32)
    
            # 4) Photometric aug (clip-consistent)
            video_thwc = self._photometric_augment_clip(video_thwc)
    
            # 5) Spatial aug (clip-consistent) -> transforms BOTH frames & GT; recompute in-bounds
            if self.enable_spatial_aug:
                video_thwc, tracks_aug, vis_after = self._apply_affine_clip(video_thwc, tracks_win, vis_win)
                vis_strict = (vis_after * (val_win > 0)).astype(np.float32)   # [T, Ntot]
                tracks_curr = tracks_aug
            else:
                vis_strict  = (vis_win * (val_win > 0)).astype(np.float32)
                tracks_curr = tracks_win
    
            # 6) SELECT POINTS **AFTER AUGS**:
            #    - must be visible at the FIRST frame of the window
            cand_mask_t0 = (vis_strict[0] > 0)                      # [Ntot] bool
            #    - (optional safety) also drop tracks invisible across ALL T
            alive_mask_T = (vis_strict.sum(axis=0) > 0)             # [Ntot] bool
            final_mask   = cand_mask_t0 & alive_mask_T
    
            if not final_mask.any():
                tries += 1
                if tries >= self.max_resample_tries:
                    raise RuntimeError(
                        f"No points survive (visible at t0 after aug) in {self.max_resample_tries} tries "
                        f"(S={S}, T={self.T}) for: {annot_path}"
                    )
                continue
    
            tracks_final = tracks_curr[:, final_mask]               # [T, Nkeep, 2]
            vis_final    = vis_strict[:, final_mask]                # [T, Nkeep]
            Nkeep = tracks_final.shape[1]
    
            # 7) Subsample to N AFTER filtering
            if self.N is not None and Nkeep > self.N:
                sel = np.random.choice(Nkeep, self.N, replace=False)
                tracks_final = tracks_final[:, sel]
                vis_final    = vis_final[:, sel]
    
            # success → break loop
            break
    
        # CHW
        video = video_thwc.transpose(0, 3, 1, 2).astype(np.float32)  # [T,3,H,W]
        return video, tracks_final.astype(np.float32), vis_final.astype(np.float32)