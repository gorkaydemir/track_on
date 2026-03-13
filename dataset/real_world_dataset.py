import glob
import os
from typing import Tuple, Optional
import cv2
import numpy as np
from torch.utils.data import Dataset

import re

def _clamp255(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 255.0, out=x)

def _maybe(p: float) -> bool:
    return np.random.rand() < p

def _rand(a: float, b: float) -> float:
    return float(np.random.uniform(a, b))

# Photometric augmentation helpers
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

def _apply_motion_blur(img: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
    """Apply motion blur with given kernel size and angle."""
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    
    # Calculate kernel center
    center = kernel_size // 2
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Calculate direction vector
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    # Draw line in kernel
    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * dx)
        y = int(center + offset * dy)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    
    # Normalize kernel
    kernel /= kernel.sum()
    
    # Apply convolution
    blurred = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT101)
    return blurred.astype(np.float32)


def _extract_frame_number(filepath: str) -> int:
    """Extract frame number from filepath, handling various naming conventions.
    Examples: '0001.jpg', 'frame0091.jpg', 'img_00123.jpg' -> returns the numeric part
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    # Extract all digits from the basename
    numbers = re.findall(r'\d+', basename)
    if numbers:
        # Return the last number found (typically the frame index)
        return int(numbers[-1])
    # Fallback: return 0 if no number found
    return 0

class Real_World_Dataset(Dataset):
    """
    Real-world dataset that loads frames, applies random crop/resize and photometric augmentation,
    and extracts queries based on SIFT keypoints and motion saliency.
    
    Returns:
      video_aug: [T, 3, H, W] float32 in 0..255 (augmented)
      video_clean: [T, 3, H, W] float32 in 0..255 (only cropped, no photometric aug)
      queries: [N, 3] float32 in (t, x, y) format
    """
    def __init__(
        self,
        dataset_location: str,
        image_size: Tuple[int, int] = (384, 512),  # (H, W)
        T: int = 48,
        N: int = 256,
        use_ovis: bool = False,
        use_vspw: bool = False,
        frame_rate: float = 1.0,
        max_video_num: Optional[int] = None,
        # photometric augmentation
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
        p_motion_blur: float = 0.2,
        motion_blur_kernel_choices: Tuple[int, ...] = (5, 7, 9),
        p_jpeg: float = 0.3,
        jpeg_quality_range: Tuple[int, int] = (40, 90),
        # SIFT parameters
        sift_contrast_threshold: float = 0.04,
        sift_edge_threshold: int = 10,
    ):
        self.image_size = image_size
        self.T = int(T)
        self.N = int(N)
        self.frame_rate = float(frame_rate)
        self.max_video_num = max_video_num
        
        # Calculate stride for frame sampling
        self.frame_stride = int(self.frame_rate) 
        
        # Photometric augmentation config
        self.enable_aug = enable_aug
        self.p_brightness_contrast = p_brightness_contrast
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.p_gamma = p_gamma
        self.gamma_range = gamma_range
        self.p_saturation = p_saturation
        self.saturation_range = saturation_range
        self.p_hue = p_hue
        self.hue_delta_deg = hue_delta_deg
        self.p_grayscale = p_grayscale
        self.grayscale_alpha_range = grayscale_alpha_range
        self.p_noise = p_noise
        self.noise_sigma_range = noise_sigma_range
        self.p_blur = p_blur
        self.blur_kernel_choices = blur_kernel_choices
        self.blur_sigma_range = blur_sigma_range
        self.p_motion_blur = p_motion_blur
        self.motion_blur_kernel_choices = motion_blur_kernel_choices
        self.p_jpeg = p_jpeg
        self.jpeg_quality_range = jpeg_quality_range
        
        # SIFT parameters
        self.sift_contrast_threshold = sift_contrast_threshold
        self.sift_edge_threshold = sift_edge_threshold
        self.sift = cv2.SIFT_create(
            contrastThreshold=sift_contrast_threshold,
            edgeThreshold=sift_edge_threshold
        )
        
        self.use_ovis = use_ovis
        self.use_vspw = use_vspw

        self._load_tao_paths(dataset_location)
    

    def _load_tao_paths(self, dataset_location: str):
        """Load TAO dataset paths with structure: parent/frames/{split}/{source}/{video_id}/{frame_idx}.jpg
        where split is one of: train, test, val
        and source is one of: ArgoVerse, BDD, Charades, HACS, LaSOT, YFCC100M, OVIS, VSPW
        """
        # Base TAO sources vs extras
        TAO_BASE_SOURCES = {"ava", "argoverse", "bdd", "charades", "hacs", "lasot", "yfcc100m"}

        # Define split folders based on dataset_content
        split_folders = ["train", "test", "val"]
        
        frames_dir = os.path.join(dataset_location, "frames")
        if not os.path.isdir(frames_dir):
            raise RuntimeError(f"Frames directory not found: {frames_dir}")
        
        self.rgb_paths_per_seq = []
        # source_name (as found on disk) -> sequence count
        source_counts: dict = {}
        
        # Iterate through each split folder
        for split_folder in split_folders:
            split_path = os.path.join(frames_dir, split_folder)
            
            # Skip if split folder doesn't exist
            if not os.path.isdir(split_path):
                print(f"Warning: Split folder not found: {split_path}")
                continue
            
            # Get all source directories (ArgoVerse, BDD, Charades, HACS, LaSOT, YFCC100M)
            source_dirs = sorted([d for d in glob.glob(os.path.join(split_path, "*")) 
                                if os.path.isdir(d)])
            
            # if use_ovis is False, then remove any source that contains "ovis" in its name
            if not self.use_ovis:
                source_dirs = [d for d in source_dirs if "ovis" not in os.path.basename(d).lower()]

            if not self.use_vspw:
                source_dirs = [d for d in source_dirs if "vspw" not in os.path.basename(d).lower()]
            
            for source_dir in source_dirs:
                source_name = os.path.basename(source_dir)
                
                # Get all video directories under this source
                video_dirs = sorted([d for d in glob.glob(os.path.join(source_dir, "*")) 
                                   if os.path.isdir(d)])
                
                for video_dir in video_dirs:
                    video_name = os.path.basename(video_dir)
                    
                    # Get all frame paths, sorted by frame number
                    frame_paths = sorted(
                        glob.glob(os.path.join(video_dir, "*.jpg")),
                        key=_extract_frame_number
                    )

                    # Only keep sequences with at least T frames    
                    if len(frame_paths) >= self.T:
                        self.rgb_paths_per_seq.append(frame_paths)
                        source_counts[source_name] = source_counts.get(source_name, 0) + 1
        
        if not self.rgb_paths_per_seq:
            raise RuntimeError(f"No valid TAO sequences found in {dataset_location}")
        
        # Print per-source statistics (before any subsampling)
        total_found = len(self.rgb_paths_per_seq)
        print("  Source breakdown:")
        for source_name in sorted(source_counts.keys()):
            count = source_counts[source_name]
            if source_name.lower() in TAO_BASE_SOURCES:
                label = f"(TAO) {source_name}"
            else:
                label = source_name
            print(f"    {label}: {count}")

        # Uniform sampling if max_video_num is specified
        if self.max_video_num is not None and len(self.rgb_paths_per_seq) > self.max_video_num:
            indices = np.linspace(0, total_found - 1, self.max_video_num, dtype=int)
            self.rgb_paths_per_seq = [self.rgb_paths_per_seq[i] for i in indices]
            print(f"Uniformly sampled {self.max_video_num} sequences from {total_found}")
    
    

    def __len__(self):
        return len(self.rgb_paths_per_seq)
    
    def _random_crop_params(self, H_orig: int, W_orig: int) -> Tuple[int, int, int, int]:
        """Sample random crop coordinates with fixed scale range."""
        H, W = self.image_size
        
        # Sample scale from range [0.6, 1.0] where scale=1.0 means crop entire image
        # scale=0.6 means crop 60% of original dimensions
        scale = np.random.uniform(0.7, 1.0)
        
        # Calculate crop size (scale * original dimensions)
        H_crop = int(H_orig * scale)
        W_crop = int(W_orig * scale)
        
        # Ensure crop is at least as large as target size
        H_crop = max(H_crop, H)
        W_crop = max(W_crop, W_orig)
        
        # Ensure crop doesn't exceed original image
        H_crop = min(H_crop, H_orig)
        W_crop = min(W_crop, W_orig)
        
        # Random crop position
        top = np.random.randint(0, max(1, H_orig - H_crop + 1))
        left = np.random.randint(0, max(1, W_orig - W_crop + 1))
        
        return top, left, H_crop, W_crop
    
    def _apply_crop_and_resize(self, frame: np.ndarray, crop_params: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply crop and resize to a single frame."""
        top, left, H_crop, W_crop = crop_params
        H, W = self.image_size
        
        # Crop
        cropped = frame[top:top+H_crop, left:left+W_crop]
        
        # Resize to target size
        resized = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def _photometric_augment_clip(self, video_thwc: np.ndarray) -> np.ndarray:
        """Apply clip-consistent photometric augmentation."""
        if not self.enable_aug:
            return video_thwc
        
        T = video_thwc.shape[0]
        out = video_thwc.copy()
        
        # Sample augmentation parameters (same for all frames)
        do_bc = _maybe(self.p_brightness_contrast)
        alpha = _rand(*self.contrast_range) if do_bc else 1.0
        beta = _rand(-self.brightness_delta, self.brightness_delta) if do_bc else 0.0
        
        do_gamma = _maybe(self.p_gamma)
        gamma = _rand(*self.gamma_range) if do_gamma else 1.0
        
        do_sat = _maybe(self.p_saturation)
        sat_s = _rand(*self.saturation_range) if do_sat else 1.0
        
        do_hue = _maybe(self.p_hue)
        hue_d = _rand(-self.hue_delta_deg, self.hue_delta_deg) if do_hue else 0.0
        
        do_gray = _maybe(self.p_grayscale)
        gray_a = _rand(*self.grayscale_alpha_range) if do_gray else 0.0
        
        do_noise = _maybe(self.p_noise)
        noise_s = _rand(*self.noise_sigma_range) if do_noise else 0.0
        
        do_blur = _maybe(self.p_blur)
        blur_k = int(np.random.choice(self.blur_kernel_choices)) if do_blur else 3
        blur_s = _rand(*self.blur_sigma_range) if do_blur else 0.0
        
        do_motion_blur = _maybe(self.p_motion_blur)
        motion_k = int(np.random.choice(self.motion_blur_kernel_choices)) if do_motion_blur else 5
        motion_angle = _rand(0, 360) if do_motion_blur else 0.0
        
        do_jpeg = _maybe(self.p_jpeg)
        jpeg_q = int(np.random.randint(self.jpeg_quality_range[0], self.jpeg_quality_range[1] + 1)) if do_jpeg else 95
        
        # Apply to all frames
        for t in range(T):
            img = out[t]
            if do_bc: img = _apply_brightness_contrast(img, alpha, beta)
            if do_gamma: img = _apply_gamma(img, gamma)
            if do_sat: img = _apply_saturation(img, sat_s)
            if do_hue: img = _apply_hue(img, hue_d)
            if do_gray: img = _apply_grayscale_blend(img, gray_a)
            if do_noise and noise_s > 0: img = _apply_gaussian_noise(img, noise_s)
            if do_blur: img = _apply_gaussian_blur(img, blur_k, blur_s)
            if do_motion_blur: img = _apply_motion_blur(img, motion_k, motion_angle)
            if do_jpeg: img = _apply_jpeg_compress(img, jpeg_q)
            out[t] = img
        
        return out


    def _extract_motion_queries(self, video_thwc: np.ndarray, num_points: int) -> np.ndarray:
        """Extract query points from regions with motion using simple frame differencing."""
        T, H, W = video_thwc.shape[:3]
        T_half = video_thwc.shape[0] // 3
        query_frame_indices = list(range(0, T_half + 1, 4))
        if len(query_frame_indices) == 0:
            query_frame_indices = [0]
        
        all_keypoints = []
        
        for t in query_frame_indices:
            # Skip last frame (no next frame to compute difference)
            if t >= T - 1:
                continue
            
            # Convert to grayscale
            frame1_gray = cv2.cvtColor(video_thwc[t].astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            frame2_gray = cv2.cvtColor(video_thwc[t + 1].astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            
            # Simple frame difference
            diff = np.abs(frame2_gray - frame1_gray)
            
            # Apply Gaussian blur to reduce noise
            diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
            
            # Threshold to get motion mask (top 20% of motion)
            threshold = np.percentile(diff_blur, 95)
            motion_mask = diff_blur > threshold
            
            # Get coordinates of motion regions
            y_coords, x_coords = np.where(motion_mask)
            
            if len(x_coords) > 0:
                for i in range(len(x_coords)):
                    all_keypoints.append([t, x_coords[i], y_coords[i]])
        
        if len(all_keypoints) == 0:
            return np.array([], dtype=np.float32).reshape(0, 3)
        
        all_keypoints = np.array(all_keypoints, dtype=np.float32)
        
        # Sample up to num_points
        if len(all_keypoints) > num_points:
            indices = np.random.choice(len(all_keypoints), num_points, replace=False)
            return all_keypoints[indices]
        else:
            return all_keypoints

    def _extract_sift_queries_limited(self, video_thwc: np.ndarray, num_points: int) -> np.ndarray:
        """Extract SIFT keypoints from first half of video with stride 6."""
        T = video_thwc.shape[0]
        T_half = video_thwc.shape[0] // 3
        query_frame_indices = list(range(0, T_half + 1, 4))
        if len(query_frame_indices) == 0:
            query_frame_indices = [0]
        
        all_keypoints = []
        
        for t in query_frame_indices:
            frame_gray = cv2.cvtColor(video_thwc[t].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            kps = self.sift.detect(frame_gray, None)
            
            for kp in kps:
                x, y = kp.pt
                all_keypoints.append([t, x, y])
        
        if len(all_keypoints) == 0:
            return np.array([], dtype=np.float32).reshape(0, 3)
        
        all_keypoints = np.array(all_keypoints, dtype=np.float32)
        
        # Sample up to num_points
        if len(all_keypoints) > num_points:
            indices = np.random.choice(len(all_keypoints), num_points, replace=False)
            return all_keypoints[indices]
        else:
            return all_keypoints

    def _extract_hybrid_queries(self, video_thwc: np.ndarray) -> np.ndarray:
        """
        Extract query points using hybrid approach:
        - N/2 from motion-based sampling (optical flow or frame differencing)
        - N/2 from SIFT keypoints
        - Remaining from random sampling if needed
        """
        H, W = video_thwc.shape[1:3]
        T_half = video_thwc.shape[0] // 3
        query_frame_indices = list(range(0, T_half + 1, 4))
        if len(query_frame_indices) == 0:
            query_frame_indices = [0]

        # Target points per method
        n_motion = self.N // 3
        n_sift = self.N - n_motion
        
        # Extract motion-based points (optical flow or frame differencing)
        motion_queries = self._extract_motion_queries(video_thwc, n_motion)

        # Extract SIFT points
        sift_queries = self._extract_sift_queries_limited(video_thwc, n_sift)
        
        # Combine motion and SIFT queries
        if len(motion_queries) > 0 and len(sift_queries) > 0:
            queries = np.vstack([motion_queries, sift_queries])
        elif len(motion_queries) > 0:
            queries = motion_queries
        elif len(sift_queries) > 0:
            queries = sift_queries
        else:
            queries = np.array([], dtype=np.float32).reshape(0, 3)
        
        # If we have fewer than N points, pad with random points
        if len(queries) < self.N:
            n_random = self.N - len(queries)
            random_queries = np.array([
                [np.random.choice(query_frame_indices),
                 np.random.uniform(10, W - 10),
                 np.random.uniform(10, H - 10)]
                for _ in range(n_random)
            ], dtype=np.float32)
            
            if len(queries) > 0:
                queries = np.vstack([queries, random_queries])
            else:
                queries = random_queries
        
        # If we have more than N points, subsample
        elif len(queries) > self.N:
            indices = np.random.choice(len(queries), self.N, replace=False)
            queries = queries[indices]
        
        return queries  # [N, 3] in (t, x, y) format

    
    def _load_video_frames(self, video_path: str, frame_indices: list) -> list:
        """Load specific frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def __getitem__(self, index: int):
        # Check if this is video mode (S4D or Kinetics)

        # Load from image files
        rgb_paths = self.rgb_paths_per_seq[index]
        S = len(rgb_paths)
        
        # === ADAPTIVE STRIDE AND T CALCULATION ===
        effective_T = self.T
        effective_stride = self.frame_stride
        
        # Calculate required span with current stride
        required_span = effective_T * effective_stride
        
        # If not enough frames, try reducing stride first
        if S < required_span:
            # Calculate minimum stride needed
            effective_stride = max(1, S // effective_T)
            required_span = effective_T * effective_stride
            
            # If still not enough frames after reducing stride to 1, reduce T
            if S < required_span:
                effective_T = S // effective_stride
                effective_T = max(1, effective_T)  # At least 1 frame
                required_span = effective_T * effective_stride
        
        # Ensure required_span doesn't exceed S
        required_span = min(required_span, S)
        
        # Sample frames
        if S <= required_span:
            start = 0
        else:
            start = np.random.randint(0, S - required_span + 1)
        
        # Sample with effective stride up to effective_T frames
        sampled_indices = list(range(start, min(start + required_span, S), effective_stride))[:effective_T]
        sampled_paths = [rgb_paths[i] for i in sampled_indices]
        
        # Load first frame to get crop parameters
        frame0 = cv2.imread(sampled_paths[0], cv2.IMREAD_COLOR)
        if frame0 is None:
            raise FileNotFoundError(f"Failed to read: {sampled_paths[0]}")
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        
        H_orig, W_orig = frame0.shape[:2]
        
        # Sample crop parameters (consistent across clip)
        crop_params = self._random_crop_params(H_orig, W_orig)
        
        # Load and crop all frames (using effective_T)
        H, W = self.image_size
        video_clean = np.empty((effective_T, H, W, 3), dtype=np.float32)
        

        for i, path in enumerate(sampled_paths):
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Failed to read: {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_cropped = self._apply_crop_and_resize(frame, crop_params)
            video_clean[i] = frame_cropped.astype(np.float32)
    
        # Apply photometric augmentation
        video_aug = self._photometric_augment_clip(video_clean.copy())
        
        # Extract SIFT queries from clean video (using effective_T)
        # queries = self._extract_sift_queries(video_clean)
        queries = self._extract_hybrid_queries(video_clean)
        
        # Convert to CHW format
        video_aug_chw = video_aug.transpose(0, 3, 1, 2)  # [effective_T, 3, H, W]
        video_clean_chw = video_clean.transpose(0, 3, 1, 2)  # [effective_T, 3, H, W]
        
        return video_aug_chw, video_clean_chw, queries
