import torch
import numpy as np
import glob
import cv2
from tqdm import tqdm

class EgoPointsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_location, image_size=(384, 512)):
        self.dataset_location = dataset_location
        self.image_size = image_size
        self.annotation_paths = sorted(glob.glob(f"{dataset_location}/*/annot.npz"))
        print(f"Found {len(self.annotation_paths)} sequences")
        self.rgb_paths = [sorted(glob.glob(f"{temp_path.split('/annot.npz')[0]}/rgbs/*.jpg")) for temp_path in self.annotation_paths]

    def __getitem__(self, index):
        rgb_paths = self.rgb_paths[index]
        seq = self.annotation_paths[index].split("/")[-2]
        annotations = np.load(self.annotation_paths[index])
        
        trajs = torch.from_numpy(annotations["trajs_2d"].astype(np.float32))  # (T, N, 2)
        valids = torch.from_numpy(annotations["valids"].astype(np.float32))  # (T, N)
        visibs = torch.from_numpy(annotations["visibs"].astype(np.float32))  # (T, N)
        vis_valids = torch.from_numpy(annotations["vis_valids"].astype(np.float32))  # (T, N)
        out_of_view = torch.from_numpy(annotations["out_of_view"].astype(np.float32))  # (T, N)
        occluded = torch.from_numpy(annotations["occluded"].astype(np.float32))  # (T, N)
        
        # Add batch dimension and permute to (B, T, N, 2)
        trajs = trajs.unsqueeze(0).permute(0, 1, 2, 3)  # (1, T, N, 2)
        valids = valids.unsqueeze(0)  # (1, T, N)
        visibs = visibs.unsqueeze(0)  # (1, T, N)
        vis_valids = vis_valids.unsqueeze(0)  # (1, T, N)
        out_of_view = out_of_view.unsqueeze(0)  # (1, T, N)
        occluded = occluded.unsqueeze(0)  # (1, T, N)
        
        # Load and process video frames
        T = len(rgb_paths)
        H, W = self.image_size
        
        # Get original image size for scaling
        rgb0_bak = cv2.imread(rgb_paths[0])
        H_bak, W_bak = rgb0_bak.shape[:2]
        sy = H / H_bak
        sx = W / W_bak
        
        # Scale trajectories
        trajs[:, :, :, 0] *= sx
        trajs[:, :, :, 1] *= sy
        
        # Load video frames
        rgbs = []
        for rgb_path in rgb_paths:
            rgb_i = cv2.imread(rgb_path)
            rgb_i = rgb_i[:, :, ::-1]  # BGR->RGB
            rgb_i = cv2.resize(rgb_i, (W, H), interpolation=cv2.INTER_LINEAR)
            rgbs.append(torch.from_numpy(rgb_i).permute(2, 0, 1))  # (3, H, W)
        
        video = torch.stack(rgbs, dim=0).float().unsqueeze(0)  # (1, T, 3, H, W)
        
        return {
            'video': video[0],
            'trajectory': trajs[0],
            'visibility': visibs[0],
            'valids': valids[0],
            'vis_valids': vis_valids[0],
            'out_of_view': out_of_view[0],
            'occluded': occluded[0],
            'seq': seq
        }

    def __len__(self):
        return len(self.rgb_paths)
