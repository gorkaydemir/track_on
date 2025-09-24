# ==== Below is based on https://github.com/facebookresearch/co-tracker/blob/main/cotracker/datasets/tap_vid_datasets.py ====
import os
import io
from glob import glob
import torch
import pickle
import numpy as np
import mediapy as media

from PIL import Image
from typing import Mapping, Tuple, Union


DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]

def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:

    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:

    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


class TAPVid(torch.utils.data.Dataset):
    def __init__(self, args, data_root=None, dataset_type=None):

        if data_root is None:
            data_root = args.tapvid_root
        else:
            self.data_root = data_root

        if dataset_type is None:
            self.dataset_type = args.eval_dataset
        else:
            self.dataset_type = dataset_type
        
        self.resize_to_256 = True
        self.queried_first = True

        if "kinetics" in self.dataset_type:
            all_paths = glob(os.path.join(data_root, "*_of_0010.pkl"))
            points_dataset = []
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    points_dataset = points_dataset + data
            self.points_dataset = points_dataset
            
        elif "davis" in self.dataset_type:
            with open(data_root, "rb") as f:
                self.points_dataset = pickle.load(f)
            if "davis" in self.dataset_type:
                self.video_names = sorted(list(self.points_dataset.keys()))

        elif "rgb_stacking" in self.dataset_type:
            with open(data_root, "rb") as f:
                self.points_dataset = pickle.load(f)

        elif "robotap" in self.dataset_type:
            all_paths = glob(os.path.join(data_root, "robotap_split*.pkl"))
            points_dataset = []
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = list(pickle.load(f).values())
                    points_dataset = points_dataset + data
            self.points_dataset = points_dataset

        # print("found %d unique videos in %s" % (len(self.points_dataset), data_root))

    def __getitem__(self, index):
        # :return rgbs: (T, 3, 256, 256)       | in range [0, 255]
        # :return trajs: (T, N, 2)             | in range [0, 256-1]
        # :return visibles: (T, N)             | Boolean
        # :return query_points: (N, 3)         | in format (t, y, x), in range [0, 256-1]

        if "davis" in self.dataset_type:
            video_name = self.video_names[index]
        else:
            video_name = index
        video = self.points_dataset[video_name]
        frames = video["video"].copy()

        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])

        target_points = self.points_dataset[video_name]["points"].copy()
        if self.resize_to_256:
            frames = resize_video(frames, [256, 256])
            # target_points *= np.array([255, 255])  # 1 should be mapped to 256-1
            target_points *= np.array([256, 256])  # 1 should be mapped to 256
        else:
            # target_points *= np.array([frames.shape[2] - 1, frames.shape[1] - 1])
            target_points *= np.array([frames.shape[2], frames.shape[1]])

        target_occ = self.points_dataset[video_name]["occluded"].copy()
        if self.queried_first:
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        trajs = torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float()  # T, N, D

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[0].permute(
            1, 0
        )  # T, N
        query_points = torch.from_numpy(converted["query_points"])[0]  # T, N
        return rgbs, trajs, visibles, query_points

    def __len__(self):
        return len(self.points_dataset)
    