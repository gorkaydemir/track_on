import os
import gzip
import torch
import numpy as np
import torch.utils.data as data
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple

import json
import dataclasses
import numpy as np
from dataclasses import Field, MISSING
from typing import IO, TypeVar, Type, get_args, get_origin, Union, Any, Tuple

_X = TypeVar("_X")

# This code is based on the previous versions of the CoTracker repository:
# For reference, check: https://github.com/facebookresearch/co-tracker/blob/3716e362497e15e4fb8ec46898dcfd8afbca89e3/cotracker/datasets/dr_dataset.py


def load_dataclass(f: IO, cls: Type[_X], binary: bool = False) -> _X:
    if binary:
        asdict = json.loads(f.read().decode("utf8"))
    else:
        asdict = json.load(f)

    # in the list case, run a faster "vectorized" version
    cls = get_args(cls)[0]
    res = list(_dataclass_list_from_dict_list(asdict, cls))

    return res

def _resolve_optional(type_: Any) -> Tuple[bool, Any]:
    if get_origin(type_) is Union:
        args = get_args(type_)
        if len(args) == 2 and args[1] == type(None):  # noqa E721
            return True, args[0]
    if type_ is Any:
        return True, Any

    return False, type_

def _unwrap_type(tp):
    # strips Optional wrapper, if any
    if get_origin(tp) is Union:
        args = get_args(tp)
        if len(args) == 2 and any(a is type(None) for a in args):  # noqa: E721
            # this is typing.Optional
            return args[0] if args[1] is type(None) else args[1]  # noqa: E721
    return tp


def _dataclass_list_from_dict_list(dlist, typeannot):
    cls = get_origin(typeannot) or typeannot

    if typeannot is Any:
        return dlist
    if all(obj is None for obj in dlist):  # 1st recursion base: all None nodes
        return dlist
    if any(obj is None for obj in dlist):
        # filter out Nones and recurse on the resulting list
        idx_notnone = [(i, obj) for i, obj in enumerate(dlist) if obj is not None]
        idx, notnone = zip(*idx_notnone)
        converted = _dataclass_list_from_dict_list(notnone, typeannot)
        res = [None] * len(dlist)
        for i, obj in zip(idx, converted):
            res[i] = obj
        return res

    is_optional, contained_type = _resolve_optional(typeannot)
    if is_optional:
        return _dataclass_list_from_dict_list(dlist, contained_type)

    # otherwise, we dispatch by the type of the provided annotation to convert to
    if issubclass(cls, tuple) and hasattr(cls, "_fields"):  # namedtuple
        # For namedtuple, call the function recursively on the lists of corresponding keys
        types = cls.__annotations__.values()
        dlist_T = zip(*dlist)
        res_T = [
            _dataclass_list_from_dict_list(key_list, tp)
            for key_list, tp in zip(dlist_T, types)
        ]
        return [cls(*converted_as_tuple) for converted_as_tuple in zip(*res_T)]
    elif issubclass(cls, (list, tuple)):
        # For list/tuple, call the function recursively on the lists of corresponding positions
        types = get_args(typeannot)
        if len(types) == 1:  # probably List; replicate for all items
            types = types * len(dlist[0])
        dlist_T = zip(*dlist)
        res_T = (
            _dataclass_list_from_dict_list(pos_list, tp)
            for pos_list, tp in zip(dlist_T, types)
        )
        if issubclass(cls, tuple):
            return list(zip(*res_T))
        else:
            return [cls(converted_as_tuple) for converted_as_tuple in zip(*res_T)]
    elif issubclass(cls, dict):
        # For the dictionary, call the function recursively on concatenated keys and vertices
        key_t, val_t = get_args(typeannot)
        all_keys_res = _dataclass_list_from_dict_list(
            [k for obj in dlist for k in obj.keys()], key_t
        )
        all_vals_res = _dataclass_list_from_dict_list(
            [k for obj in dlist for k in obj.values()], val_t
        )
        indices = np.cumsum([len(obj) for obj in dlist])
        assert indices[-1] == len(all_keys_res)

        keys = np.split(list(all_keys_res), indices[:-1])
        all_vals_res_iter = iter(all_vals_res)
        return [cls(zip(k, all_vals_res_iter)) for k in keys]
    elif not dataclasses.is_dataclass(typeannot):
        return dlist

    # dataclass node: 2nd recursion base; call the function recursively on the lists
    # of the corresponding fields
    assert dataclasses.is_dataclass(cls)
    fieldtypes = {
        f.name: (_unwrap_type(f.type), _get_dataclass_field_default(f))
        for f in dataclasses.fields(typeannot)
    }

    # NOTE the default object is shared here
    key_lists = (
        _dataclass_list_from_dict_list([obj.get(k, default) for obj in dlist], type_)
        for k, (type_, default) in fieldtypes.items()
    )
    transposed = zip(*key_lists)
    return [cls(*vals_as_tuple) for vals_as_tuple in transposed]

@dataclass
class ImageAnnotation:
    # path to jpg file, relative w.r.t. dataset_root
    path: str
    # H x W
    size: Tuple[int, int]


@dataclass
class DynamicReplicaFrameAnnotation:
    """A dataclass used to load annotations from json."""

    # can be used to join with `SequenceAnnotation`
    sequence_name: str
    # 0-based, continuous frame number within sequence
    frame_number: int
    # timestamp in seconds from the video start
    frame_timestamp: float

    image: ImageAnnotation
    meta: Optional[Dict[str, Any]] = None

    camera_name: Optional[str] = None
    trajectories: Optional[str] = None

def _get_dataclass_field_default(field: Field) -> Any:
    if field.default_factory is not MISSING:
        # pyre-fixme[29]: `Union[dataclasses._MISSING_TYPE,
        #  dataclasses._DefaultFactory[typing.Any]]` is not a function.
        return field.default_factory()
    elif field.default is not MISSING:
        return field.default
    else:
        return None


class DynamicReplicaDataset(data.Dataset):
    def __init__(
        self,
        root,
        split="valid",
        traj_per_sample=256,
        crop_size=None,
        sample_len=-1,
        only_first_n_samples=-1,
        rgbd_input=False,
    ):
        super(DynamicReplicaDataset, self).__init__()
        self.root = root
        self.sample_len = sample_len
        self.split = split
        self.traj_per_sample = traj_per_sample
        self.rgbd_input = rgbd_input
        self.crop_size = crop_size
        frame_annotations_file = f"frame_annotations_{split}.jgz"
        self.sample_list = []
        with gzip.open(
            os.path.join(root, split, frame_annotations_file), "rt", encoding="utf8"
        ) as zipfile:
            frame_annots_list = load_dataclass(
                zipfile, List[DynamicReplicaFrameAnnotation]
            )
        seq_annot = defaultdict(list)
        for frame_annot in frame_annots_list:
            if frame_annot.camera_name == "left":
                seq_annot[frame_annot.sequence_name].append(frame_annot)

        for seq_name in seq_annot.keys():
            seq_len = len(seq_annot[seq_name])

            step = self.sample_len if self.sample_len > 0 else seq_len
            counter = 0

            for ref_idx in range(0, seq_len, step):
                sample = seq_annot[seq_name][ref_idx : ref_idx + step]
                self.sample_list.append(sample)
                counter += 1
                if only_first_n_samples > 0 and counter >= only_first_n_samples:
                    break

    def __len__(self):
        return len(self.sample_list)

    def crop(self, rgbs, trajs):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        H_new = H
        W_new = W

        # simple random crop
        y0 = 0 if self.crop_size[0] >= H_new else (H_new - self.crop_size[0]) // 2
        x0 = 0 if self.crop_size[1] >= W_new else (W_new - self.crop_size[1]) // 2
        rgbs = [
            rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            for rgb in rgbs
        ]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        return rgbs, trajs

    def __getitem__(self, index):
        sample = self.sample_list[index]
        T = len(sample)
        rgbs, visibilities, traj_2d = [], [], []

        H, W = sample[0].image.size
        image_size = (H, W)

        for i in range(T):
            traj_path = os.path.join(
                self.root, self.split, sample[i].trajectories["path"]
            )
            traj = torch.load(traj_path)

            visibilities.append(traj["verts_inds_vis"].numpy())

            rgbs.append(traj["img"].numpy())
            traj_2d.append(traj["traj_2d"].numpy()[..., :2])

        traj_2d = np.stack(traj_2d)
        visibility = np.stack(visibilities)
        T, N, D = traj_2d.shape
        # subsample trajectories for augmentations

        # visible_inds_sampled = torch.randperm(N)[: self.traj_per_sample]
        # instead of random subsampling, we do a uniform subsampling
        if N >= self.traj_per_sample:
            step = N // self.traj_per_sample
            visible_inds_sampled = np.arange(0, N, step)[: self.traj_per_sample]
        else:
            visible_inds_sampled = np.arange(0, N)
        
        traj_2d = traj_2d[:, visible_inds_sampled]
        visibility = visibility[:, visible_inds_sampled]

        if self.crop_size is not None:
            rgbs, traj_2d = self.crop(rgbs, traj_2d)
            H, W, _ = rgbs[0].shape
            image_size = self.crop_size

        visibility[traj_2d[:, :, 0] > image_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > image_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False

        # filter out points that're visible for less than 10 frames
        visible_inds_resampled = visibility.sum(0) > 10
        traj_2d = torch.from_numpy(traj_2d[:, visible_inds_resampled])
        visibility = torch.from_numpy(visibility[:, visible_inds_resampled])

        rgbs = np.stack(rgbs, 0)
        video = torch.from_numpy(rgbs).reshape(T, H, W, 3).permute(0, 3, 1, 2).float()
        return video, traj_2d, visibility

EPS = 1e-6

def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (EPS + denom)
    return mean

