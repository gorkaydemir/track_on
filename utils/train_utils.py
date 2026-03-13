import math
import numpy as np

import os
import sys
import random

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import yaml
from argparse import Namespace

from dataset.movi_f import Movi_F_Large
from dataset.tapvid import TAPVid
from dataset.real_world_dataset import Real_World_Dataset
from dataset.epic_k import EpicK
from dataset.syn_real_dataset import SynRealDataset
from datetime import timedelta


EPS = 1e-6

def load_args_from_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    args = Namespace(**cfg)
    return args

def get_dataloaders(args, train_set="kubric"):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if train_set == "kubric":
        train_dataset = Movi_F_Large(args)
    elif train_set == "epic_k":
        train_dataset = EpicK(args.epic_k_path, T=24, N=args.N, enable_aug=True)
    elif train_set == "real":
        real_dataset = Real_World_Dataset(args.rw_dataset_path,
                                          T=args.T,
                                          N=args.N,
                                          use_ovis=args.use_ovis,
                                          use_vspw=args.use_vspw,
                                          frame_rate=args.frame_sample_ratio)
        if getattr(args, "syn_real_training", False):
            syn_dataset   = Movi_F_Large(args)
            train_dataset = SynRealDataset(real_dataset, syn_dataset)
            print(f"SynReal training: {len(real_dataset)} real + {len(syn_dataset)} synthetic = {len(train_dataset)} total samples")
        else:
            # Wrap in SynRealDataset (real-only) so the training loop always
            # receives the unified 6-tuple format.
            train_dataset = SynRealDataset(real_dataset, syn_dataset=None)
            print(f"Real-only training: {len(real_dataset)} samples")

    val_dataset = TAPVid(args)

    # (#cpus * #gpus) in a node:
    num_workers = 8 * 4                     
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, 
                                                        rank=args.rank, 
                                                        shuffle=True, 
                                                        seed=args.seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.bs,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=False)

    return train_dataloader, val_dataloader


def get_scheduler(args, optimizer, train_loader, constant=False, warmup=False):
    if constant:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epoch_num)

    elif warmup:
        T_max = (len(train_loader)) * args.epoch_num
        warmup_steps = int(T_max * 0.01)
        steps = T_max - warmup_steps

        linear_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, total_iters=warmup_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[warmup_steps])
    else:
        T_max = (len(train_loader)) * args.epoch_num
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    return scheduler


def restart_from_checkpoint(remove_module, checkpoint_path, run_variables, **kwargs):
    # open checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if remove_module:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            name = k[7:]
            new_state_dict[name] = v

        checkpoint["model"] = new_state_dict

    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint with msg {}".format(key, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint".format(key))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint".format(key))
        else:
            print("=> key '{}' not found in checkpoint".format(key))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
                

# === Loading pretrained Track-On2 checkpoint ===

ALLOWED_MISSING_PREFIXES = (
    "backbone.vit_encoder.dinov2",
    "backbone.vit_encoder.dinov3",
)

def _strip_module_prefix(state_dict: dict) -> OrderedDict:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return OrderedDict((k[len("module."):], v) if k.startswith("module.") else (k, v)
                       for k, v in state_dict.items())

def _extract_state_dict(obj) -> dict:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state", "ema_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    return obj

def load_pretrained_weights(model, checkpoint_path):
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(raw))

    # Try load with strict=False, then validate missing keys
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)

    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint (not present in model): {unexpected}")

    disallowed_missing = [
        k for k in missing
        if not any(k.startswith(pfx) for pfx in ALLOWED_MISSING_PREFIXES)
    ]

    if disallowed_missing:
        raise RuntimeError(
            "Checkpoint is missing parameters outside the allowed encoders.\n"
            f"Disallowed missing keys:\n  {disallowed_missing}\n"
            f"(Allowed missing prefixes: {ALLOWED_MISSING_PREFIXES})")

    print(f"Loaded model weights from {checkpoint_path}")
    if missing:
        print(f"Info: missing (allowed) weights: {len(missing)} keys under {set(p.split('.')[0] + '.' + p.split('.')[1] + '.' + p.split('.')[2] for p in missing)}")



# ===  Distributed Settings ===
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L467C1-L499C42

    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        if args.rank == 0:
            print(f"First condition: {args.rank}, {args.world_size}, {args.gpu}")

    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NPROCS'])   # 16 or 4
        args.gpu = args.rank % torch.cuda.device_count()
        if args.rank == 0:
            print(f"Second condition: {args.rank}, {args.world_size}, {args.gpu}")


    # launched naively with `python train.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        if args.rank == 0:
            print(f"Third condition: {args.rank}, {args.world_size}, {args.gpu}")
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(seconds=3000)
    )

    print('| distributed init (rank {}): {}'.format(args.rank, 'env://'), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


    print(f"World size: {args.world_size}")
    print(f"Device count: {torch.cuda.device_count()}")


def setup_for_distributed(is_master):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L452C1-L464C30
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def fix_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
