import math
import numpy as np

import os
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from dataset.mofi_f import Movi_F
from dataset.tapvid import TAPVid

EPS = 1e-6

def get_dataloaders(args):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if args.validation:
        val_dataset = TAPVid(args)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=False)

        return None, val_dataloader

    train_dataset = Movi_F(args)
    val_dataset = TAPVid(args)

    num_workers = 8 # 12
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, rank=args.rank, shuffle=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.bs, # // args.gpus,
        num_workers=num_workers,      # cpu per gpu
        worker_init_fn=seed_worker,
        drop_last=True,
        pin_memory=False,
    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False)

    return train_dataloader, val_dataloader


def get_scheduler(args, optimizer, train_loader):
    T_max = (len(train_loader)) * args.epoch_num
    warmup_steps = int(T_max * 0.01)
    steps = T_max - warmup_steps

    linear_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, total_iters=warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[warmup_steps])

    return scheduler


def restart_from_checkpoint(args, run_variables, **kwargs):

    # checkpoint_path = os.path.join(args.checkpoint_path, "best_checkpoint.pt")
    checkpoint_path = args.checkpoint_path

    assert checkpoint_path is not None
    assert os.path.exists(checkpoint_path)

    # open checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

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


def restart_from_checkpoint_not_dist(args, run_variables, **kwargs):

    checkpoint_path = args.checkpoint_path

    assert checkpoint_path is not None
    assert os.path.exists(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        name = k[7:] # remove `module.`
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
