import sys
import os
import argparse
from pathlib import Path

import torch

def print_args(args):
    if args.validation:
        print("====== Validation ======")
        print(f"Evaluating on: {args.eval_dataset}")
        print(f"Top-K regions: {args.top_k_regions}")
        print(f"Memory size: {args.val_memory_size}")
        print(f"Visibility Delta: {args.val_vis_delta}")

    elif args.online_validation:
        print("====== Online Validation ======")
    else:
        print("====== Training ======")
        print(f"Evaluating on: {args.eval_dataset}")
        print()

        print(f"Input size: {args.input_size}")
        print(f"N: {args.N}")
        print(f"T: {args.T}")
        print(f"Stride: {args.stride}")
        print(f"Transformer embedding dim: {args.transformer_embedding_dim}")
        print(f"CNN for correlation: {args.cnn_corr}")
        print(f"Linear visibility: {args.linear_visibility}")
        print()

        print(f"Number of layers in query decoder: {args.num_layers}")
        print(f"Number of layers in offset head: {args.num_layers_offset_head}")
        print()

        print(f"Number of layers in deformable rerank head: {args.num_layers_rerank}")
        print(f"Number of layers of fusion layer in deformable rerank head: {args.num_layers_rerank_fusion}")
        print(f"Top-K regions: {args.top_k_regions}")
        print()

        print(f"Number of layers in spatial memory writer: {args.num_layers_spatial_writer}")
        print(f"Number of layers in spatial memory self attention (over memory): {args.num_layers_spatial_self}")
        print(f"Number of layers in spatial memory cross attention: {args.num_layers_spatial_cross}")
        print()

        print(f"Memory size: {args.memory_size}")
        print(f"Random memory mask drop: {args.random_memory_mask_drop}")
        print()

        print(f"Patch classification CE loss: {args.lambda_point}")
        print(f"Visibility BCE loss: {args.lambda_vis}")
        print(f"Offset prediction L1 loss: {args.lambda_offset}")
        print(f"Uncertainty BCE loss: {args.lambda_uncertainty}")
        print(f"Top-K Uncertainty BCE loss: {args.lambda_top_k}")
        print()

        print(f"Epochs: {args.epoch_num}")
        print(f"Learning rate: {args.lr}")
        print(f"Weight decay: {args.wd}")
        print(f"Batch size per GPU: {args.bs}")
        print(f"Using AMP: {args.amp}")
        print(f"Loss after query: {args.loss_after_query}")


    print("====== ======= ======\n")

def get_args():
    parser = argparse.ArgumentParser("Track-On")

    # === Data Related Parameters ===
    parser.add_argument('--movi_f_root', type=str, default=None)
    parser.add_argument('--tapvid_root', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, choices=["davis", "rgb_stacking", "kinetics", "robotap"], default="davis")
    parser.add_argument('--augmentation', action="store_true")
    parser.add_argument('--input_size', type=int, nargs=2, default=[384, 512])

    # === Model Related Parameters ===
    parser.add_argument('--N', type=int, default=480)
    parser.add_argument('--T', type=int, default=24)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--transformer_embedding_dim', type=int, default=256)
    parser.add_argument('--cnn_corr', action="store_true")
    parser.add_argument('--linear_visibility', action="store_true")


    # === Main Transformer ===
    parser.add_argument('--num_layers', type=int, default=3, help="Number of layers in query decoder")
    parser.add_argument('--num_layers_offset_head', type=int, default=3, help="Number of layers in offset head")

    # === Rerank Module ===
    parser.add_argument('--num_layers_rerank', type=int, default=3, help="Number of layers in deformable rerank head")
    parser.add_argument('--num_layers_rerank_fusion', type=int, default=1, help="Number of layers of fusion layer in deformable rerank head")
    parser.add_argument('--top_k_regions', type=int, default=16)

    # === Spatial Memory ===
    parser.add_argument('--num_layers_spatial_writer', type=int, default=3, help="Number of layers in spatial memory writer")
    parser.add_argument('--num_layers_spatial_self', type=int, default=1, help="Number of layers in spatial memory self attention (over memory)")
    parser.add_argument('--num_layers_spatial_cross', type=int, default=1, help="Number of layers in spatial memory cross attention")

    # === Memory Parameters ===
    parser.add_argument('--memory_size', type=int, default=12, help="Number of memory slots in training")
    parser.add_argument('--val_memory_size', type=int, default=48)
    parser.add_argument('--val_vis_delta', type=float, default=0.8)
    parser.add_argument('--random_memory_mask_drop', type=float, default=0.1, help="Randomly drop memory slots")

    # === Loss Related Parameters ===
    parser.add_argument('--lambda_point', type=float, default=3, help="Patch classification CE loss")
    parser.add_argument('--lambda_vis', type=float, default=1, help="Visibility BCE loss")
    parser.add_argument('--lambda_offset', type=float, default=1, help="Offset prediction L1 loss")
    parser.add_argument('--lambda_uncertainty', type=float, default=1, help="Uncertainty BCE loss")
    parser.add_argument('--lambda_top_k', type=float, default=1, help="Top-K Uncertainty BCE loss")


    # === Training Related Parameters ===
    parser.add_argument('--epoch_num', type=int, default=150, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--wd', type=float, default=1e-5, help="Weight decay")
    parser.add_argument('--bs', type=int, default=16, help="Batch size per GPU")
    parser.add_argument('--amp', action="store_true")
    parser.add_argument('--loss_after_query', action="store_true")

    # === Misc ===
    parser.add_argument('--validation', action="store_true")
    parser.add_argument('--online_validation', action="store_true")
    parser.add_argument('--model_save_path', type=str, default=None)  # saves under checkpoints folder
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    # === Extra settings ===
    args.gpus = torch.cuda.device_count()

    if args.online_validation:
        args.validation = True

    if not args.validation:
        assert args.model_save_path is not None, "Model save path is not provided"
        args.model_save_path = os.path.join("checkpoints", args.model_save_path)
        Path(args.model_save_path).mkdir(parents=True, exist_ok=True)

    # === Set Memory Sizes for Validation ===
    if args.validation:
        args.val_vis_delta = 0.8
        
        if args.eval_dataset in ["davis", "robotap"]:
            args.val_memory_size = 48

        elif args.eval_dataset == "kinetics":
            args.val_memory_size = 96

        elif args.eval_dataset == "rgb_stacking":
            args.val_memory_size = 80
            args.val_vis_delta = 0.5

    return args