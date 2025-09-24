import sys
import os
import argparse
from pathlib import Path

import torch

def get_args():
    parser = argparse.ArgumentParser("Track-On2")

    # === Data Related Parameters ===
    parser.add_argument('--movi_f_root', type=str, default=None)
    parser.add_argument('--tapvid_root', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, choices=["davis", "rgb_stacking", "kinetics", "robotap"], default="davis")

    parser.add_argument('--frame_sampling', type=str, choices=["uniform", "random", "uniform_mixed"], default="uniform")
    parser.add_argument('--frame_sample_ratio', type=float, default=1.0, help="Ratio of frames to sample from the video")

    # === Training Related Parameters ===
    parser.add_argument('--input_size', type=int, nargs=2, default=[384, 512])
    parser.add_argument('--N', type=int, default=384, help="Number of tracks in a training video")
    parser.add_argument('--T', type=int, default=48, help="Number of frames in a training video")
    
    # === Model Related Parameters ===
    parser.add_argument('--M', type=int, default=24, help="Memory size")
    parser.add_argument('--D', type=int, default=256, help="Feature dimension")
    parser.add_argument('--K', type=int, default=16, help="Top-K regions for re-ranking")
    parser.add_argument('--decoder_layer_num', type=int, default=3, help="Number of layers in query/point decoder")
    parser.add_argument('--predicton_head_layer_num', type=int, default=3, help="Number of layers in prediction head")
    parser.add_argument('--rerank_layer_num', type=int, default=3, help="Number of layers in reranking")

    parser.add_argument('--vit_backbone', type=str, choices=["dinov2_s", "dinov2_b", "dinov3_s", "dinov3_s_plus", "dinov3_b"], default="dinov2_s", help="Type of ViT backbone of ViT Adapter")
    parser.add_argument('--vit_upsample_factor', type=float, default=1.0, help="Upsample factor for ViT inputs")

    # === Inference Related Parameters ===
    parser.add_argument('--M_i', type=int, default=48, help="Inference time memory size")
    parser.add_argument('--delta_v', type=float, default=0.8, help="Visibility threshold for inference")

    # === Training Related Parameters ===
    parser.add_argument('--lambda_patch_cls', type=float, default=3, help="Patch classification CE loss")

    # === Misc ===
    parser.add_argument('--evaluation', action="store_true")
    parser.add_argument('--epoch_num', type=int, default=150, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--wd', type=float, default=1e-5, help="Weight decay")
    parser.add_argument('--bs', type=int, default=16, help="Batch size per GPU")
    parser.add_argument('--grad_checkpoint', type=bool, default=False, help="Use gradient checkpointing to save memory")
    parser.add_argument('--amp', type=bool, default=True, help="Use mixed precision training")
    parser.add_argument('--model_save_path', required=True, type=str, help="Path to save the model")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to load the checkpoint from")
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    # === Extra settings ===
    args.gpus = torch.cuda.device_count()
    Path(args.model_save_path).mkdir(parents=True, exist_ok=True)

    return args
