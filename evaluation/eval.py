import torch
from torch.utils import data

import argparse
import yaml
from argparse import Namespace

from model.trackon_predictor import Predictor
from utils.train_utils import load_args_from_yaml, fix_random_seeds


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation script for Track-On2")
    parser.add_argument('--model_config_path', type=str, required=True, default="./config/test.yaml", help='Path to the model config file')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the evaluation dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model_checkpoint_path', type=str, required=True, help='Path to the model checkpoint file')
    return parser.parse_args()


if __name__ == "__main__":
    eval_args = get_args()
    model_args = load_args_from_yaml(eval_args.model_config_path)

    # set seeds and cudnn for reproducibility
    fix_random_seeds(1234)

    if eval_args.dataset_name in ["davis"]:
        model_args.M_i = 24

    model = Predictor(model_args, checkpoint_path=eval_args.model_checkpoint_path, support_grid_size=20)
    model = model.cuda()

    if eval_args.dataset_name in ["davis", "kinetics", "robotap"]:
        from dataset.tapvid import TAPVid
        from evaluation.evaluator import evaluate_tapvid
        dataset = TAPVid(None, data_root=eval_args.dataset_path, dataset_type=eval_args.dataset_name)
        dataloader = data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=8,
                                        pin_memory=False)
        evaluate_tapvid(model, dataloader, delta_v=model_args.delta_v, print_each=True)

    elif eval_args.dataset_name in ["point_odyssey"]:
        from dataset.point_odyssey import PointOdysseyDataset
        from evaluation.evaluator import evaluate_po
        dataset = PointOdysseyDataset(eval_args.dataset_path, "test", 256)
        dataloader = data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1,
                                        pin_memory=False)
        evaluate_po(model, dataloader, print_each=True)

    elif eval_args.dataset_name in ["dynamic_replica"]:
        from dataset.dynamic_replica import DynamicReplicaDataset
        from evaluation.evaluator import evaluate_dr
        dataset = DynamicReplicaDataset(eval_args.dataset_path, sample_len=300, only_first_n_samples=1)
        dataloader = data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1,
                                        pin_memory=False)
        evaluate_dr(model, dataloader, print_each=True)

    else:
        raise NotImplementedError(f"Dataset {eval_args.dataset_name} not supported for evaluation yet.")