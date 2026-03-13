import torch
from torch.utils import data

import argparse

from utils.train_utils import load_args_from_yaml, fix_random_seeds


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation script for benchmarks")
    parser.add_argument('--model_names', nargs='+', required=True, help='List of model names to evaluate')

    parser.add_argument('--trackon_config_path', type=str, default="./config/test.yaml", help='Path to the model config file')
    parser.add_argument('--trackon_checkpoint_path', type=str, help='Path to the model checkpoint file')

    parser.add_argument('--tapir_checkpoint_path', type=str, help='Path to the tapir checkpoint file')
    parser.add_argument('--bootstapir_checkpoint_path', type=str, help='Path to the bootsapir checkpoint file')
    parser.add_argument('--bootstapnext_checkpoint_path', type=str, help='Path to the bootstapnext checkpoint file')
    parser.add_argument('--tapnext_checkpoint_path', type=str, help='Path to the tapnext checkpoint file')
    parser.add_argument('--anthro_locotrack_checkpoint_path', type=str, help='Path to the locotrack checkpoint file')
    parser.add_argument('--locotrack_checkpoint_path', type=str, help='Path to the locotrack checkpoint file')
    parser.add_argument('--alltracker_checkpoint_path', type=str, help='Path to the alltracker checkpoint file')

    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the evaluation dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')

    parser.add_argument('--verifier_config_path', type=str, help='Path to the verifier config file')
    parser.add_argument('--verifier_checkpoint_path', type=str, help='Path to the verifier checkpoint file')

    parser.add_argument('--cache_predictions', action='store_true', help='Whether to cache predictions for faster re-evaluation')
    parser.add_argument('--cache_dir', type=str, default="./cached_predictions", help='Directory to store cached predictions')

    return parser.parse_args()


def prepare_models(eval_args):
    models = {}
    verifier_model = None
    
    # Load all requested models
    model_names = eval_args.model_names
    for model_name in model_names:
        if model_name == "verifier":
            if eval_args.verifier_checkpoint_path is None:
                raise ValueError("--verifier_checkpoint_path is required for verifier")
            
            if len(model_names) == 1:
                raise ValueError("At least one other model must be evaluated if verifier is being evaluated")
            
            verifier_args = load_args_from_yaml(eval_args.verifier_config_path)
            from verifier.verifier_predictor import VerifierPredictor
            verifier_model = VerifierPredictor(verifier_args, checkpoint_path=eval_args.verifier_checkpoint_path)
            verifier_model = verifier_model.cuda()
            print(f"Loaded Verifier model")
            # Continue to load other models

        elif model_name == "trackon2":
            if eval_args.trackon_checkpoint_path is None:
                raise ValueError("--trackon_checkpoint_path is required for trackon2")
            
            from model.trackon_predictor import Predictor as Trackon_Predictor

            if eval_args.trackon_config_path is not None:
                trackon2_args = load_args_from_yaml(eval_args.trackon_config_path)
                if eval_args.dataset_name == "davis":
                    trackon2_args.M_i = 24
            else:
                trackon2_args = None

            model = Trackon_Predictor(trackon2_args, checkpoint_path=eval_args.trackon_checkpoint_path, support_grid_size=20)
            model = model.cuda()
            models[model_name] = model
            print(f"Loaded Trackon2 model")
        
        elif model_name == "tapnext":
            if eval_args.tapnext_checkpoint_path is None:
                raise ValueError("--tapnext_checkpoint_path is required for tapnext")
            
            from ensemble.tapnext.tapnext_predictor import TAPNextPredictor
            model = TAPNextPredictor(eval_args.tapnext_checkpoint_path).cuda()
            models[model_name] = model
            print(f"Loaded TAP-Next model")

        elif model_name == "bootstapnext":
            if eval_args.bootstapnext_checkpoint_path is None:
                raise ValueError("--bootstapnext_checkpoint_path is required for bootstapnext")
            
            from ensemble.tapnext.tapnext_predictor import TAPNextPredictor
            model = TAPNextPredictor(eval_args.bootstapnext_checkpoint_path).cuda()
            models[model_name] = model
            print(f"Loaded TAP-Next model") 

        elif model_name == "tapir":
            if eval_args.tapir_checkpoint_path is None:
                raise ValueError("--tapir_checkpoint_path is required for tapir")
            
            from ensemble.bootstapir.bootstapir_predictor import TAPIRPredictor
            model = TAPIRPredictor(eval_args.tapir_checkpoint_path).cuda()
            models[model_name] = model
            print(f"Loaded TAPIR model")

        elif model_name == "bootstapir":
            if eval_args.bootstapir_checkpoint_path is None:
                raise ValueError("--bootstapir_checkpoint_path is required for bootstapir")
            
            from ensemble.bootstapir.bootstapir_predictor import TAPIRPredictor
            model = TAPIRPredictor(eval_args.bootstapir_checkpoint_path).cuda()
            models[model_name] = model
            print(f"Loaded BootsTAPIR model")

        elif model_name == "cotracker3_video":
            from ensemble.cotracker import CoTracker_Predictor
            model = CoTracker_Predictor(windowed=False).cuda()
            models[model_name] = model
            print(f"Loaded CoTracker3 Video model")

        elif model_name == "cotracker3_window":
            from ensemble.cotracker import CoTracker_Predictor
            model = CoTracker_Predictor(windowed=True).cuda()
            models[model_name] = model
            print(f"Loaded CoTracker3 Windowed model")

        elif model_name == "locotrack":
            if eval_args.locotrack_checkpoint_path is None:
                raise ValueError("--locotrack_checkpoint_path is required for locotrack")
            from ensemble.locotrack.locotrack_predictor import LocoTrackPredictor
            model = LocoTrackPredictor(eval_args.locotrack_checkpoint_path).cuda()
            models[model_name] = model
            print(f"Loaded LocoTrack model")

        elif model_name == "anthro_locotrack":
            if eval_args.anthro_locotrack_checkpoint_path is None:
                raise ValueError("--anthro_locotrack_checkpoint_path is required for anthro_locotrack")
            from ensemble.locotrack.locotrack_predictor import LocoTrackPredictor
            model = LocoTrackPredictor(eval_args.anthro_locotrack_checkpoint_path).cuda()
            models[model_name] = model
            print(f"Loaded Anthro LocoTrack model")
        
        elif model_name == "alltracker":
            from ensemble.alltracker.alltracker_predictor import AllTrackerPredictor
            model = AllTrackerPredictor(checkpoint_path=eval_args.alltracker_checkpoint_path).cuda()
            models[model_name] = model
            print(f"Loaded AllTracker model")
        
        elif model_name == "oracle":
            # Oracle is computed inside the evaluator from the other models' predictions.
            # Use None as a sentinel so the evaluator knows to compute it.
            models['oracle'] = None
            print("Oracle upper bound will be computed during evaluation")

        elif model_name == "random":
            # Random (round-robin) baseline: for sample i, selects model z = i % M.
            # Computed inside the evaluator from the other models' predictions.
            models['random'] = None
            print("Random (round-robin) baseline will be computed during evaluation")

        else:
            raise ValueError(f"Model name {model_name} is not recognized. Supported models are: trackon2, bootstapnext, bootstapir, cotracker3_video, cotracker3_window, locotrack, anthro_locotrack, alltracker, verifier, oracle, random.")
    
    # Add verifier to models dict if it was loaded (it will be evaluated last)
    if verifier_model is not None:
        models['verifier'] = verifier_model
    
    return models

if __name__ == "__main__":
    eval_args = get_args()
    fix_random_seeds(1234)
        
    models = prepare_models(eval_args)

    if eval_args.dataset_name in ["davis", "kinetics", "robotap"]:
        from dataset.tapvid import TAPVid
        from evaluation.evaluator import evaluate_tapvid
        dataset = TAPVid(None, data_root=eval_args.dataset_path, dataset_type=eval_args.dataset_name)
        dataloader = data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=8,
                                        pin_memory=False)
        evaluate_tapvid(models, dataloader, print_each=True, 
                       cache_predictions=eval_args.cache_predictions,
                       cache_dir=eval_args.cache_dir,
                       dataset_name=eval_args.dataset_name)

    elif eval_args.dataset_name == "point_odyssey":
        from dataset.point_odyssey import PointOdysseyDataset
        from evaluation.evaluator import evaluate_po
        dataset = PointOdysseyDataset(eval_args.dataset_path, "test", 256)
        dataloader = data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1,
                                        pin_memory=False)
        evaluate_po(models, dataloader, print_each=True,
                   cache_predictions=eval_args.cache_predictions,
                   cache_dir=eval_args.cache_dir,
                   dataset_name=eval_args.dataset_name)

    elif eval_args.dataset_name == "dynamic_replica":
        from dataset.dynamic_replica import DynamicReplicaDataset
        from evaluation.evaluator import evaluate_dr
        dataset = DynamicReplicaDataset(eval_args.dataset_path, sample_len=300, only_first_n_samples=1)
        dataloader = data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1,
                                        pin_memory=False)
        evaluate_dr(models, dataloader, print_each=True,
                   cache_predictions=eval_args.cache_predictions,
                   cache_dir=eval_args.cache_dir,
                   dataset_name=eval_args.dataset_name)
    
    elif eval_args.dataset_name == "ego_points":
        from dataset.ego_points import EgoPointsDataset
        from evaluation.evaluator import evaluate_ego_points
        dataset = EgoPointsDataset(eval_args.dataset_path)
        dataloader = data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1,
                                        pin_memory=False)
        evaluate_ego_points(models, dataloader, print_each=True,
                           cache_predictions=eval_args.cache_predictions,
                           cache_dir=eval_args.cache_dir,
                           dataset_name=eval_args.dataset_name)

    else:
        raise NotImplementedError(f"Dataset {eval_args.dataset_name} not supported for evaluation yet.")