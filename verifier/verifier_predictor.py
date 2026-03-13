import torch
import torch.nn as nn
import random as py_random
from verifier.verifier import Verifier
from verifier.utils import get_ensemble
from collections import OrderedDict


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


class VerifierPredictor(nn.Module):
    def __init__(self, verifier_args, checkpoint_path=None, majority_ratio=0.5):
        super().__init__()

        if verifier_args is None:
            verifier_args = self._default_args()

        self.verifier = Verifier(verifier_args).cuda()
        
        if checkpoint_path is not None:
            self._load_model_and_check(checkpoint_path)
        
        self.verifier.eval()
        self.majority_ratio = majority_ratio
    
    def _default_args(self):
        class Args:
            def __init__(self):
                self.verifier_input_size = [384, 512]
                self.verifier_embedding_dim = 256
                self.verifier_num_layers = 6
                self.local_encoder_num_layers = 3
                self.query_critic = "feature"  # "feature", "none"
                self.T = 24


    def _load_model_and_check(self, checkpoint_path):
        raw = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _strip_module_prefix(_extract_state_dict(raw))

        # Load with strict=False to check for errors
        load_result = self.verifier.load_state_dict(state_dict, strict=False)
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)

        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint (not present in model): {unexpected}")
        
        if missing:
            raise RuntimeError(f"Missing keys in checkpoint: {missing}")

        print(f"Loaded verifier weights from {checkpoint_path}")
        
    @torch.no_grad()
    def forward(self, video, queries, ensemble_tracks, ensemble_visibility, use_amp=False):
        """
        Args:
            video: (B, T, 3, H, W) - input video
            queries: (B, N, 3) - query points in (t, x, y) format
            ensemble_tracks: (B, T, N, M, 2) - predictions from M teacher models
            ensemble_visibility: (B, T, N, M) - visibility predictions from M teacher
            aug: bool - whether to use augmentation in ensemble predictor
            
        Returns:
            selected_tracks: (B, T, N, 2) - selected predictions
            selected_visibility: (B, T, N) - visibility (majority vote)
            selected_uncertainties: (B, T, N) - uncertainty scores (1 - certainty)
        """
        device = video.device

        # Run ensemble predictor (non-random path)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):            
            # Run verifier to get rankings and certainties
            ranks = self.verifier.efficient_forward(
                video.clone(), 
                ensemble_tracks.clone(), 
                queries.clone()
            )
            # ranks: (B, T, N, M) - ranking logits
            # offsets: (B, T, N, M, 2) - offset predictions
            
            # Get verifier's selection (highest ranked prediction)
            selected_tracks = get_ensemble(ensemble_tracks, ranks, max=True)  # (B, T, N, 2)

            # Get majority vote visibility
            selected_visibility = ensemble_visibility.float().mean(dim=-1) > self.majority_ratio  # (B, T, N)
                    
        return selected_tracks, selected_visibility
