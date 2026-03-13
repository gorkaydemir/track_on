import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn.functional as F

import warnings

class Ensemble_Predictor(nn.Module):
    def __init__(self, teacher_models_dict):
        super().__init__()

        self.teacher_models_dict = nn.ModuleDict(teacher_models_dict)
        self.with_verifier = "verifier" in teacher_models_dict

    def teacher_forward(self, video, queries, model_idx):
        """
        Forward through a specific teacher model.
        
        Args:
            video: (B, T, 3, H, W)
            queries: (B, N, 3)
            model_idx: int, index of the teacher model to use
            
        Returns:
            pred_trajectory: (B, T, N, 2)
            pred_visibility: (B, T, N)
        """
        device = video.device
        teacher_model = self.teacher_models_dict[list(self.teacher_models_dict.keys())[model_idx]]
        pred_trajectory, pred_visibility = teacher_model(video, queries)
        # to the initial device
        pred_trajectory = pred_trajectory.to(device)
        pred_visibility = pred_visibility.to(device)

        return pred_trajectory, pred_visibility
    
    def forward(self, video, queries, only_verifier=False, use_amp=False):
        """
        Args:
            video: (B, T, 3, H, W)
            queries: (B, N, 3)

            Returns:
            output: dict containing:
                - "ensemble": (ensemble_tracks, ensemble_visibility) if only_verifier is False
                - "verifier": (verifier_tracks, verifier_visibility) if verifier is available
        """

        if only_verifier and not self.with_verifier:
            raise ValueError("Ensemble predictor is not initialized with a verifier model, but only_verifier is set to True.")

        ensemble_tracks = []
        ensemble_visibility = []
        
        for model_name, model in self.teacher_models_dict.items():
            if model_name == "verifier":
                continue
            
            # IMPORTANT NOTE: I realized that locotrack models sometimes produce NaN values in their predictions when using mixed precision (float16)
            #                 Turn off amp for locotrack models, even if you want to use it for other models.
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and ("loco" not in model_name)), torch.no_grad():
                pred_trajectory, pred_visibility = model(video.clone(), queries.clone())

            # check for nans in predictions, if there are nans, warning with the model name, as warning not print
            if torch.isnan(pred_trajectory).any():
                warnings.warn(f"NaN values found in predictions of {model_name}. This may cause issues in the verifier model.")
    
            else:
                ensemble_tracks.append(pred_trajectory.to(video.device))
                ensemble_visibility.append(pred_visibility.to(video.device))
        
        ensemble_tracks = torch.stack(ensemble_tracks, dim=-2)
        ensemble_visibility = torch.stack(ensemble_visibility, dim=-1)

        if self.with_verifier:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp), torch.no_grad():
                # Run verifier to get rankings and certainties
                verifier_tracks, verifier_visibility = self.teacher_models_dict["verifier"](
                    video.clone(), 
                    queries.clone(), 
                    ensemble_tracks, 
                    ensemble_visibility
                )

        output = {}
        if not only_verifier:
            output["ensemble"] = (ensemble_tracks, ensemble_visibility)

        if self.with_verifier:
            output["verifier"] = (verifier_tracks, verifier_visibility)

        return output