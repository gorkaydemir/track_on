import torch
import torch.nn.functional as F

from ensemble.tapnext.tapnext_torch import TAPNext
from ensemble.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty


class TAPNextPredictor(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()

        self.interp_shape = (256, 256)
        tapnext_model = TAPNext(image_size=self.interp_shape)
        self.model = restore_model_from_jax_checkpoint(tapnext_model, checkpoint_path)
        self.model.eval()

    def forward(self, rgbs, queries=None):
        B, T, C, H, W = rgbs.shape
        rgbs_ = rgbs.reshape(B * T, C, H, W)
        rgbs_ = F.interpolate(rgbs_, tuple(self.interp_shape), mode="bilinear")
        rgbs_ = rgbs_.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
        rgbs_ = rgbs_[0].permute(0, 2, 3, 1)
        rgbs_ = (rgbs_ / 255.0) * 2 - 1
        rgbs_ = rgbs_.unsqueeze(0)

        queries = queries.clone().float()
        B, N, D = queries.shape
        assert D == 3
        assert B == 1
        queries[:, :, 1] *= self.interp_shape[1] / W
        queries[:, :, 2] *= self.interp_shape[0] / H
        queries = torch.stack(
            [queries[..., 0], queries[..., 2], queries[..., 1]], dim=-1
        )

        # check if there are nan values in queries
        if torch.isnan(queries).any():
            raise ValueError("Queries contain NaN values")
        
        pred_tracks, track_logits, visible_logits, tracking_state = self.model(video=rgbs_[:, :1], query_points=queries)
        pred_visible = visible_logits > 0
        pred_tracks, pred_visible = [pred_tracks.cpu()], [pred_visible.cpu()]
        pred_track_logits, pred_visible_logits = [track_logits.cpu()], [
            visible_logits.cpu()
        ]

        for frame in range(1, T):
            (curr_tracks, curr_track_logits, curr_visible_logits, tracking_state) = self.model(video=rgbs_[:, frame : frame + 1], state=tracking_state)
            
            curr_visible = curr_visible_logits > 0
            pred_tracks.append(curr_tracks.cpu())
            pred_visible.append(curr_visible.cpu())
            pred_track_logits.append(curr_track_logits.cpu())
            pred_visible_logits.append(curr_visible_logits.cpu())
            
        tracks = torch.cat(pred_tracks, dim=1).transpose(1, 2)
        pred_visible = torch.cat(pred_visible, dim=1).transpose(1, 2)
        track_logits = torch.cat(pred_track_logits, dim=1).transpose(1, 2)
        visible_logits = torch.cat(pred_visible_logits, dim=1).transpose(1, 2)

        visibility = (pred_visible.squeeze(-1)).permute(0, 2, 1)
        tracks = tracks.permute(0, 2, 1, 3).flip(-1)

        tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
        tracks[:, :, :, 1] *= H / float(self.interp_shape[0])

        tracks = tracks.to(rgbs.device)
        visibility = visibility.to(rgbs.device)

        return tracks, visibility