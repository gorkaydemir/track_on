import torch
import torch.nn.functional as F
from ensemble.locotrack.locotrack_model import LocoTrack


def postprocess_occlusions(occlusions, expected_dist):
    visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
    # return (1 - F.sigmoid(occlusions)) > 0.5
    return visibles


class LocoTrackPredictor(torch.nn.Module):
    def __init__(self, checkpoint_path, model_size='base'):
        super().__init__()
        
        self.interp_shape = (256, 256)
        self.model = LocoTrack(model_size=model_size, num_pips_iter=4)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        self.model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
        self.model.eval()

    def forward(self, rgbs, queries=None):
        B, T, C, H, W = rgbs.shape
        rgbs_ = rgbs.reshape(B * T, C, H, W)
        rgbs_ = F.interpolate(rgbs_, tuple(self.interp_shape), mode="bilinear")
        rgbs_ = rgbs_.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
        rgbs_ = rgbs_[0].permute(0, 2, 3, 1)
        rgbs_ = (rgbs_ / 255.0) * 2 - 1

        if queries is not None:
            queries = queries.clone().float()
            B, N, D = queries.shape
            assert D == 3
            assert B == 1
            queries[:, :, 1] *= self.interp_shape[1] / W
            queries[:, :, 2] *= self.interp_shape[0] / H
            queries = torch.stack(
                [queries[..., 0], queries[..., 2], queries[..., 1]], dim=-1
            )

        outputs = self.model(video=rgbs_[None], query_points=queries, refinement_resolutions=[self.interp_shape])
        tracks, occlusions, expected_dist = (
            outputs["tracks"],
            outputs["occlusion"][0],
            outputs["expected_dist"][0],
        )
        visibility = postprocess_occlusions(occlusions, expected_dist)[None].permute(
            0, 2, 1
        )

        tracks = tracks.permute(0, 2, 1, 3)

        tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
        tracks[:, :, :, 1] *= H / float(self.interp_shape[0])

        return tracks, visibility