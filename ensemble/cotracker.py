import torch
import torch.nn as nn

class CoTracker_Predictor(nn.Module):
    def __init__(self, windowed=False):
        super().__init__()
        version = "cotracker3_online" if windowed else "cotracker3_offline"
        self.windowed = windowed
        self.model = torch.hub.load("facebookresearch/co-tracker", version).eval()
        self.grid_size = 10

    def forward(self, video, queries):
        # :args video: (B, T, 3, H, W), in [0, 255] range
        # :args queries: (B, N, 3)
        #
        # :returns pred_tracks: (B, T, N, 2)
        # :returns pred_visibility: (B, T, N)
        
        if self.windowed:
            self.model(video_chunk=video, is_first_step=True, queries=queries, grid_size=self.grid_size)  
            for ind in range(0, video.shape[1] - self.model.step, self.model.step):
                pred_tracks, pred_visibility = self.model(video[:, ind : ind + self.model.step * 2])
                
        else:
            pred_tracks, pred_visibility = self.model(video, queries, grid_size=self.grid_size)

        return pred_tracks, pred_visibility


