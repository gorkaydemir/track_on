import torch
from model.trackon import Track_On2
from evaluation.evaluator import get_points_on_a_grid  # adjust import if needed
from collections import OrderedDict

ALLOWED_MISSING_PREFIXES = (
    "backbone.vit_encoder.dinov2",
    "backbone.vit_encoder.dinov3",
)

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

class Predictor(torch.nn.Module):
    def __init__(self, model_args, checkpoint_path=None, support_grid_size=20):
        super().__init__()
        self.model = Track_On2(model_args)

        if checkpoint_path is not None:
            self._load_model_and_check(checkpoint_path)

        # Optional inference-time memory extension
        ime_size = getattr(model_args, "M_i", 72)
        if ime_size != model_args.M:
            self.model.memory_extension(ime_size)

        self.model.eval()
        self.model = torch.compile(self.model, mode="reduce-overhead")

        self.delta_v = model_args.delta_v
        self.support_grid_size = support_grid_size
        self.uniform_tracking_grid_size = 32  # fixed size for uniform grid queries

    def _load_model_and_check(self, checkpoint_path):
        raw = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _strip_module_prefix(_extract_state_dict(raw))

        # Try load with strict=False, then validate missing keys
        load_result = self.model.load_state_dict(state_dict, strict=False)
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)

        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint (not present in model): {unexpected}")

        disallowed_missing = [
            k for k in missing
            if not any(k.startswith(pfx) for pfx in ALLOWED_MISSING_PREFIXES)
        ]

        if disallowed_missing:
            raise RuntimeError(
                "Checkpoint is missing parameters outside the allowed encoders.\n"
                f"Disallowed missing keys:\n  {disallowed_missing}\n"
                f"(Allowed missing prefixes: {ALLOWED_MISSING_PREFIXES})")

        print(f"Loaded model weights from {checkpoint_path}")
        if missing:
            print(f"Info: missing (allowed) weights: {len(missing)} keys under {set(p.split('.')[0] + '.' + p.split('.')[1] + '.' + p.split('.')[2] for p in missing)}")

        
    @torch.no_grad()
    def forward(self, video, queries=None):
        """
        video: Tensor of shape (1, 3, T, H, W)
        queries: Tensor of shape (1, N, 3), where each query is (t, x, y), in pixel coordinates
        """
        B, C, T, H, W = video.shape
        device = video.device

        # Add extra queries from uniform grid if none provided
        if queries is None:
            base = get_points_on_a_grid(self.uniform_tracking_grid_size, (H, W), device)  # (1, K^2, 2)
            base = torch.cat([torch.zeros(1, self.uniform_tracking_grid_size ** 2, 1, device=device), base], dim=-1)
            queries = base.repeat(B, 1, 1)  # (B, K^2, 3)
            N = queries.shape[1]
            
            
        elif self.support_grid_size > 0:
            N = queries.shape[1]
            extra = get_points_on_a_grid(self.support_grid_size, (H, W), device)  # (1, S^2, 2)
            extra = torch.cat([torch.zeros(1, self.support_grid_size ** 2, 1, device=device), extra], dim=-1)
            queries = torch.cat([queries, extra.repeat(B, 1, 1)], dim=1)  # (B, N + S^2, 3)


        # Forward through model
        out = self.model.forward_online(video, queries)

        pred_trajectory = out["P"][:, :, :N]  # (B, T, N, 2)
        pred_visibility = (out["V_logit"][:, :, :N].sigmoid() >= self.delta_v)  # (B, T, N)

        return pred_trajectory, pred_visibility