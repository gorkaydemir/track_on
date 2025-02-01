from typing import Tuple

import torch
from torch import nn

__all__ = ["DinoV2", "Upsample"]

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    pretrained: bool = True,
    **kwargs,
):
    from models.baseline_dino.vit import vision_transformer as vits

    model_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model


def dinov2_vits14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained, **kwargs)


def dinov2_vitb14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-B/14 model pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, **kwargs)

def dinov2_vitl14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained, **kwargs)


def dinov2_vitg14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_giant2", ffn_layer="swiglufused",
                              pretrained=pretrained, **kwargs)


class DinoV2(torch.nn.Module):
    def __init__(self, image_size: Tuple[int, int], model: str = "vits14", pretrained: bool = False,
                 frozen: bool = False, drop_path_rate: float = 0.0, window_block_indexes=(),
                 window_size: int = 0, use_multi_scale_features: bool = False):
        super().__init__()
        self.frozen = frozen
        if model == "vits14":
            self.feat_dim = 384
            self.patch_size = 14
            self.model = dinov2_vits14(pretrained=pretrained, drop_path_rate=drop_path_rate,
                                       window_block_indexes=window_block_indexes,
                                       window_size=window_size)
        elif model == "vitb14":
            self.feat_dim = 768
            self.patch_size = 14
            self.model = dinov2_vitb14(pretrained=pretrained, drop_path_rate=drop_path_rate,
                                       window_block_indexes=window_block_indexes,
                                       window_size=window_size)

        self.model.mask_token = None  # can't use ddp_find_unused_parameters_false otherwise
        # Freeze model
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        # Multi-scale features
        if use_multi_scale_features:
            self.multi_scale_blocks = [5, 8, 11]
            self.scale_adapters = nn.ModuleList()
            self.scale_adapters.append(nn.ConvTranspose2d(self.feat_dim, self.feat_dim,
                                                          kernel_size=2, stride=2)) # upsample by 2
            self.scale_adapters.append(nn.Identity())
            self.scale_adapters.append(nn.MaxPool2d(kernel_size=2, stride=2)) # downsample by 2
        else:
            self.multi_scale_blocks = None

        self.H_out = image_size[0] // self.patch_size
        self.W_out = image_size[1] // self.patch_size

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                x = self.model.forward_features(x, return_blocks=self.multi_scale_blocks)
        else:
            x = self.model.forward_features(x, return_blocks=self.multi_scale_blocks)

        if self.multi_scale_blocks is not None:
            for i in range(len(x)):
                if i == len(x) - 1:
                    x[i] = x[i]["x_norm_patchtokens"]  # (B, Patches, feat_dim)
                else:
                    x[i] = x[i]["x"][:, 1:, :]  # remove CLS token (B, Patches, feat_dim)
                x[i] = x[i].permute(0, 2, 1).contiguous()  # (B, feat_dim, Patches)
                x[i] = x[i].view(-1, self.feat_dim, self.H_out, self.W_out)  # (B, feat_dim, H, W)
                x[i] = self.scale_adapters[i](x[i])
            return x
        else:
            # x keys: ["x_norm_clstoken", "x_norm_patchtokens", "x_prenorm"]
            x = x["x_norm_patchtokens"]  # (B, Patches, feat_dim)
            x = x.permute(0, 2, 1).contiguous()  # (B, feat_dim, Patches)
            x = x.view(-1, self.feat_dim, self.H_out, self.W_out)  # (B, feat_dim, H, W)
            return [x]

    def init_weights(self) -> None:
        pass


class Upsample(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)
