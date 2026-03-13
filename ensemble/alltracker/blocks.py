import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from itertools import repeat
import collections
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence
from functools import partial
import einops
import math
from torchvision.ops.misc import Conv2dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

to_2tuple = _ntuple(2)

class InputPadder:
    """ Pads images such that dimensions are divisible by a certain stride """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 64) + 1) * 64 - self.ht) % 64
        pad_wd = (((self.wd // 64) + 1) * 64 - self.wd) % 64
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def bilinear_sampler(
        input, coords,
        align_corners=True,
        padding_mode="border",
        normalize_coords=True):
    # func from mattie (oct9)
    if input.ndim not in [4, 5]:
        raise ValueError("input must be 4D or 5D.")
    
    if input.ndim == 4 and not coords.ndim == 4:
        raise ValueError("input is 4D, but coords is not 4D.")

    if input.ndim == 5 and not coords.ndim == 5:
        raise ValueError("input is 5D, but coords is not 5D.")

    if coords.ndim == 5:
        coords = coords[..., [1, 2, 0]]  # t x y -> x y t to match what grid_sample() expects.

    if normalize_coords:
        if align_corners:
            # Normalize coordinates from [0, W/H - 1] to [-1, 1].
            coords = (
                coords
                * torch.tensor([2 / max(size - 1, 1) for size in reversed(input.shape[2:])], device=coords.device)
                - 1
            )
        else:
            # Normalize coordinates from [0, W/H] to [-1, 1].
            coords = coords * torch.tensor([2 / size for size in reversed(input.shape[2:])], device=coords.device) - 1
            
    return F.grid_sample(input, coords, align_corners=align_corners, padding_mode=padding_mode)


class CorrBlock:
    def __init__(self, fmap1, fmap2, corr_levels, corr_radius):
        self.num_levels = corr_levels
        self.radius = corr_radius
        self.corr_pyramid = []
        # all pairs correlation
        for i in range(self.num_levels):
            corr = CorrBlock.corr(fmap1, fmap2, 1)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch*h1*w1, dim, h2, w2)
            fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode='area')
            # print('corr', corr.shape)
            self.corr_pyramid.append(corr)

    def __call__(self, coords, dilation=None):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        if dilation is None:
            dilation = torch.ones(batch, 1, h1, w1, device=coords.device)

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            device = coords.device
            dx = torch.linspace(-r, r, 2*r+1, device=device)
            dy = torch.linspace(-r, r, 2*r+1, device=device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            delta_lvl = delta_lvl * dilation.view(batch * h1 * w1, 1, 1, 1)
            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous().float()  
        return out

    @staticmethod
    def corr(fmap1, fmap2, num_head):
        batch, dim, h1, w1 = fmap1.shape
        h2, w2 = fmap2.shape[2:]
        fmap1 = fmap1.view(batch, num_head, dim // num_head, h1*w1)
        fmap2 = fmap2.view(batch, num_head, dim // num_head, h2*w2) 
        corr = fmap1.transpose(2, 3) @ fmap2
        corr = corr.reshape(batch, num_head, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)
        return corr  / torch.sqrt(torch.tensor(dim).float())
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
class CNBlock1d(nn.Module):
    def __init__(
        self,
        dim,
        output_dim,
        layer_scale: float = 1e-6,
        stochastic_depth_prob: float = 0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dense=True,
        use_attn=True,
        use_mixer=False,
        use_conv=False,
        use_convb=False,
        use_layer_scale=True,
    ) -> None:
        super().__init__()
        self.dense = dense
        self.use_attn = use_attn
        self.use_mixer = use_mixer
        self.use_conv = use_conv
        self.use_layer_scale = use_layer_scale

        if use_attn:
            assert not use_mixer
            assert not use_conv
            assert not use_convb
        
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if use_attn:
            num_heads = 8
            self.block = AttnBlock(
                hidden_size=dim,
                num_heads=num_heads,
                mlp_ratio=4,
                attn_class=Attention,
            )
        elif use_mixer:
            self.block = MLPMixerBlock(
                S=16,
                dim=dim,
                depth=1,
                expansion_factor=2,
            )
        elif use_conv:
            self.block = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True, padding_mode='zeros'),
                Permute([0, 2, 1]),
                norm_layer(dim),
                nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
                nn.GELU(),
                nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
                Permute([0, 2, 1]),
            )
        elif use_convb:
            self.block = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, bias=True, padding_mode='zeros'),
                Permute([0, 2, 1]),
                norm_layer(dim),
                nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
                nn.GELU(),
                nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
                Permute([0, 2, 1]),
            )
        else:
            assert(False) # choose attn, mixer, or conv please

        if self.use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim, 1) * layer_scale)
        else:
            self.layer_scale = 1.0
            
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        if output_dim != dim:
            self.final = nn.Conv1d(dim, output_dim, kernel_size=1, padding=0)
        else:
            self.final = nn.Identity()

    def forward(self, input, S=None):
        if self.dense:
            assert S is not None
            BS,C,H,W = input.shape
            B = BS//S

            input = einops.rearrange(input, '(b s) c h w -> (b h w) c s', b=B, s=S, c=C, h=H, w=W)

            if self.use_mixer or self.use_attn:
                # mixer/transformer blocks want B,S,C
                result = self.layer_scale * self.block(input.permute(0,2,1)).permute(0,2,1)
            else:
                result = self.layer_scale * self.block(input)
            result = self.stochastic_depth(result)
            result += input
            result = self.final(result)

            result = einops.rearrange(result, '(b h w) c s -> (b s) c h w', b=B, s=S, c=C, h=H, w=W)
        else:
            B,S,C = input.shape

            if S<7:
                return input

            input = einops.rearrange(input, 'b s c -> b c s', b=B, s=S, c=C)

            result = self.layer_scale * self.block(input)
            result = self.stochastic_depth(result)
            result += input

            result = self.final(result)

            result = einops.rearrange(result, 'b c s -> b s c', b=B, s=S, c=C)
        
        return result
    
class CNBlock2d(nn.Module):
    def __init__(
        self,
        dim,
        output_dim,
        layer_scale: float = 1e-6,
        stochastic_depth_prob: float = 0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_layer_scale=True,
    ) -> None:
        super().__init__()
        self.use_layer_scale = use_layer_scale
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True, padding_mode='zeros'),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        if self.use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        else:
            self.layer_scale = 1.0
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        
        if output_dim != dim:
            self.final = nn.Conv2d(dim, output_dim, kernel_size=1, padding=0)
        else:
            self.final = nn.Identity()

    def forward(self, input, S=None):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        result = self.final(result)
        return result

class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
        downsample: bool,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.downsample = downsample

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ", downsample={downsample}"
        s += ")"
        return s.format(**self.__dict__)
    
class ConvNeXt(nn.Module):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],
            stochastic_depth_prob: float = 0.0,
            layer_scale: float = 1e-6,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            init_weights=True):
        super().__init__()

        self.init_weights = init_weights
        
        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock2d

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                if cnf.downsample:
                    layers.append(
                        nn.Sequential(
                            norm_layer(cnf.input_channels),
                            nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                        )
                    )
                else:
                    # we convert the 2x2 downsampling layer into a 3x3 with dilation2 and replicate padding.
                    # replicate padding compensates for the fact that this kernel never saw zero-padding.
                    layers.append(
                        nn.Sequential(
                            norm_layer(cnf.input_channels),
                            nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='zeros'),
                        )
                    )

        self.features = nn.Sequential(*layers)
        
        # self.final_conv = conv1x1(block_setting[-1].input_channels, output_dim)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.init_weights:
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            pretrained_dict = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).state_dict()
            # from torchvision.models import convnext_base, ConvNeXt_Base_Weights
            # pretrained_dict = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            for k, v in pretrained_dict.items():
                if k == 'features.4.1.weight': # this is the layer normally in charge of 2x2 downsampling
                    # convert to 3x3 filter
                    pretrained_dict[k] = F.interpolate(v, (3, 3), mode='bicubic', align_corners=True) * (4/9.0)
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
        

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # x = self.final_conv(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = num_heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_bias=None):
        B, N1, C = x.shape
        H = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        x = F.scaled_dot_product_attention(q, k, v) # scale default is already dim^-0.5
        x = einops.rearrange(x, 'b h n d -> b n (h d)')
        return self.to_out(x)
    
class CrossAttnBlock(nn.Module):
    def __init__(
        self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context, mask=None):
        attn_bias = None
        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(
                    -1, self.cross_attn.heads, x.shape[1], -1
                )

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.cross_attn(
            self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias
        )
        x = x + self.mlp(self.norm2(x))
        return x
    
class AttnBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_class: Callable[..., nn.Module] = Attention,
        mlp_ratio=4.0,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_class(hidden_size, num_heads=num_heads, qkv_bias=True, dim_head=hidden_size//num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, mask=None):
        attn_bias = mask
        if mask is not None:
            mask = (
                (mask[:, None] * mask[:, :, None])
                .unsqueeze(1)
                .expand(-1, self.attn.num_heads, -1, -1)
            )
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value

        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4,
            output_dim * 2,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        def _bilinear_intepolate(x):
            return F.interpolate(
                x,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )

        a = _bilinear_intepolate(a)
        b = _bilinear_intepolate(b)
        c = _bilinear_intepolate(c)
        d = _bilinear_intepolate(d)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x

class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        num_virtual_tracks=64,
        add_space_attn=True,
        linear_layer_for_vis_conf=False,
        use_time_conv=False,
        use_time_mixer=False,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        if linear_layer_for_vis_conf:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim - 2, bias=True)
            self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(
            torch.randn(1, num_virtual_tracks, 1, hidden_size)
        )
        self.add_space_attn = add_space_attn
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf

        if use_time_conv:
            self.time_blocks = nn.ModuleList(
                [
                    CNBlock1d(hidden_size, hidden_size, dense=False)
                    for _ in range(time_depth)
                ]
            )
        elif use_time_mixer:
            self.time_blocks = nn.ModuleList(
                [
                    MLPMixerBlock(
                        S=16,
                        dim=hidden_size,
                        depth=1,
                    )
                    for _ in range(time_depth)
                ]
            )
        else:
            self.time_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=Attention,
                    )
                    for _ in range(time_depth)
                ]
            )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=Attention,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            if self.linear_layer_for_vis_conf:
                torch.nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        def _trunc_init(module):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None, add_space_attn=True):
        tokens = self.input_transform(input_tensor)

        B, _, T, _ = tokens.shape
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape
        j = 0
        layers = []
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            if (
                add_space_attn
                and hasattr(self, "space_virtual_blocks")
                and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0)
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C

                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = self.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, mask=mask
                )

                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens, mask=mask
                )

                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(
                    0, 2, 1, 3
                )  # (B T) N C -> B N T C
                j += 1
        tokens = tokens[:, : N - self.num_virtual_tracks]

        flow = self.flow_head(tokens)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        return flow


class MMPreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x
    
def MMFeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )
    
def MLPMixer(S, input_dim, dim, output_dim, depth=6, expansion_factor=4, dropout=0., do_reduce=False):
    # input is coming in as B,S,C, as standard for mlp and transformer
    # chan_first treats S as the channel dim, and transforms it to a new S
    # chan_last treats C as the channel dim, and transforms it to a new C 
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
    if do_reduce:
        return nn.Sequential(
            nn.Linear(input_dim, dim),
            *[nn.Sequential(
                MMPreNormResidual(dim, MMFeedForward(S, expansion_factor, dropout, chan_first)),
                MMPreNormResidual(dim, MMFeedForward(dim, expansion_factor, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(dim, output_dim)
        )
    else:
        return nn.Sequential(
            nn.Linear(input_dim, dim),
            *[nn.Sequential(
                MMPreNormResidual(dim, MMFeedForward(S, expansion_factor, dropout, chan_first)),
                MMPreNormResidual(dim, MMFeedForward(dim, expansion_factor, dropout, chan_last))
            ) for _ in range(depth)],
        )

def MLPMixerBlock(S, dim, depth=1, expansion_factor=4, dropout=0., do_reduce=False):
    # input is coming in as B,S,C, as standard for mlp and transformer
    # chan_first treats S as the channel dim, and transforms it to a new S
    # chan_last treats C as the channel dim, and transforms it to a new C 
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
    return nn.Sequential(
        *[nn.Sequential(
            MMPreNormResidual(dim, MMFeedForward(S, expansion_factor, dropout, chan_first)),
            MMPreNormResidual(dim, MMFeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
    )
    
    
class MlpUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        num_virtual_tracks=64,
        add_space_attn=True,
        linear_layer_for_vis_conf=False,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        if linear_layer_for_vis_conf:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim - 2, bias=True)
            self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(
            torch.randn(1, num_virtual_tracks, 1, hidden_size)
        )
        self.add_space_attn = add_space_attn
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.time_blocks = nn.ModuleList(
            [
                MLPMixer(
                    S=16,
                    input_dim=hidden_size,
                    dim=hidden_size,
                    output_dim=hidden_size,
                    depth=1,
                )
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=Attention,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            if self.linear_layer_for_vis_conf:
                torch.nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        def _trunc_init(module):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None, add_space_attn=True):
        tokens = self.input_transform(input_tensor)

        B, _, T, _ = tokens.shape
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape
        j = 0
        layers = []
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            if (
                add_space_attn
                and hasattr(self, "space_virtual_blocks")
                and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0)
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C

                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = self.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, mask=mask
                )

                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens, mask=mask
                )

                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(
                    0, 2, 1, 3
                )  # (B T) N C -> B N T C
                j += 1
        tokens = tokens[:, : N - self.num_virtual_tracks]

        flow = self.flow_head(tokens)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        return flow
    
class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_channel, dim=128, pdim=2):
        super(BasicMotionEncoder, self).__init__()
        self.pdim = pdim
        self.convc1 = nn.Conv2d(corr_channel, dim*4, 1, padding=0)
        self.convc2 = nn.Conv2d(dim*4, dim+dim//2, 3, padding=1)
        if pdim==2 or pdim==4:
            self.convf1 = nn.Conv2d(pdim, dim*2, 5, padding=2)
            self.convf2 = nn.Conv2d(dim*2, dim//2, 3, padding=1)
            self.conv = nn.Conv2d(dim*2, dim-pdim, 3, padding=1)
        else:
            self.conv = nn.Conv2d(dim+dim//2+pdim, dim, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        if self.pdim==2 or self.pdim==4:
            flo = F.relu(self.convf1(flow))
            flo = F.relu(self.convf2(flo))
            cor_flo = torch.cat([cor, flo], dim=1)
            out = F.relu(self.conv(cor_flo))
            return torch.cat([out, flow], dim=1)
        else:
            # the flow is already encoded to something nice
            cor_flo = torch.cat([cor, flow], dim=1)
            return F.relu(self.conv(cor_flo))
            # return torch.cat([out, flow], dim=1)
    
def conv133_encoder(input_dim, dim, expansion_factor=4):
    return nn.Sequential(
        nn.Conv2d(input_dim, dim*expansion_factor, kernel_size=1),
        nn.GELU(),
        nn.Conv2d(dim*expansion_factor, dim*expansion_factor, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(dim*expansion_factor, dim, kernel_size=3, padding=1),
    )        
    
class BasicUpdateBlock(nn.Module):
    def __init__(self, corr_channel, num_blocks, hdim=128, cdim=128):
        # flowfeat is hdim; ctxfeat is dim. typically hdim==cdim.
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_channel, dim=cdim)
        self.compressor = conv1x1(2*cdim+hdim, hdim)
        
        self.refine = []
        for i in range(num_blocks):
            self.refine.append(CNBlock1d(hdim, hdim))
            self.refine.append(CNBlock2d(hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, flowfeat, ctxfeat, corr, flow, S, upsample=True):
        BS,C,H,W = flowfeat.shape
        B = BS//S

        # with torch.no_grad():
        motion_features = self.encoder(flow, corr)
        flowfeat = self.compressor(torch.cat([flowfeat, ctxfeat, motion_features], dim=1))
            
        for blk in self.refine:
            flowfeat = blk(flowfeat, S)
        return flowfeat
    
class FullUpdateBlock(nn.Module):
    def __init__(self, corr_channel, num_blocks, hdim=128, cdim=128, pdim=2, use_attn=False):
        # flowfeat is hdim; ctxfeat is dim. typically hdim==cdim.
        super(FullUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_channel, dim=cdim, pdim=pdim)

        # note we have hdim==cdim
        # compressor chans:
        #   dim for flowfeat
        #   dim for ctxfeat
        #   dim for motion_features
        #   pdim for flow (if p 2, like if we give sincos(relflow))
        #   2 for visconf
        
        if pdim==2:
            # hdim==cdim
            # dim for flowfeat
            # dim for ctxfeat
            # dim for motion_features
            # 2 for visconf
            self.compressor = conv1x1(2*cdim+hdim+2, hdim) 
        else:
            # we concatenate the flow info again, to not lose it (e.g., from the relu)
            self.compressor = conv1x1(2*cdim+hdim+2+pdim, hdim)
        
        self.refine = []
        for i in range(num_blocks):
            self.refine.append(CNBlock1d(hdim, hdim, use_attn=use_attn))
            self.refine.append(CNBlock2d(hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, flowfeat, ctxfeat, visconf, corr, flow, S, upsample=True):
        BS,C,H,W = flowfeat.shape
        B = BS//S
        motion_features = self.encoder(flow, corr)
        flowfeat = self.compressor(torch.cat([flowfeat, ctxfeat, motion_features, visconf], dim=1))
        for blk in self.refine:
            flowfeat = blk(flowfeat, S)
        return flowfeat

class MixerUpdateBlock(nn.Module):
    def __init__(self, corr_channel, num_blocks, hdim=128, cdim=128):
        # flowfeat is hdim; ctxfeat is dim. typically hdim==cdim.
        super(MixerUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_channel, dim=cdim)
        self.compressor = conv1x1(2*cdim+hdim, hdim)
        
        self.refine = []
        for i in range(num_blocks):
            self.refine.append(CNBlock1d(hdim, hdim, use_mixer=True))
            self.refine.append(CNBlock2d(hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, flowfeat, ctxfeat, corr, flow, S, upsample=True):
        BS,C,H,W = flowfeat.shape
        B = BS//S

        # with torch.no_grad():
        motion_features = self.encoder(flow, corr)
        flowfeat = self.compressor(torch.cat([flowfeat, ctxfeat, motion_features], dim=1))
            
        for ii, blk in enumerate(self.refine):
            flowfeat = blk(flowfeat, S)
        return flowfeat
    
class FacUpdateBlock(nn.Module):
    def __init__(self, corr_channel, num_blocks, hdim=128, cdim=128, pdim=84, use_attn=False):
        super(FacUpdateBlock, self).__init__()
        self.corr_encoder = conv133_encoder(corr_channel, cdim)
        # note we have hdim==cdim
        # compressor chans:
        #   dim for flowfeat
        #   dim for ctxfeat
        #   dim for corr
        #   pdim for flow
        #   2 for visconf
        self.compressor = conv1x1(2*cdim+hdim+2+pdim, hdim)
        self.refine = []
        for i in range(num_blocks):
            self.refine.append(CNBlock1d(hdim, hdim, use_attn=use_attn))
            self.refine.append(CNBlock2d(hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, flowfeat, ctxfeat, visconf, corr, flow, S, upsample=True):
        BS,C,H,W = flowfeat.shape
        B = BS//S
        corr = self.corr_encoder(corr)
        flowfeat = self.compressor(torch.cat([flowfeat, ctxfeat, corr, visconf, flow], dim=1))
        for blk in self.refine:
            flowfeat = blk(flowfeat, S)
        return flowfeat
    
class CleanUpdateBlock(nn.Module):
    def __init__(self, corr_channel, num_blocks, cdim=128, hdim=256, pdim=84, use_attn=False, use_layer_scale=True):
        super(CleanUpdateBlock, self).__init__()
        self.corr_encoder = conv133_encoder(corr_channel, cdim)
        # compressor chans:
        #   cdim for flowfeat
        #   cdim for ctxfeat
        #   cdim for corrfeat
        #   pdim for flow
        #   2 for visconf
        self.compressor = conv1x1(3*cdim+pdim+2, hdim)
        self.refine = []
        for i in range(num_blocks):
            self.refine.append(CNBlock1d(hdim, hdim, use_attn=use_attn, use_layer_scale=use_layer_scale))
            self.refine.append(CNBlock2d(hdim, hdim, use_layer_scale=use_layer_scale))
        self.refine = nn.ModuleList(self.refine)
        self.final_conv = conv1x1(hdim, cdim)

    def forward(self, flowfeat, ctxfeat, visconf, corr, flow, S, upsample=True):
        BS,C,H,W = flowfeat.shape
        B = BS//S
        corrfeat = self.corr_encoder(corr)
        flowfeat = self.compressor(torch.cat([flowfeat, ctxfeat, corrfeat, flow, visconf], dim=1))
        for blk in self.refine:
            flowfeat = blk(flowfeat, S)
        flowfeat = self.final_conv(flowfeat)
        return flowfeat

class RelUpdateBlock(nn.Module):
    def __init__(self, corr_channel, num_blocks, cdim=128, hdim=128, pdim=4, use_attn=True, use_mixer=False, use_conv=False, use_convb=False, use_layer_scale=True, no_time=False, no_space=False, no_ctx=False):
        super(RelUpdateBlock, self).__init__()
        self.motion_encoder = BasicMotionEncoder(corr_channel, dim=hdim, pdim=pdim) # B,hdim,H,W
        self.no_ctx = no_ctx
        if no_ctx:
            self.compressor = conv1x1(cdim+hdim+2, hdim)
        else:
            self.compressor = conv1x1(2*cdim+hdim+2, hdim)
        self.refine = []
        for i in range(num_blocks):
            if not no_time:
                self.refine.append(CNBlock1d(hdim, hdim, use_attn=use_attn, use_mixer=use_mixer, use_conv=use_conv, use_convb=use_convb, use_layer_scale=use_layer_scale))
            if not no_space:
                self.refine.append(CNBlock2d(hdim, hdim, use_layer_scale=use_layer_scale))
        self.refine = nn.ModuleList(self.refine)
        self.final_conv = conv1x1(hdim, cdim)

    def forward(self, flowfeat, ctxfeat, visconf, corr, flow, S, upsample=True):
        BS,C,H,W = flowfeat.shape
        B = BS//S
        motion_features = self.motion_encoder(flow, corr)
        if self.no_ctx:
            flowfeat = self.compressor(torch.cat([flowfeat, motion_features, visconf], dim=1))
        else:
            flowfeat = self.compressor(torch.cat([flowfeat, ctxfeat, motion_features, visconf], dim=1))
        for blk in self.refine:
            flowfeat = blk(flowfeat, S)
        flowfeat = self.final_conv(flowfeat)
        return flowfeat