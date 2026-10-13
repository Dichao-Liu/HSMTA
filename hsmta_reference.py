import torch
from torch import nn
from torch.nn import functional as F


def _patches(x, kernel, stride):
    batch, channels, height, width = x.shape
    patches = F.unfold(x, kernel_size=kernel, stride=stride)
    return patches.transpose(1, 2).reshape(batch, -1, channels, kernel, kernel)


def _bn_patches(x, bn):
    batch, count, channels, height, width = x.shape
    return bn(x.reshape(batch * count, channels, height, width)).reshape(batch, count, channels, height, width)


class ReferenceLinear(nn.Linear):
    def __init__(self, dim_in, dim_out, kernel_size=1, stride=1, bias=False):
        self.kernel_size = kernel_size
        self.stride = stride
        super().__init__(dim_in * kernel_size * kernel_size, dim_out, bias=bias)

    def forward(self, x):
        batch, count, channels, height, width = x.shape
        kernel = self.kernel_size
        patches = F.unfold(x.reshape(batch * count, channels, height, width), kernel_size=kernel, stride=self.stride)
        projected = F.linear(patches.transpose(1, 2), self.weight, self.bias)
        out_height = (height - kernel) // self.stride + 1
        out_width = (width - kernel) // self.stride + 1
        return projected.transpose(1, 2).reshape(batch, count, self.out_features, out_height, out_width)


class ReferenceBMMBlock(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        nn.Linear(dim, dim)
        self.net_layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch, count, locations, channels, slots = x.shape
        x = self.net_layers[0](x)
        x = x.reshape(batch * count * locations, channels, slots).transpose(1, 2).unsqueeze(-1)
        x = self.net_layers[1](x)
        x = self.net_layers[2](x)
        return x.squeeze(-1).transpose(1, 2).reshape(batch, count, locations, channels, slots)


class ReferenceBMMSeq(nn.Sequential):
    def __init__(self, dim=0, depth=2):
        super().__init__(*[ReferenceBMMBlock(dim=dim) for _ in range(depth)])


class ReferenceAgn(nn.Module):
    def __init__(self, window=2, stride=2, depth=2, dim_in=64, dim_middle=256, dim_out=256):
        super().__init__()
        self.window = window
        self.stride = stride
        self.BMM_Seq = ReferenceBMMSeq(dim=window * window, depth=depth)
        self.spatial_merge_and_refine = nn.Sequential(
            ReferenceLinear(dim_in, dim_middle, kernel_size=window, stride=stride),
            nn.BatchNorm2d(dim_middle, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReferenceLinear(dim_middle, dim_middle),
            nn.BatchNorm2d(dim_middle, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReferenceLinear(dim_middle, dim_out),
            nn.BatchNorm2d(dim_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch, count, channels, height, width = x.shape
        kernel = self.window
        local = F.unfold(x.reshape(batch * count, channels, height, width), kernel_size=kernel, stride=kernel)
        locations = local.shape[-1]
        local = local.transpose(1, 2).reshape(batch, count, locations, channels, kernel * kernel)
        local = self.BMM_Seq(local)
        local = local.permute(0, 1, 3, 4, 2).reshape(batch * count, channels * kernel * kernel, locations)
        x = F.fold(local, output_size=(height, width), kernel_size=kernel, stride=kernel)
        x = x.reshape(batch, count, channels, height, width)
        for i in (0, 2, 4):
            x = self.spatial_merge_and_refine[i](x)
            x = _bn_patches(x, self.spatial_merge_and_refine[i + 1])
        return self.spatial_merge_and_refine[6](x)


class ReferenceAggregationBlock(nn.Module):
    def __init__(self, dim_in=256, dim_middle=128, dim_out=512, stride=1, downsample=None):
        super().__init__()
        self.net_layers = nn.Sequential(
            ReferenceLinear(dim_in, dim_middle),
            nn.BatchNorm2d(dim_middle, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReferenceLinear(dim_middle, dim_middle, kernel_size=3, stride=stride),
            nn.BatchNorm2d(dim_middle, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReferenceLinear(dim_middle, dim_out),
            nn.BatchNorm2d(dim_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.downsample = downsample

    def forward(self, x):
        y = x
        for i in (0, 2, 4):
            y = self.net_layers[i](y)
            y = _bn_patches(y, self.net_layers[i + 1])
        y = self.net_layers[6](y)
        batch, count, channels, height, width = x.shape
        res = self.downsample[0][0](x.reshape(batch * count, channels, height, width))
        res = res.reshape(batch, count, channels, res.shape[-2], res.shape[-1])
        res = self.downsample[0][1](res)
        res = _bn_patches(res, self.downsample[0][2])
        return y + res


class ReferenceRefineBlock(nn.Module):
    def __init__(self, dim_1=512, dim_2=256):
        super().__init__()
        self.net_layers = nn.Sequential(
            ReferenceLinear(dim_1, dim_2),
            nn.BatchNorm2d(dim_2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReferenceLinear(dim_2, dim_2),
            nn.BatchNorm2d(dim_2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReferenceLinear(dim_2, dim_1),
            nn.BatchNorm2d(dim_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in (0, 2, 4):
            x = self.net_layers[i](x)
            x = _bn_patches(x, self.net_layers[i + 1])
        return self.net_layers[6](x)


class ReferenceDownsamplingBlock(nn.Sequential):
    def __init__(self, in_dim=256, out_dim=512, kernel_size=3, stride=1):
        super().__init__(nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride),
            ReferenceLinear(in_dim, out_dim),
            nn.BatchNorm2d(out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))


class ReferenceModel(nn.Module):
    def __init__(self, num_class=1000, input_dim=3, patch_size=36, patch_stride=12):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.stretch = nn.Sequential(
            ReferenceLinear(input_dim, 32),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.Agn1 = ReferenceAgn(window=3, stride=3, depth=2, dim_in=32, dim_middle=16, dim_out=64)
        self.Agn2 = ReferenceAgn(window=2, stride=2, depth=2, dim_in=64, dim_middle=32, dim_out=128)
        self.Agn3 = ReferenceAgn(window=2, stride=2, depth=2, dim_in=128, dim_middle=64, dim_out=256)
        self.Agn4 = nn.Sequential(
            ReferenceAggregationBlock(dim_in=256, dim_middle=128, dim_out=512, stride=1,
                                      downsample=ReferenceDownsamplingBlock(in_dim=256, out_dim=512, kernel_size=3, stride=1)),
            ReferenceRefineBlock(dim_1=512, dim_2=256),
            ReferenceRefineBlock(dim_1=512, dim_2=256)
        )
        self.GlobalPooling = nn.AdaptiveAvgPool2d((1, 1))
        self.Classifier = nn.Sequential(nn.Linear(512, num_class))

    def load_state_dict(self, state_dict, strict=True):
        converted = {}
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.ndim == 4 and value.shape[-2:] in ((1, 1), (2, 2), (3, 3)):
                converted[name] = value.reshape(value.shape[0], -1)
            else:
                converted[name] = value
        return super().load_state_dict(converted, strict=strict)

    def forward(self, x):
        batch, channels, height, width = x.shape
        if self.patch_size != 36 or self.patch_stride != 12:
            raise ValueError("ReferenceModel requires patch_size=36 and patch_stride=12")
        if height < 36 or width < 36 or height % 12 or width % 12:
            raise ValueError("Input height and width must each be at least 36 and divisible by 12")
        x = _patches(x, self.patch_size, self.patch_stride)
        x = self.stretch[0](x)
        x = _bn_patches(x, self.stretch[1])
        x = self.stretch[2](x)
        x = self.Agn1(x)
        x = self.Agn2(x)
        x = self.Agn3(x)
        x = self.Agn4[0](x)
        x = self.Agn4[1](x)
        x = self.Agn4[2](x)
        x = x.mean(dim=(1, 3, 4))
        return self.Classifier(x)
