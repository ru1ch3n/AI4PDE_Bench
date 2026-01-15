from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        # Complex weights: (in, out, modes)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, X)
        b, c, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, X//2+1)
        out_ft = torch.zeros(b, self.out_channels, x_ft.size(-1), device=x.device, dtype=torch.cfloat)
        m = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :m] = torch.einsum("bix,iox->box", x_ft[:, :, :m], self.weight[:, :, :m])
        x_out = torch.fft.irfft(out_ft, n=n, dim=-1)
        return x_out


class FNO1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int = 64, modes: int = 16, depth: int = 4):
        super().__init__()
        self.proj_in = nn.Conv1d(in_channels, width, kernel_size=1)
        self.spec_convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)])
        self.proj_out = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, X)
        x = self.proj_in(x)
        for spec, w in zip(self.spec_convs, self.ws):
            x = F.gelu(spec(x) + w(x))
        return self.proj_out(x)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # (B, C, H, W//2+1)
        out_ft = torch.zeros(b, self.out_channels, h, x_ft.size(-1), device=x.device, dtype=torch.cfloat)

        m1 = min(self.modes1, h)
        m2 = min(self.modes2, x_ft.size(-1))

        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weight[:, :, :m1, :m2]
        )
        x_out = torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))
        return x_out


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes1: int = 16,
        modes2: int = 16,
        depth: int = 4,
    ):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spec_convs = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.proj_out = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj_in(x)
        for spec, w in zip(self.spec_convs, self.ws):
            x = F.gelu(spec(x) + w(x))
        return self.proj_out(x)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, X, Y, Z)
        b, c, x1, x2, x3 = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))  # (B,C,X,Y,Z//2+1)
        out_ft = torch.zeros(b, self.out_channels, x1, x2, x_ft.size(-1), device=x.device, dtype=torch.cfloat)

        m1 = min(self.modes1, x1)
        m2 = min(self.modes2, x2)
        m3 = min(self.modes3, x_ft.size(-1))

        out_ft[:, :, :m1, :m2, :m3] = torch.einsum(
            "bixyz,ioxyz->boxyz", x_ft[:, :, :m1, :m2, :m3], self.weight[:, :, :m1, :m2, :m3]
        )
        x_out = torch.fft.irfftn(out_ft, s=(x1, x2, x3), dim=(-3, -2, -1))
        return x_out


class FNO3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        depth: int = 4,
    ):
        super().__init__()
        self.proj_in = nn.Conv3d(in_channels, width, kernel_size=1)
        self.spec_convs = nn.ModuleList(
            [SpectralConv3d(width, width, modes1, modes2, modes3) for _ in range(depth)]
        )
        self.ws = nn.ModuleList([nn.Conv3d(width, width, kernel_size=1) for _ in range(depth)])
        self.proj_out = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, X, Y, Z)
        x = self.proj_in(x)
        for spec, w in zip(self.spec_convs, self.ws):
            x = F.gelu(spec(x) + w(x))
        return self.proj_out(x)
