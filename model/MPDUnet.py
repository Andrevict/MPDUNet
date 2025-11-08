import torch
import torch.nn as nn
from mamba_ssm import Mamba


class Head(nn.Module):
    def __init__(self, in_channel):
        super(Head, self).__init__()
        self.HeadLayer = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU()
        )

    def forward(self, x):
        return self.HeadLayer(x)  # -> (*, *, *, 32)


class Tail(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tail, self).__init__()
        self.TailLayer = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(16, out_channel, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.TailLayer(x)


class OPD(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OPD, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel // 2, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channel // 2)

        self.gelu = nn.GELU()

        self.conv2 = nn.Conv2d(out_channel // 2, out_channel // 2, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_channel // 2)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        tmp = self.gelu(x)
        x = self.conv2(tmp)
        x = self.norm2(x)
        x = self.gelu(x)
        x = torch.cat([x, tmp], dim=1)
        x = self.maxpool(x)
        return x


class OPU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OPU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(4, out_channel)
        self.gelu = nn.GELU()
        self.upsample = nn.ConvTranspose2d(in_channel + out_channel, out_channel, 2, 2, 0)

    def forward(self, x):
        tmp = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        x = torch.cat([x, tmp], dim=1)
        x = self.upsample(x)
        return x


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class PMDUNet(nn.Module):
    def __init__(self, in_channel):
        super(PMDUNet, self).__init__()
        self.Head = Head(in_channel)

        self.OPD1 = OPD(32, 64)
        self.OPD2 = OPD(64, 128)
        self.OPD3 = OPD(128, 256)

        self.PVM1 = PVMLayer(32, 16)
        self.PVM2 = PVMLayer(64, 32)
        self.PVM3 = PVMLayer(128, 64)

        self.OPU1 = OPU(256, 64)
        self.OPU2 = OPU(128, 32)
        self.OPU3 = OPU(64, 16)

        # 测试以下内容
        self.PVM4 = PVMLayer(64, 32)
        self.PVM5 = PVMLayer(128, 64)
        self.PVM6 = PVMLayer(256, 128)

        self.OPD4 = OPD(32, 32)
        self.OPD5 = OPD(64, 64)
        self.OPD6 = OPD(128, 128)

        self.PVM7 = PVMLayer(128, 64)
        self.PVM8 = PVMLayer(64, 32)
        self.PVM9 = PVMLayer(32, 16)

        self.OPU4 = OPU(256, 64)
        self.OPU5 = OPU(128, 32)
        self.OPU6 = OPU(64, 16)

        self.Tail = Tail(32, 1)

    def forward(self, x):
        x1 = self.Head(x)

        x2 = self.OPD1(x1)
        x3 = self.OPD2(x2)
        x4 = self.OPD3(x3)

        x3 = torch.cat([self.OPU1(x4), self.PVM3(x3)], dim=1)
        x2 = torch.cat([self.OPU2(x3), self.PVM2(x2)], dim=1)
        x1 = torch.cat([self.OPU3(x2), self.PVM1(x1)], dim=1)

        x2 = torch.cat([self.OPD4(x1), self.PVM4(x2)], dim=1)
        x3 = torch.cat([self.OPD5(x2), self.PVM5(x3)], dim=1)
        x4 = torch.cat([self.OPD6(x3), self.PVM6(x4)], dim=1)

        x3 = torch.cat([self.OPU4(x4), self.PVM7(x3)], dim=1)
        x2 = torch.cat([self.OPU5(x3), self.PVM8(x2)], dim=1)
        x1 = torch.cat([self.OPU6(x2), self.PVM9(x1)], dim=1)
        return self.Tail(x1)


if __name__ == "__main__":
    # 测试成功
    device = 'cuda'
    x = torch.ones((2, 3, 224, 224)).to(device)
    model = PMDUNet(3).to(device)
    y_pred = model(x)
    print("hello")
    pass
