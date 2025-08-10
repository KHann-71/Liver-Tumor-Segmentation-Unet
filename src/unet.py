import torch
import torch.nn as nn


def conv_block(ic: int, oc: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1, bias=False),
        nn.BatchNorm2d(oc),
        nn.ReLU(inplace=True),
        nn.Conv2d(oc, oc, 3, padding=1, bias=False),
        nn.BatchNorm2d(oc),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 64):
        super().__init__()
        self.e1 = conv_block(in_channels, base)
        self.p1 = nn.MaxPool2d(2)

        self.e2 = conv_block(base, base * 2)
        self.p2 = nn.MaxPool2d(2)

        self.e3 = conv_block(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)

        self.e4 = conv_block(base * 4, base * 8)
        self.p4 = nn.MaxPool2d(2)

        self.bott = conv_block(base * 8, base * 16)

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.d4 = conv_block(base * 16, base * 8)

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = conv_block(base * 8, base * 4)

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = conv_block(base * 4, base * 2)

        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = conv_block(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        e4 = self.e4(self.p3(e3))

        b  = self.bott(self.p4(e4))

        d4 = self.u4(b); d4 = self.d4(torch.cat([d4, e4], dim=1))
        d3 = self.u3(d4); d3 = self.d3(torch.cat([d3, e3], dim=1))
        d2 = self.u2(d3); d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = self.u1(d2); d1 = self.d1(torch.cat([d1, e1], dim=1))

        return self.out(d1)
