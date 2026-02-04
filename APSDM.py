class PSConv(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()

        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))

class APSDM(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )

        self.stem2 = PSConv(mid_chs, mid_chs, 2, 1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

        self.stem3 = ConvBNAct(
            mid_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )

    def forward(self, x):
        x = self.stem1(x)

        x = self.stem2(x)

        x = F.pad(x, (0, 1, 0, 1))

        x = self.pool(x)

        x = self.stem3(x)

        return x