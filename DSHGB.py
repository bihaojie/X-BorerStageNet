class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x
class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class GhostConv_V2(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 3, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

class DSH(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',
            drop_path=0.,
    ):
        super().__init__()
        self.residual = residual
        self.layers_1 = nn.ModuleList()

        if light_block:

            self.layers_1.extend([
                LightConvBNAct(
                    in_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size,
                    use_lab=use_lab,
                ),
                LightConvBNAct(
                    mid_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size + 2,
                    use_lab=use_lab,
                ),
                LightConvBNAct(
                    mid_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size - 2,
                    use_lab=use_lab,
                )]
            )
        else:
            self.layers_1.extend([
                ConvBNAct(
                    in_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size,
                    stride=1,
                    use_lab=use_lab,
                ),
                ConvBNAct(
                    mid_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size + 2,
                    stride=1,
                    use_lab=use_lab,
                ),
                ConvBNAct(
                    mid_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size - 2,
                    stride=1,
                    use_lab=use_lab,
                )]
            )

        self.layers_2 = nn.ModuleList()
        if light_block:
            self.layers_2.extend([
                LightConvBNAct(
                    in_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size,
                    use_lab=use_lab,
                ),
                LightConvBNAct(
                    mid_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size,
                    use_lab=use_lab,
                ),
                LightConvBNAct(
                    mid_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size,
                    use_lab=use_lab,
                )]
            )
        else:
            self.layers_2.extend([
                ConvBNAct(
                    in_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size,
                    stride=1,
                    use_lab=use_lab,
                ),
                ConvBNAct(
                    mid_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size,
                    stride=1,
                    use_lab=use_lab,
                ),
                ConvBNAct(
                    mid_chs // 2,
                    mid_chs // 2,
                    kernel_size=kernel_size,
                    stride=1,
                    use_lab=use_lab,
                )]
            )
        total_chs = in_chs + layer_num * mid_chs

        if agg == 'se':
            self.aggregation = GhostConv_V2(total_chs, out_chs, k=1, s=1)

    def forward(self, x):
        identity = x
        x1, x2 = x.chunk(2, dim=1)
        output = [x1, x2]
        for layer_1 in self.layers_1:
            x1 = layer_1(x1)
            output.append(x1)

        for layer_2 in self.layers_2:
            x2 = layer_2(x2)
            output.append(x2)

        x = torch.cat(output, dim=1)

        x = self.aggregation(x)

        if self.residual:
            x = self.drop_path(x) + identity

        return x


class DSHGB(nn.Module):

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0.,
    ):
        super().__init__()
        self.downsample = downsample

        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
        else:
            self.downsample = nn.Identity()


        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                DSH(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):

        x = self.downsample(x)
        x = self.blocks(x)
        return x