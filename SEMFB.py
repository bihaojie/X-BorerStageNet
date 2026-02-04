class WindowFrequencyModulation(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.ratio = 1
        self.complex_weight = nn.Parameter(torch.cat(
            (torch.ones(self.window_size, self.window_size // 2 + 1, self.ratio * dim, 1, dtype=torch.float32), \
             torch.zeros(self.window_size, self.window_size // 2 + 1, self.ratio * dim, 1, dtype=torch.float32)),
            dim=-1))

    def forward(self, x):
        x = rearrange(x, 'b c (w1 p1) (w2 p2) -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(3, 4), norm='ortho')

        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.window_size, self.window_size), dim=(3, 4), norm='ortho')

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b c (w1 p1) (w2 p2)')
        return x

class FMFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, window_size=4, act_layer=nn.GELU) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.ffn = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            act_layer(),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.fm = WindowFrequencyModulation(out_features, window_size)

    def forward(self, x):
        return self.fm(self.ffn(x))


class GSA(nn.Module):
    def __init__(self, channels, num_heads=8, bias=False):
        super(GSA, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.act(attn)
        out = (attn @ v)
        y = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        y = rearrange(y, 'b (head c) h w -> b (c head) h w', head=self.num_heads, h=h, w=w)
        y = self.project_out(y)
        return y

class SEMFB(nn.Module):

    def __init__(self, in_dim, dim,
                 token_mixer=GSA, mlp=FMFFN,
                 norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6),
                 drop_path=0., mlp_ratio=2,
                 layer_scale_init_value=None, res_scale_init_value=None, selfatt=False
                 ):

        super().__init__()

        self.norm1 = norm_layer((dim, 1, 1))
        if selfatt:
            self.token_mixer = token_mixer(dim)
        else:
            self.token_mixer = token_mixer(dim, dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer((dim, 1, 1))
        self.mlp = mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.conv1x1 = Conv(in_dim, dim, 1) if in_dim != dim else nn.Identity()

    def forward(self, x):
        x = self.conv1x1(x)
        # x size: [B, C, H, W]
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x