import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_


# =========================================================
# Utils
# =========================================================
def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return x * cos + rotate_every_two(x) * sin


# =========================================================
# Faster Window Ops (Unfold/Fold)
# =========================================================
def window_partition(x, ws):
    """
    x: [B, C, H, W]
    return:
        windows: [B*nW, C, ws, ws]
    """
    B, C, H, W = x.shape

    x = F.unfold(x, kernel_size=ws, stride=ws)
    x = x.transpose(1, 2)
    x = x.reshape(-1, C, ws, ws)

    return x


def window_reverse(windows, ws, H, W):
    """
    windows: [B*nW, C, ws, ws]
    """
    B = int(windows.shape[0] / (H * W / ws / ws))
    C = windows.shape[1]

    x = windows.reshape(B, -1, C * ws * ws)
    x = x.transpose(1, 2)

    x = F.fold(
        x,
        output_size=(H, W),
        kernel_size=ws,
        stride=ws
    )

    return x


# =========================================================
# Faster DWConv
# =========================================================
class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size,
            stride,
            padding,
            groups=dim,
            bias=True
        )

    def forward(self, x):
        return self.conv(x)


# =========================================================
# Relative Position
# =========================================================
class RetNetRelPos2d(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            initial_value=1,
            heads_range=3,
            window_size=8
    ):
        super().__init__()

        self.window_size = window_size

        head_dim = embed_dim // num_heads

        angle = 1.0 / (
                10000 ** torch.linspace(0, 1, head_dim // 2)
        )

        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()

        decay = torch.log(
            1 - 2 ** (
                    -initial_value
                    - heads_range
                    * torch.arange(num_heads, dtype=torch.float)
                    / num_heads
            )
        )

        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

        mask_h = self.generate_1d_decay(window_size)
        mask_w = self.generate_1d_decay(window_size)

        self.register_buffer("mask_h", mask_h)
        self.register_buffer("mask_w", mask_w)

        index = torch.arange(window_size * window_size)

        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])

        sin = sin.reshape(window_size, window_size, -1)
        cos = cos.reshape(window_size, window_size, -1)

        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def generate_1d_decay(self, l):
        index = torch.arange(l).to(self.decay)

        mask = (index[:, None] - index[None, :]).abs()
        mask = mask * self.decay[:, None, None]

        return mask

    def forward(self):
        return (self.sin, self.cos), (self.mask_h, self.mask_w)


# =========================================================
# Faster Conv FFN
# =========================================================
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, 1)

        self.dwconv = nn.Conv2d(
            ffn_dim,
            ffn_dim,
            3,
            1,
            1,
            groups=ffn_dim
        )

        self.act = nn.GELU()

        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, 1)

    def forward(self, x):
        identity = x

        x = self.fc1(x)
        x = self.act(x)

        x = self.dwconv(x) + x

        x = self.fc2(x)

        return x


# =========================================================
# Faster Retention
# =========================================================
class WindowedVisionRetention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            window_size=8,
            shift_size=0,
            value_factor=1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.factor = value_factor

        self.key_dim = embed_dim // num_heads
        self.head_dim = embed_dim * value_factor // num_heads

        self.scaling = self.key_dim ** -0.5

        self.qkv_proj = nn.Conv2d(
            embed_dim,
            embed_dim * 2 + embed_dim * value_factor,
            1
        )

        self.lepe = DWConv2d(embed_dim * value_factor, 3, 1, 1)

        self.out_proj = nn.Conv2d(
            embed_dim * value_factor,
            embed_dim,
            1
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.qkv_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)

    def forward(self, x, rel_pos):
        """
        x: [B,C,H,W]
        """

        B, C, H, W = x.shape

        (sin, cos), (mask_h, mask_w) = rel_pos

        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(2, 3)
            )

        x_windows = window_partition(x, self.window_size)

        nWB, _, ws, _ = x_windows.shape

        qkv = self.qkv_proj(x_windows)

        q, k, v = torch.split(
            qkv,
            [
                self.embed_dim,
                self.embed_dim,
                self.embed_dim * self.factor
            ],
            dim=1
        )

        lepe = self.lepe(v)

        q = q.reshape(
            nWB,
            self.num_heads,
            self.key_dim,
            ws,
            ws
        ).permute(0, 1, 3, 4, 2)

        k = k.reshape(
            nWB,
            self.num_heads,
            self.key_dim,
            ws,
            ws
        ).permute(0, 1, 3, 4, 2)

        v = v.reshape(
            nWB,
            self.num_heads,
            self.head_dim,
            ws,
            ws
        ).permute(0, 1, 3, 4, 2)

        k = k * self.scaling

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        # =====================================================
        # width retention
        # =====================================================
        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)

        v_w = v.transpose(1, 2)

        qk_w = qr_w @ kr_w.transpose(-1, -2)

        qk_w = qk_w + mask_w

        qk_w = F.softmax(
            qk_w,
            dim=-1,
            dtype=torch.float32
        ).to(q.dtype)

        v = torch.matmul(qk_w, v_w)

        # =====================================================
        # height retention
        # =====================================================
        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)

        v_h = v.permute(0, 3, 2, 1, 4)

        qk_h = qr_h @ kr_h.transpose(-1, -2)

        qk_h = qk_h + mask_h

        qk_h = F.softmax(
            qk_h,
            dim=-1,
            dtype=torch.float32
        ).to(q.dtype)

        output = torch.matmul(qk_h, v_h)

        output = output.permute(0, 3, 1, 2, 4)

        output = output.reshape(
            nWB,
            self.embed_dim * self.factor,
            ws,
            ws
        )

        output = output + lepe

        output = self.out_proj(output)

        x = window_reverse(
            output,
            self.window_size,
            H,
            W
        )

        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size),
                dims=(2, 3)
            )

        return x


# =========================================================
# Faster Block
# =========================================================
class RetBlock(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            ffn_dim,
            window_size=8,
            shift_size=0,
            drop_path=0.
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(1, embed_dim)

        self.retention = WindowedVisionRetention(
            embed_dim,
            num_heads,
            window_size,
            shift_size
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0 else nn.Identity()
        )

        self.norm2 = nn.GroupNorm(1, embed_dim)

        self.ffn = FeedForwardNetwork(
            embed_dim,
            ffn_dim
        )

        self.pos = DWConv2d(embed_dim, 3, 1, 1)

    def forward(self, x, rel_pos):
        x = x + self.pos(x)

        x = x + self.drop_path(
            self.retention(
                self.norm1(x),
                rel_pos
            )
        )

        x = x + self.drop_path(
            self.ffn(
                self.norm2(x)
            )
        )

        return x


# =========================================================
# Patch Embed
# =========================================================
class PatchEmbed(nn.Module):
    def __init__(
            self,
            in_chans=1,
            embed_dim=64,
            patch_size=4
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        return self.proj(x)


# =========================================================
# Patch Merging
# =========================================================
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.reduction = nn.Conv2d(
            dim * 4,
            dim * 2,
            1,
            bias=False
        )

        self.norm = nn.GroupNorm(1, dim * 4)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]

        x = torch.cat([x0, x1, x2, x3], dim=1)

        x = self.norm(x)

        x = self.reduction(x)

        return x


# =========================================================
# Faster Upsample
# =========================================================
class TokenUpSample(nn.Module):
    def __init__(self, in_dim, scale=2):
        super().__init__()

        if scale == 2:
            self.up = nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    2 * in_dim,
                    1,
                    bias=False
                ),
                nn.GELU(),
                nn.PixelShuffle(2)
            )

            self.out_dim = in_dim // 2

        elif scale == 4:
            self.up = nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    16 * in_dim,
                    1,
                    bias=False
                ),
                nn.GELU(),
                nn.PixelShuffle(4)
            )

            self.out_dim = in_dim

    def forward(self, x):
        return self.up(x)


# =========================================================
# Basic Layer
# =========================================================
class BasicLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_heads,
            window_size=8,
            ffn_ratio=2.0,
            init_value=1,
            heads_range=3,
            drop_path=0.,
            downsample=None,
            use_checkpoint=False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.rel_pos_module = RetNetRelPos2d(
            dim,
            num_heads,
            init_value,
            heads_range,
            window_size
        )

        self.blocks = nn.ModuleList([
            RetBlock(
                embed_dim=dim,
                num_heads=num_heads,
                ffn_dim=int(ffn_ratio * dim),
                window_size=window_size,
                shift_size=0 if (i % 2 == 0)
                else window_size // 2,
                drop_path=drop_path[i]
                if isinstance(drop_path, list)
                else drop_path
            )
            for i in range(depth)
        ])

        self.downsample = (
            downsample(dim)
            if downsample is not None
            else None
        )

    def forward(self, x):
        rel_pos = self.rel_pos_module()

        for blk in self.blocks:

            if self.use_checkpoint:
                x = checkpoint.checkpoint(
                    blk,
                    x,
                    rel_pos,
                    use_reentrant=False
                )
            else:
                x = blk(x, rel_pos)

        x_down = (
            self.downsample(x)
            if self.downsample is not None
            else x
        )

        return x, x_down


# =========================================================
# Basic Layer Up
# =========================================================
class BasicLayerUp(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_heads,
            window_size=8,
            ffn_ratio=2.0,
            init_value=1,
            heads_range=3,
            drop_path=0.,
            use_checkpoint=False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.rel_pos_module = RetNetRelPos2d(
            dim,
            num_heads,
            init_value,
            heads_range,
            window_size
        )

        self.blocks = nn.ModuleList([
            RetBlock(
                embed_dim=dim,
                num_heads=num_heads,
                ffn_dim=int(ffn_ratio * dim),
                window_size=window_size,
                shift_size=0 if (i % 2 == 0)
                else window_size // 2,
                drop_path=drop_path[i]
                if isinstance(drop_path, list)
                else drop_path
            )
            for i in range(depth)
        ])

    def forward(self, x):
        rel_pos = self.rel_pos_module()

        for blk in self.blocks:

            if self.use_checkpoint:
                x = checkpoint.checkpoint(
                    blk,
                    x,
                    rel_pos,
                    use_reentrant=False
                )
            else:
                x = blk(x, rel_pos)

        return x

class RUNet(nn.Module):
    def __init__(
            self,
            inp_channels=1,
            out_channels=1,
            dim=64,
            patch_size=4,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            window_size=8,
            use_checkpoint=False,
            drop_path_rate=0.1
    ):
        super().__init__()

        self.conv_first = nn.Conv2d(
            inp_channels,
            dim,
            3,
            1,
            1
        )

        self.patch_embed = PatchEmbed(
            in_chans=dim,
            embed_dim=dim,
            patch_size=patch_size
        )

        dpr = [
            x.item()
            for x in torch.linspace(
                0,
                drop_path_rate,
                sum(num_blocks) * 2
                + num_refinement_blocks
            )
        ]

        # =====================================================
        # Encoder
        # =====================================================
        self.encoder1 = BasicLayer(
            dim=dim,
            depth=num_blocks[0],
            num_heads=max(1, dim // 32),
            window_size=window_size,
            drop_path=dpr[0:num_blocks[0]],
            downsample=PatchMerging,
            use_checkpoint=use_checkpoint
        )

        self.encoder2 = BasicLayer(
            dim=dim * 2,
            depth=num_blocks[1],
            num_heads=max(1, (dim * 2) // 32),
            window_size=window_size,
            drop_path=dpr[num_blocks[0]:sum(num_blocks[:2])],
            downsample=PatchMerging,
            use_checkpoint=use_checkpoint
        )

        self.encoder3 = BasicLayer(
            dim=dim * 4,
            depth=num_blocks[2],
            num_heads=max(1, (dim * 4) // 32),
            window_size=window_size,
            drop_path=dpr[sum(num_blocks[:2]):sum(num_blocks[:3])],
            downsample=PatchMerging,
            use_checkpoint=use_checkpoint
        )

        self.latent = BasicLayer(
            dim=dim * 8,
            depth=num_blocks[3],
            num_heads=max(1, (dim * 8) // 32),
            window_size=window_size,
            drop_path=dpr[sum(num_blocks[:3]):sum(num_blocks[:4])],
            downsample=None,
            use_checkpoint=use_checkpoint
        )

        # =====================================================
        # Decoder
        # =====================================================
        self.up3 = TokenUpSample(dim * 8, scale=2)

        self.dec3_reduce = nn.Conv2d(
            dim * 8,
            dim * 4,
            1
        )

        self.dec3 = BasicLayerUp(
            dim=dim * 4,
            depth=num_blocks[2],
            num_heads=max(1, (dim * 4) // 32),
            window_size=window_size,
            use_checkpoint=use_checkpoint
        )

        self.up2 = TokenUpSample(dim * 4, scale=2)

        self.dec2_reduce = nn.Conv2d(
            dim * 4,
            dim * 2,
            1
        )

        self.dec2 = BasicLayerUp(
            dim=dim * 2,
            depth=num_blocks[1],
            num_heads=max(1, (dim * 2) // 32),
            window_size=window_size,
            use_checkpoint=use_checkpoint
        )

        self.up1 = TokenUpSample(dim * 2, scale=2)

        self.dec1_reduce = nn.Conv2d(
            dim * 2,
            dim,
            1
        )

        self.dec1 = BasicLayerUp(
            dim=dim,
            depth=num_blocks[0],
            num_heads=max(1, dim // 32),
            window_size=window_size,
            use_checkpoint=use_checkpoint
        )

        self.refinement = BasicLayerUp(
            dim=dim,
            depth=num_refinement_blocks,
            num_heads=max(1, dim // 32),
            window_size=window_size,
            use_checkpoint=use_checkpoint
        )

        self.final_up = TokenUpSample(dim, scale=4)

        self.output = nn.Conv2d(
            dim,
            out_channels,
            3,
            1,
            1
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)

            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

            if getattr(m, "weight", None) is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, inp_img):
        residual = inp_img[:,:1,:,:]

        x = self.conv_first(inp_img)

        x = self.patch_embed(x)

        # =====================================================
        # Encoder
        # =====================================================
        skip1, x = self.encoder1(x)

        skip2, x = self.encoder2(x)

        skip3, x = self.encoder3(x)

        x, _ = self.latent(x)

        # =====================================================
        # Decoder
        # =====================================================
        x = self.up3(x)

        x = torch.cat([x, skip3], dim=1)

        x = self.dec3_reduce(x)

        x = self.dec3(x)

        x = self.up2(x)

        x = torch.cat([x, skip2], dim=1)

        x = self.dec2_reduce(x)

        x = self.dec2(x)

        x = self.up1(x)

        x = torch.cat([x, skip1], dim=1)

        x = self.dec1_reduce(x)

        x = self.dec1(x)

        x = self.refinement(x)

        x = self.final_up(x)

        out = self.output(x) + residual

        return out,x


# =========================================================
# Count Params
# =========================================================
def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =========================================================
# Test
# =========================================================
if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import time
    from thop import profile, clever_format

    torch.backends.cudnn.benchmark = True

    model = RUNet(
        img_size=256,
        inp_channels=1,
        out_channels=1,
        dim=64,
        patch_size=4,
        window_size=8,
        use_checkpoint=False
    ).cuda()

    model = model.to(memory_format=torch.channels_last)

    x = torch.zeros(
        (1, 1, 256, 256),
        device="cuda"
    ).to(memory_format=torch.channels_last)

    # warmup
    for _ in range(20):
        _ = model(x)

    torch.cuda.synchronize()

    since = time.time()

    with torch.no_grad():
        y = model(x)

    torch.cuda.synchronize()

    print(f"前向时间: {time.time() - since:.4f}s")

    print(f"输入形状: {x.shape}")

    print(f"输出形状: {y.shape}")

    print(
        f"参数量: "
        f"{count_parameters(model)/1e6:.2f}M"
    )

    flops, params = profile(
        model,
        inputs=(x,),
        verbose=False
    )

    flops, params = clever_format(
        [flops, params],
        "%.6f"
    )

    print(f"FLOPs: {flops}")

    print(f"Params: {params}")