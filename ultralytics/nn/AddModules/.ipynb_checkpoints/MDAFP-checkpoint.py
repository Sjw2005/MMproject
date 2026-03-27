import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class ResidualScale(nn.Module):
    """可学习残差缩放，防止注意力输出过强导致不稳定"""
    def __init__(self, init=1.0):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(init, dtype=torch.float32))
    def forward(self, x):
        return x * self.s

class ChannelGate(nn.Module):
    """通道级门控（SE风格）"""
    def __init__(self, dim, r=16):
        super().__init__()
        hidden = max(dim // r, 8)
        self.g = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, dim, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.g(x)

class PixelGate(nn.Module):
    """像素级可靠性门控：g∈(0,1)，fuse = g*x + (1-g)*y"""
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x, y):
        gate = self.g(torch.cat([x, y], dim=1))
        return gate, gate * x + (1.0 - gate) * y

class FusionStrength(nn.Module):
    """动态融合强度 α：决定这层融合用力多少，防止过融合"""
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // 4 if dim >= 64 else 16, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(dim // 4 if dim >= 64 else 16, 1, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x, y):
        # α ∈ (0,1)，标量（每张图一个）
        return self.a(torch.cat([x, y], dim=1))

class MDAFP(nn.Module):
    """
    MDAF++ / MDAFv3:
    - 多尺度方向DWConv (局部方向上下文)
    - 双向轴向 cross-attn (H/W + 双向)
    - 像素级 gate + 通道 gate
    - 动态融合强度 alpha (防止过融合)
    - 残差缩放 (更稳)
    """
    def __init__(self, dim, num_heads=8, LayerNorm_type='WithBias',
                 k_list=(7, 11, 21), attn_drop=0.0, lite=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.lite = lite  # lite=True 时可弱化P5的交互（建议P5用lite）

        self.norm_x = LayerNorm(dim, LayerNorm_type)
        self.norm_y = LayerNorm(dim, LayerNorm_type)

        # 多尺度方向DWConv：每个k生成 1×k + k×1
        def make_dir_convs():
            conv_1xk = nn.ModuleList([nn.Conv2d(dim, dim, (1, k), padding=(0, k//2), groups=dim) for k in k_list])
            conv_kx1 = nn.ModuleList([nn.Conv2d(dim, dim, (k, 1), padding=(k//2, 0), groups=dim) for k in k_list])
            return conv_1xk, conv_kx1

        self.x_1xk, self.x_kx1 = make_dir_convs()
        self.y_1xk, self.y_kx1 = make_dir_convs()

        self.proj_x = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_y = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_h = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_w = nn.Conv2d(dim, dim, 1, bias=False)

        # 门控与强度
        self.pixel_gate = PixelGate(dim)
        self.chan_gate = ChannelGate(dim, r=16)
        self.alpha = FusionStrength(dim)

        # 残差缩放
        self.rs_attn = ResidualScale(1.0)
        self.rs_fuse = ResidualScale(1.0)

        self.out = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )

    def _dir_mix(self, z, conv_1xk, conv_kx1):
        out = 0
        for m in conv_1xk:
            out = out + m(z)
        for m in conv_kx1:
            out = out + m(z)
        return out

    def _attn_axis(self, q, k, v):
        # q,k,v: [B, head, L, D]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn @ v

    def forward(self, data):
        x, y = data
        x0, y0 = x, y

        x = self.norm_x(x)
        y = self.norm_y(y)
        B, C, H, W = x.shape

        # 1) 方向上下文
        x_dir = self.proj_x(self._dir_mix(x, self.x_1xk, self.x_kx1))
        y_dir = self.proj_y(self._dir_mix(y, self.y_1xk, self.y_kx1))

        # 2) 轴向双向 cross-attn（lite 模式可只做一半/只做一向）
        # y -> x (H axis)
        q_h = rearrange(y_dir, 'b (h c) H W -> b h H (W c)', h=self.num_heads)
        k_h = rearrange(x_dir, 'b (h c) H W -> b h H (W c)', h=self.num_heads)
        v_h = rearrange(x_dir, 'b (h c) H W -> b h H (W c)', h=self.num_heads)
        out_h = self._attn_axis(q_h, k_h, v_h) + q_h
        out_h = rearrange(out_h, 'b h H (W c) -> b (h c) H W', h=self.num_heads, H=H, W=W)

        # y -> x (W axis)
        q_w = rearrange(y_dir, 'b (h c) H W -> b h W (H c)', h=self.num_heads)
        k_w = rearrange(x_dir, 'b (h c) H W -> b h W (H c)', h=self.num_heads)
        v_w = rearrange(x_dir, 'b (h c) H W -> b h W (H c)', h=self.num_heads)
        out_w = self._attn_axis(q_w, k_w, v_w) + q_w
        out_w = rearrange(out_w, 'b h W (H c) -> b (h c) H W', h=self.num_heads, H=H, W=W)

        x_att = self.proj_h(out_h) + self.proj_w(out_w)

        if not self.lite:
            # x -> y (H axis)
            q_h2 = rearrange(x_dir, 'b (h c) H W -> b h H (W c)', h=self.num_heads)
            k_h2 = rearrange(y_dir, 'b (h c) H W -> b h H (W c)', h=self.num_heads)
            v_h2 = rearrange(y_dir, 'b (h c) H W -> b h H (W c)', h=self.num_heads)
            out_h2 = self._attn_axis(q_h2, k_h2, v_h2) + q_h2
            out_h2 = rearrange(out_h2, 'b h H (W c) -> b (h c) H W', h=self.num_heads, H=H, W=W)

            # x -> y (W axis)
            q_w2 = rearrange(x_dir, 'b (h c) H W -> b h W (H c)', h=self.num_heads)
            k_w2 = rearrange(y_dir, 'b (h c) H W -> b h W (H c)', h=self.num_heads)
            v_w2 = rearrange(y_dir, 'b (h c) H W -> b h W (H c)', h=self.num_heads)
            out_w2 = self._attn_axis(q_w2, k_w2, v_w2) + q_w2
            out_w2 = rearrange(out_w2, 'b h W (H c) -> b (h c) H W', h=self.num_heads, H=H, W=W)

            y_att = self.proj_h(out_h2) + self.proj_w(out_w2)
        else:
            # lite 模式：不做反向交互，降低P5过融合风险
            y_att = 0

        # 3) 加残差缩放（更稳）
        x_enh = x + self.rs_attn(x_att)
        y_enh = y + self.rs_attn(y_att) if not self.lite else y

        # 4) 像素级可靠性融合 + 通道 gate
        gate_map, fused = self.pixel_gate(x_enh, y_enh)
# ✅ RDR: cache reliability gate for loss (detach to avoid cheating)
        self.last_gate = gate_map.mean(1, keepdim=True).detach()  # [B,1,H,W]
        fused = self.chan_gate(fused)

        # 5) 动态融合强度：α控制“融合占比”，避免过融合
        a = self.alpha(x_enh, y_enh)  # [B,1,1,1]
        base = 0.5 * (x_enh + y_enh)
        fused = a * fused + (1.0 - a) * base

        # 6) 输出 + 残差（保细节）
        out = self.out(fused)
        return out + self.rs_fuse(base)  


