# import torch
# import torch.nn as nn
# from einops import rearrange

# def autopad(k, p=None, d=1):
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p

# class Conv(nn.Module):
#     default_act = nn.SiLU()
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))


# class LAEF(nn.Module):
#     def __init__(self, ch, ch_out=None, group=8):
#         super().__init__()
#         # 如果没有指定输出通道，默认等于输入通道（这是 FeatureAdd 的前提）
#         if ch_out is None:
#             ch_out = ch

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.ca_mlp = nn.Sequential(
#             nn.Conv2d(ch * 2, ch, 1), # 融合两路特征
#             nn.ReLU(),
#             nn.Conv2d(ch, ch * 2, 1), # 生成两路的通道权重
#             nn.Sigmoid()
#         )

#         # 2. 空间注意力分支 (Spatial Attention) - 借鉴了 MM_LAE 思想
#         # 不做下采样，而是提取局部上下文
#         self.spatial_mix = nn.Sequential(
#             Conv(ch * 2, ch, k=3, s=1),      # 先混合两路特征
#             Conv(ch, 2, k=7, s=1, p=3, act=False), # 大感受野生成空间权重 map (2通道)
#             nn.Sigmoid()
#         )

#         # 3. 特征融合层
#         self.fusion_conv = Conv(ch * 2, ch_out, k=1, s=1)

#     def forward(self, x):
#         rgb, ir = x[0], x[1]
#         cat_feat = torch.cat([rgb, ir], dim=1) # [B, 2C, H, W]
#         global_desc = self.gap(cat_feat)       # [B, 2C, 1, 1]
#         ch_weights = self.ca_mlp(global_desc)  # [B, 2C, 1, 1]
#         w_rgb_c, w_ir_c = torch.chunk(ch_weights, 2, dim=1) # 分割权重
        
#         # 通道加权
#         rgb_c = rgb * w_rgb_c
#         ir_c  = ir  * w_ir_c
        
#         mix_feat = torch.cat([rgb_c, ir_c], dim=1)
#         spatial_weights = self.spatial_mix(mix_feat) # [B, 2, H, W]
        
#         w_rgb_s = spatial_weights[:, 0:1, :, :] # RGB 的空间权重
#         w_ir_s  = spatial_weights[:, 1:2, :, :] # IR 的空间权重
        
#         # 空间加权
#         rgb_final = rgb_c * w_rgb_s
#         ir_final  = ir_c  * w_ir_s
        
#         out = self.fusion_conv(torch.cat([rgb_final, ir_final], dim=1))
        
#         return out


#加   RGB基准,对其IR
# import torch
# import torch.nn as nn
# import torchvision.ops as ops

# def autopad(k, p=None, d=1):
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p

# class Conv(nn.Module):
#     default_act = nn.SiLU()
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

# class LAEF(nn.Module):
#     def __init__(self, ch, ch_out=None, kernel_size=3):
#         super().__init__()
#         if ch_out is None:
#             ch_out = ch
        
#         self.kernel_size = kernel_size
#         self.padding = kernel_size // 2
        
#         # --- 1. 对齐模块 (Alignment) ---
#         # 计算偏移量 (offset) 和 调制掩码 (mask)
#         # 输入是 cat(rgb, ir)，输出 offset (2*k*k) 和 mask (k*k)
#         # 这里的 offset 是为了让 IR 对齐到 RGB
#         self.offset_mask_conv = nn.Sequential(
#             nn.Conv2d(ch * 2, ch, 3, padding=1, bias=True),
#             nn.SiLU(),
#             # 输出通道: 2*k*k (offset) + k*k (mask)
#             nn.Conv2d(ch, 3 * kernel_size * kernel_size, 3, padding=1, bias=True)
#         )
        
#         # DCNv2 层 (本身不带权重，只做重采样，所以我们需要一个卷积层来承载权重)
#         # 这里定义一个标准的 DCNv2 包装层
#         self.dcn_conv = ops.DeformConv2d(
#             in_channels=ch,
#             out_channels=ch,
#             kernel_size=kernel_size,
#             padding=self.padding,
#             bias=False
#         )
#         # DCN 的权重需要单独定义，因为 ops.DeformConv2d 只是一个操作符(在新版torchvision中)
#         # 或者使用 nn.Parameter 定义权重，这里为了方便直接使用 DCN 层的 weight 属性（如果它是Module）
#         # 注意：torchvision.ops.DeformConv2d 是 Module，自带 weight 参数
        
#         # --- 2. 通道注意力 (保持原逻辑) ---
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.ca_mlp = nn.Sequential(
#             nn.Conv2d(ch * 2, ch, 1), 
#             nn.ReLU(),
#             nn.Conv2d(ch, ch * 2, 1), 
#             nn.Sigmoid()
#         )

#         # --- 3. 空间注意力 (保持原逻辑，但用对齐后的特征) ---
#         self.spatial_mix = nn.Sequential(
#             Conv(ch * 2, ch, k=3, s=1),
#             Conv(ch, 2, k=7, s=1, p=3, act=False), # k=7 大感受野
#             nn.Sigmoid()
#         )

#         # --- 4. 最终融合 ---
#         self.fusion_conv = Conv(ch * 2, ch_out, k=1, s=1)

#     def forward(self, x):
#         rgb, ir = x[0], x[1]
        
#         # ===========================
#         # Step 1: Deformable Alignment
#         # ===========================
#         # 拼接 RGB 和 IR，计算偏移量
#         # 假设以 RGB 为基准，我们将 IR 对齐到 RGB
#         feat_cat = torch.cat([rgb, ir], dim=1)
#         out = self.offset_mask_conv(feat_cat)
        
#         # 拆分 offset 和 mask
#         k2 = self.kernel_size * self.kernel_size
#         offset = out[:, :2*k2, :, :]
#         mask = out[:, 2*k2:, :, :]
#         mask = torch.sigmoid(mask) # mask 必须在 [0, 1] 之间

#         # 对 IR 进行 DCN 对齐
#         # 输入: ir (待校正特征), offset, weight (自带), mask
#         ir_aligned = self.dcn_conv(ir, offset, mask)
        
#         # 现在我们有了 rgb 和 ir_aligned
        
#         # ===========================
#         # Step 2: Channel Attention
#         # ===========================
#         # 使用对齐后的特征重新拼接
#         cat_feat_aligned = torch.cat([rgb, ir_aligned], dim=1)
        
#         global_desc = self.gap(cat_feat_aligned)
#         ch_weights = self.ca_mlp(global_desc)
#         w_rgb_c, w_ir_c = torch.chunk(ch_weights, 2, dim=1)
        
#         rgb_c = rgb * w_rgb_c
#         ir_c  = ir_aligned * w_ir_c # 注意这里用的是对齐后的 IR
        
#         # ===========================
#         # Step 3: Spatial Attention
#         # ===========================
#         mix_feat = torch.cat([rgb_c, ir_c], dim=1)
#         spatial_weights = self.spatial_mix(mix_feat)
        
#         w_rgb_s = spatial_weights[:, 0:1, :, :]
#         w_ir_s  = spatial_weights[:, 1:2, :, :]
        
#         rgb_final = rgb_c * w_rgb_s
#         ir_final  = ir_c  * w_ir_s
        
#         # ===========================
#         # Step 4: Final Fusion
#         # ===========================
#         out = self.fusion_conv(torch.cat([rgb_final, ir_final], dim=1))
        
#         return out


#加 第三版

# import torch
# import torch.nn as nn
# import torchvision.ops as ops

# class LAEF(nn.Module):
#     """
#     Cross-Modality Deformable Fusion
#     思路修正：
#     1. 实验表明 RGB 特征空间更有利 (RGB做主干好)，因此我们以 RGB 为锚点。
#     2. 将 IR 通过 DCN 对齐到 RGB (IR -> RGB)。
#     3. 使用 Concat + Conv 的方式进行深度融合，而不是简单的残差叠加。
#        让网络自己学习 RGB 纹理和 IR 结构的组合方式。
#     """
#     def __init__(self, ch, stride=1):
#         super().__init__()
        
#         # 1. 对齐模块：IR -> RGB
#         # 注意：这里我们让 IR 去适应 RGB
#         self.offset_conv = nn.Sequential(
#             nn.Conv2d(ch * 2, ch, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(ch, 3 * 3 * 3, 3, padding=1, bias=True) # 3*k*k for offset(2)+mask(1)
#         )
#         # 零初始化，保证初始状态平稳
#         nn.init.zeros_(self.offset_conv[-1].weight)
#         nn.init.zeros_(self.offset_conv[-1].bias)
        
#         self.dcn = ops.DeformConv2d(ch, ch, 3, padding=1)

#         # 2. 增强型注意力 (Attention Refinement)
#         # 用对齐后的 IR 来生成注意力，净化 RGB
#         self.attention = nn.Sequential(
#             nn.Conv2d(ch * 2, ch // 2, 1),
#             nn.SiLU(),
#             nn.Conv2d(ch // 2, 2, 1), # 输出两通道：一个给RGB加权，一个给IR加权
#             nn.Sigmoid()
#         )

#         # 3. 深度融合层 (Feature Reconstruction)
#         # 不再强制 IR+RGB，而是 Concat 后重构
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(ch * 2, ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(ch),
#             nn.SiLU()
#         )
        
#         # 4. 快捷连接 (Identity Shortcut)
#         # 既然 RGB 做主干好，我们保留 RGB 的残差
#         self.shortcut = nn.Identity()

#     def forward(self, x):
#         rgb, ir = x[0], x[1] # 假设输入顺序
        
#         # --- Step 1: Align IR to RGB ---
#         # 拼接两者预测 offset
#         cat_feat = torch.cat([rgb, ir], dim=1)
#         out = self.offset_conv(cat_feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
        
#         # 将 IR 对齐到 RGB
#         ir_aligned = self.dcn(ir, offset, mask)
        
#         # --- Step 2: Attention Interaction ---
#         # 再次拼接 RGB 和 对齐后的 IR
#         fusion_cat = torch.cat([rgb, ir_aligned], dim=1)
        
#         # 计算互补注意力
#         att_map = self.attention(fusion_cat)
#         att_rgb, att_ir = att_map[:, 0:1], att_map[:, 1:2]
        
#         # --- Step 3: Weighted Fusion & Reconstruction ---
#         # RGB 既然重要，我们用 IR 告诉 RGB 哪里该增强 (att_rgb)
#         # IR 既然强，我们用 att_ir 提取 IR 的特征
#         # 然后 Concat 让卷积层去重组它们
        
#         rgb_enhanced = rgb * att_rgb
#         ir_enhanced = ir_aligned * att_ir
        
#         out = self.fusion_conv(torch.cat([rgb_enhanced, ir_enhanced], dim=1))
        
#         # --- Step 4: Residual from RGB ---
#         return out + self.shortcut(ir)

import torch
import torch.nn as nn
import torchvision.ops as ops

class LAEF(nn.Module):
    """
    改进版LAEF：
    - IR -> RGB 对齐（DCN）
    - 可靠性门控：决定注入多少RGB信息
    - 输出残差以 IR 为主（避免RGB差时拖累）
    """
    def __init__(self, ch):
        super().__init__()
        self.ch = ch

        # 1) 预测 offset + mask，用于把 IR 对齐到 RGB
        self.offset_conv = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, 3 * 3 * 3, 3, padding=1, bias=True)  # (o1,o2,mask) each 3x3
        )
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)

        self.dcn = ops.DeformConv2d(ch, ch, 3, padding=1)

        # 2) 轻量“可靠性门控” g：决定注入RGB多少（每像素一个g）
        # g≈1 => 更信IR；g≈0 => 更信RGB补充（你也可以反过来）
        mid = max(ch // 8, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(ch * 2, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, 1, 1, bias=True)
        )
        # 初始化让 g 高一些 => 初始更偏向 IR（sigmoid(2)=0.88）
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 1.0)

        # 3) 融合重构（把 IR + RGB补充拼接后重构成ch通道）
        self.fusion = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU()
        )

        # 4) 输出残差缩放（防止融合过强）
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def forward(self, x):
        rgb, ir = x[0], x[1]

        # --- Align IR to RGB ---
        cat = torch.cat([rgb, ir], dim=1)
        out = self.offset_conv(cat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = torch.sigmoid(mask)
        ir_aligned = self.dcn(ir, offset, mask)

        # --- Reliability gate ---
        g = torch.sigmoid(self.gate(torch.cat([rgb, ir_aligned], dim=1)))  # [B,1,H,W]
        # g 高 => 更信 IR；注入的RGB补充为 (1-g)*rgb
        rgb_supp = (1.0 - g) * rgb
        ir_base  = g * ir_aligned

        # --- Reconstruct ---
        fused = self.fusion(torch.cat([ir_base, rgb_supp], dim=1))

        # --- IR-dominant residual (退化保护) ---
        # 你也可以用 ir（未对齐）做残差：看你对齐质量
        return fused + self.res_scale * ir_aligned



