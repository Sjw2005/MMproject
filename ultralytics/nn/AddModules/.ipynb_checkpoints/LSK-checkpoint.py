# import torch
# import torch.nn as nn

# class LSK(nn.Module):
#     """
#     Drone_LSK: 专为无人机小目标+双模态视差优化的 LSK 融合模块
#     适配 YOLO 接口: __init__(self, c1, dim, ...)
#     """
#     def __init__(self, c1, dim):
#         super().__init__()
#         # c1 是 YOLO 自动传进来的 [ch_in_vis, ch_in_ir]，我们主要用 dim
#         self.dim = dim
        
#         # Branch 1: 小核 (3x3)，专注 RGB 细节/纹理
#         # groups=dim 降低计算量 (Depthwise)
#         self.conv_detail = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        
#         # Branch 2: 大核 (7x7, d=3 -> RF=19)，专注 IR 轮廓/上下文/容忍视差
#         self.conv_context = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        
#         # 混合特征以生成 Attention
#         self.conv_mix = nn.Conv2d(dim * 2, dim, 1) 
        
#         # 空间选择门控 (Spatial Selector)
#         # 输出 2 个通道，分别给 detail 分支和 context 分支加权
#         self.spatial_gate = nn.Sequential(
#             nn.Conv2d(dim, 2, 7, padding=3),
#             nn.Sigmoid()
#         )
        
#         # 最终融合层
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(dim * 2, dim, 1, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.SiLU()
#         )

#     def forward(self, data):
#         # 假设输入是一个列表 [rgb, ir]
#         rgb, ir = data
        
#         # 1. 特征提取
#         feat_detail = self.conv_detail(rgb)   # 细节特征
#         feat_context = self.conv_context(ir)  # 上下文特征
        
#         # 2. 生成 LSK 注意力
#         raw_mix = torch.cat([feat_detail, feat_context], dim=1)
#         mixed_feat = self.conv_mix(raw_mix)
        
#         # 生成权重 [B, 2, H, W]
#         gate = self.spatial_gate(mixed_feat)
        
#         # 3. 加权得到 "互补特征"
#         # gate[:, 0:1] 选细节, gate[:, 1:2] 选上下文
#         attn_feat = feat_detail * gate[:, 0:1] + feat_context * gate[:, 1:2]
        
#         # 4. 残差注入 (关键步骤)
#         # 将 LSK 提取的互补特征作为增益，分别加回 RGB 和 IR
#         # 这样即使 Attention 算得不好，原始特征还在
#         rgb_enhanced = rgb + attn_feat
#         ir_enhanced = ir + attn_feat
        
#         # 5. 最终深度融合
#         out = self.out_conv(torch.cat([rgb_enhanced, ir_enhanced], dim=1))
        
#         return out

#加
import torch
import torch.nn as nn

class LSK(nn.Module):
    def __init__(self, c1, dim):
        super().__init__()
        self.dim = dim
        
        # 1. 特征提取 (保持不变)
        self.conv_detail = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) # RGB分支
        self.conv_context = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3) # IR分支
        
        # 2. 生成 Gate (保持不变)
        self.conv_mix = nn.Conv2d(dim * 2, dim, 1) 
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, 2, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 3. 核心修改：融合层改为 RGB 注入层
        # 我们不再 Concat 输出，而是把 RGB 注入到 IR 里
        self.rgb_reduce = nn.Conv2d(dim, dim, 1) # 调整 RGB 特征
        
        # 4. 关键：可学习的融合权重，初始化为 0
        # 这意味着训练刚开始时，RGB 的贡献为 0，模型 = 纯 IR 模型
        self.fusion_weight = nn.Parameter(torch.zeros(1)) 

    def forward(self, data):
        # x: RGB (弱), y: IR (强)
        rgb, ir = data
        
        # 1. LSK 提取互补特征 (Attention mask)
        feat_detail = self.conv_detail(rgb)
        feat_context = self.conv_context(ir)
        
        raw_mix = torch.cat([feat_detail, feat_context], dim=1)
        mixed_feat = self.conv_mix(raw_mix)
        gate = self.spatial_gate(mixed_feat) # [B, 2, H, W]
        
        # 2. 计算 RGB 分支的有效信息
        # 我们只关心 RGB 中那些“被 Gate 选中”的纹理细节
        rgb_useful = feat_detail * gate[:, 0:1] 
        
        # 3. 最终融合：以 IR 为底座，加权注入 RGB
        # out = IR + weight * (RGB_useful)
        # 初始时刻 weight=0，保证不掉点
        out = ir + self.fusion_weight * self.rgb_reduce(rgb_useful)
        
        return out
