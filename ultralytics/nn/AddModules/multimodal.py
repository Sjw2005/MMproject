import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.data.multimodal import normalize_channel_order

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)    #全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MF(nn.Module):     #Multi-modal Fusion，双模态融合模块
    def __init__(self, c1, c2, reduction=16):   # c1：输入特征图的通道数； c2：输出特征图的通道数
        super(MF, self).__init__()
        # 掩码卷积（mask_map）：给RGB、红外特征分别生成“注意力掩码”
        # c1//2：假设输入c1是双模态通道和，则各分3通道；输出1通道掩码
        self.mask_map_r = nn.Conv2d(c1//2, 1, 1, 1, 0, bias=True)  #rgb
        self.mask_map_i = nn.Conv2d(c1//2, 1, 1, 1, 0, bias=True)  #ir
        self.softmax = nn.Softmax(-1)
        self.bottleneck1 = nn.Conv2d(c1//2, c2//2, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(c1//2, c2//2, 3, 1, 1, bias=False)
        self.se = SE_Block(c2, reduction)

    def forward(self, x):   #拆分→掩码加权→特征增强→融合→SE加权
        x_left_ori,x_right_ori = x[:, :3, :, :], x[:, 3:, :, :]
        x_left = x_left_ori * 0.5
        x_right = x_right_ori * 0.5

        x_mask_left = torch.mul(self.mask_map_r(x_left), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # RGB

        out = self.se(torch.cat([out_RGB, out_IR], 1))

        return out
    
class IN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Add(nn.Module):
        def __init__(self, arg):
            super().__init__()
            self.arg = arg  # 保留原始参数

        def forward(self, x):
            # x是包含两个待相加张量的列表
            assert len(x) == 2, "输入必须包含两个待相加的张量"
            tensor_a, tensor_b = x[0], x[1]

            if tensor_a.shape[2:] != tensor_b.shape[2:]:
                # 统一调整为较大的尺寸
                target_size = tensor_a.shape[2:] if tensor_a.shape[2] >= tensor_b.shape[2] else tensor_b.shape[2:]
                tensor_a = F.interpolate(tensor_a, size=target_size, mode='bilinear', align_corners=False)
                tensor_b = F.interpolate(tensor_b, size=target_size, mode='bilinear', align_corners=False)

            # 执行加法操作
            return torch.add(tensor_a, tensor_b)


class FeatureAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5

    def forward(self, x):
        a, b = x
        w = torch.sigmoid(self.w)  # (0,1)
        # 要求 a,b 的 shape 完全一致
        return w * a + (1.0 - w) * b


class Multiin(nn.Module):  # stereo attention block
    def __init__(self, out=1, order="vi_ir"):
        super().__init__()
        self.out = out
        self.order = normalize_channel_order(order)

    def forward(self, x):
        if self.order == "vi_ir":
            x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        else:
            x2, x1 = x[:, :3, :, :], x[:, 3:, :, :]
        if self.out == 1:
            x = x1               #输出rgb特征
        else:
            x = x2               #输出ir特征
        return x
