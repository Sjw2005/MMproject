# IR-VI 多模态 YOLO 检测项目 mAP 提升方案（分层次分析）

> 基于 ultralytics-pro 改进方案库与 MMproject 现状分析
>
> 目标：提升多模态（红外+可见光）目标检测的 mAP50 精度
>
> 分析日期：2026-03-27

---

## 📋 执行摘要

根据对 `ir_vi_yolo_project_analysis.md` 的分析和 ultralytics-pro 项目中 400+ 改进方案的梳理，本文档将改进方案按**提升梯度**分为 5 个层级：

- **P0 级（必做）**：数据与工程基础修复，预期 mAP50 提升 5-15%
- **P1 级（高收益）**：小目标优化与训练策略，预期 mAP50 提升 3-10%
- **P2 级（中收益）**：损失函数与标签分配优化，预期 mAP50 提升 2-5%
- **P3 级（探索性）**：Neck 与注意力机制改进，预期 mAP50 提升 1-4%
- **P4 级（锦上添花）**：Backbone 替换与高级模块，预期 mAP50 提升 0-3%

**核心建议**：优先完成 P0 和 P1 级改进，这两级的投入产出比最高，能解决当前项目的根本问题。

---

## 🚨 P0 级：数据与工程基础修复（必做，预期提升 5-15%）

### 为什么这是最高优先级？

根据项目分析，当前最大的风险不是模型不够先进，而是**数据配对、模态顺序、训练验证流程存在潜在错误**。如果这些基础问题没解决，后续任何改进都是空中楼阁。

---

### P0.1 模态通道顺序一致性检查 ⚠️

**问题描述**：
- `ultralytics/data/base.py` 中 6 通道拼接顺序可能是 `[IR, VI]`
- 但 `Multiin`、`DMFNet.yaml`、`RandomHSV` 假设是 `[VI, IR]`
- 如果顺序错误，visible 分支实际吃到 IR，infrared 分支吃到 VI

**检查方法**：
```python
# 在训练脚本中添加可视化代码
import cv2
import numpy as np

def verify_channel_order(batch_data):
    """验证 6 通道数据的模态顺序"""
    img_6ch = batch_data[0].cpu().numpy()  # [6, H, W]

    # 分离前 3 通道和后 3 通道
    ch_0_2 = img_6ch[0:3].transpose(1, 2, 0)  # [H, W, 3]
    ch_3_5 = img_6ch[3:6].transpose(1, 2, 0)  # [H, W, 3]

    # 保存查看
    cv2.imwrite('check_ch_0_2.jpg', (ch_0_2 * 255).astype(np.uint8))
    cv2.imwrite('check_ch_3_5.jpg', (ch_3_5 * 255).astype(np.uint8))

    # 人工判断哪个是彩色（VI），哪个是灰度（IR）
```

**修复方案**：
1. 统一确认 6 通道顺序为 `[VI_R, VI_G, VI_B, IR, IR, IR]` 或 `[IR, IR, IR, VI_R, VI_G, VI_B]`
2. 修改 `Multiin` 模块的 `out=1/out=2` 对应关系
3. 确保 `RandomHSV` 只作用于 VI 的 3 个通道

**预期提升**：如果当前顺序错误，修复后 mAP50 可能直接提升 **10-15%**

---

### P0.2 IR/VI 文件配对验证

**问题描述**：
- 当前通过字符串替换 `images` → `image` 来找 IR 图
- 强依赖目录命名，容易失配
- 没有配对正确性校验

**验证脚本**：
```python
import os
from pathlib import Path

def verify_ir_vi_pairing(data_root):
    """验证 IR 和 VI 图像是否严格一一对应"""
    vi_dir = Path(data_root) / 'images'
    ir_dir = Path(data_root) / 'image'  # 根据实际路径调整

    vi_files = sorted([f.name for f in vi_dir.glob('*.jpg')])
    ir_files = sorted([f.name for f in ir_dir.glob('*.jpg')])

    # 检查文件名是否一致
    if set(vi_files) != set(ir_files):
        missing_in_ir = set(vi_files) - set(ir_files)
        missing_in_vi = set(ir_files) - set(vi_files)
        print(f"❌ 配对不一致！")
        print(f"VI 有但 IR 缺失: {missing_in_ir}")
        print(f"IR 有但 VI 缺失: {missing_in_vi}")
        return False

    # 检查图像尺寸是否一致
    for fname in vi_files[:100]:  # 抽样检查
        vi_img = cv2.imread(str(vi_dir / fname))
        ir_img = cv2.imread(str(ir_dir / fname))
        if vi_img.shape[:2] != ir_img.shape[:2]:
            print(f"❌ {fname} 尺寸不一致: VI={vi_img.shape}, IR={ir_img.shape}")
            return False

    print(f"✅ 配对检查通过！共 {len(vi_files)} 对图像")
    return True
```

**预期提升**：如果存在配对错误，修复后 mAP50 可能提升 **5-10%**

---

### P0.3 训练/验证/预测脚本统一

**问题描述**：
- `train.py` 训练 `DMFNet.yaml`
- `val.py` 验证 `./results/try/weights/best.pt`
- `predict.py` 使用 `RTDETR` 和不同路径
- 容易验证错模型、错权重

**修复方案**：
创建统一的实验配置管理：

```python
# config.py
class ExperimentConfig:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.model_yaml = './improve_multimodal/DMFNet.yaml'
        self.data_yaml = './data.yaml'
        self.save_dir = f'./runs/{exp_name}'
        self.weights = f'{self.save_dir}/weights/best.pt'

    def get_train_args(self):
        return {
            'data': self.data_yaml,
            'epochs': 150,
            'imgsz': 640,
            'batch': 16,
            'project': './runs',
            'name': self.exp_name,
        }
```

**预期提升**：避免实验混乱，确保结果可复现

---

### P0.4 建立标准 Baseline 对照组

在做任何改进之前，必须先有可靠的对照基准，否则无法判断改进是否有效。

| Baseline | 输入 | 模型 | 目的 |
|----------|------|------|------|
| A: VI-only | 3ch VI | 标准 YOLO11s | VI 单模态上限 |
| B: IR-only | 3ch IR | 标准 YOLO11s | IR 单模态上限 |
| C: 简单 Early Fusion | 6ch concat | YOLO11s 改首层 | 最简融合基线 |
| D: 当前 DMFNet | 6ch | DMFNet.yaml | 当前方案水平 |

**关键判断逻辑**：
- 如果 D < C：说明当前融合模块设计有问题，先做减法
- 如果 D ≈ C：说明融合模块没带来净收益，需要重新设计
- 如果 D > C > A/B：说明融合有效，可以继续在 D 基础上优化
- 如果 A > D：说明 IR 引入了噪声，融合策略需要根本性调整

**预期提升**：建立科学实验框架，避免盲目改进

---

## ⬆️ P1 级：小目标优化与训练策略（高收益，预期提升 3-10%）

### 为什么 P1 级收益高？

当前项目面向 DroneVehicle 无人机车辆数据集，目标普遍偏小。同时模型大概率从头训练、没有预训练初始化，这两个问题是 mAP 上不去的核心瓶颈。

---

### P1.1 预训练权重初始化（最高优先级训练策略）

**问题描述**：
当前 `train.py` 直接加载 `DMFNet.yaml` 从头训练，没有使用预训练权重。双分支 + 多融合模块的复杂结构从零学习，优化难度极大。

**改进方案（分阶段）**：

```python
# 阶段 1：先训练 VI-only baseline，获得稳定的单模态权重
model = YOLO('yolo11s.yaml')
model.train(data='data_vi_only.yaml', epochs=200, imgsz=640, pretrained=True)
# 此时 backbone 已经学会了基础的纹理、边缘、形状特征

# 阶段 2：将 VI backbone 权重映射到双分支模型
# 手动加载权重，将 VI backbone 的参数复制到 IR backbone（作为初始化）
import torch
vi_ckpt = torch.load('runs/vi_only/weights/best.pt')
dmfnet_model = YOLO('improve_multimodal/DMFNet.yaml')

# 映射 VI backbone 权重到双分支
state_dict = dmfnet_model.model.state_dict()
for key in vi_ckpt['model'].state_dict():
    if 'backbone_vi' in state_dict and key in state_dict:
        state_dict[key] = vi_ckpt['model'].state_dict()[key]
    # IR 分支也用 VI 权重初始化（迁移学习）
    ir_key = key.replace('backbone_vi', 'backbone_ir')
    if ir_key in state_dict:
        state_dict[ir_key] = vi_ckpt['model'].state_dict()[key]

# 阶段 3：冻结 backbone，只训练融合层和检测头
model.train(data='data.yaml', epochs=50, freeze=[0,1,2,3,4,5,6,7,8,9,10])

# 阶段 4：解冻全网微调
model.train(data='data.yaml', epochs=150, lr0=0.001)
```

**ultralytics-pro 中可参考**：
- `yolo11-aux.yaml`：辅助训练头思路，可在阶段 3 使用辅助头加速融合层收敛

**预期提升**：mAP50 提升 **5-8%**，这是从头训 vs 预训练初始化的典型差距

---

### P1.2 提高输入分辨率

**问题描述**：
当前 `imgsz=640`，对于无人机视角的小车辆目标，分辨率偏低。很多小目标在 640 下只有几个像素，特征提取困难。

**改进方案**：

| 实验 | imgsz | batch | 显存估算(单卡) | 说明 |
|------|-------|-------|---------------|------|
| 基线 | 640 | 16 | ~8GB | 当前设置 |
| 实验1 | 960 | 8 | ~12GB | 推荐首选 |
| 实验2 | 1280 | 4 | ~18GB | 显存充足时尝试 |
| 实验3 | 640→960 渐进 | 16→8 | ~12GB | 先 640 训 100ep，再 960 微调 50ep |

```python
# 推荐：渐进式分辨率训练
# 第一阶段：640 快速收敛
model.train(data='data.yaml', imgsz=640, epochs=100, batch=16)

# 第二阶段：960 精细化
model.train(data='data.yaml', imgsz=960, epochs=80, batch=8,
            resume=True, lr0=0.001)
```

**预期提升**：mAP50 提升 **2-5%**，小目标召回率提升尤为明显

---

### P1.3 增加 P2 小目标检测头

**问题描述**：
当前 DMFNet 只有 P3/P4/P5 三个检测头，最小特征图步长为 8。对于无人机视角下的小车辆，P2（步长 4）检测头能显著提升小目标检测能力。

**ultralytics-pro 中可直接参考的方案**：

| 方案 | 配置文件 | 特点 | 推荐度 |
|------|---------|------|--------|
| 标准 P2 头 | `yolo11-p2.yaml` | 最简单，增加 P2 层级的上采样和检测头 | ⭐⭐⭐⭐⭐ |
| AFPN-P2345 | `yolo11-AFPN-P2345.yaml` | 渐近式特征金字塔，P2-P5 四层检测 | ⭐⭐⭐⭐ |
| AFPN-P2345-Custom | `yolo11-AFPN-P2345-Custom.yaml` | 可自定义 block 的 AFPN | ⭐⭐⭐ |
| BiFPN + P2 | 基于 `yolo11-bifpn.yaml` 扩展 | BiFPN 加权融合 + P2 | ⭐⭐⭐⭐ |

**适配到 MMproject 的思路**：
```yaml
# 在 DMFNet.yaml 的 head 部分增加 P2 层
# 原来：融合后输出 P3, P4, P5 → Detect
# 改为：融合后输出 P2, P3, P4, P5 → Detect

# 关键修改点：
# 1. backbone 中 P2 特征（layer 2 输出）需要引出
# 2. 两个分支的 P2 特征也需要做融合
# 3. head 中增加 P2 的上采样路径
# 4. Detect 改为 4 输入
```

**注意事项**：
- P2 检测头会增加约 30-50% 的计算量和显存占用
- 建议配合降低 batch size 或使用梯度累积
- 如果显存紧张，可以只在 P2 用轻量化的 C3k2-Faster 模块

**预期提升**：mAP50 提升 **2-4%**，主要体现在小目标类别上

---

### P1.4 优化器与学习率策略

**问题描述**：
当前使用 SGD 优化器，对于自定义融合网络，AdamW 通常更友好。

**推荐实验矩阵**：

| 实验 | 优化器 | lr0 | cos_lr | epochs | 说明 |
|------|--------|-----|--------|--------|------|
| 基线 | SGD | 0.01 | False | 150 | 当前设置 |
| 实验1 | AdamW | 0.001 | True | 200 | 推荐首选 |
| 实验2 | SGD | 0.01 | True | 200 | 余弦退火 |
| 实验3 | AdamW | 0.0005 | True | 300 | 低学习率长训练 |

```python
# 推荐配置
model.train(
    data='data.yaml',
    epochs=200,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,        # 最终学习率 = lr0 * lrf
    cos_lr=True,      # 余弦退火
    warmup_epochs=5,
    weight_decay=0.05,
)
```

**预期提升**：mAP50 提升 **1-3%**

---

### P1.5 数据增强策略优化（多模态专用）

**问题描述**：
多模态数据增强必须保证"模态一致 + 物理合理"。IR 不能做颜色抖动，几何变换必须对 VI 和 IR 同步施加。

**推荐增强策略**：

```python
# 训练参数中的增强设置
model.train(
    # 几何增强（VI 和 IR 同步）
    mosaic=1.0,          # Mosaic 增强
    close_mosaic=20,     # 最后 20 epoch 关闭 mosaic
    mixup=0.1,           # MixUp（需确保 IR/VI 同步混合）
    degrees=5.0,         # 旋转角度
    translate=0.1,       # 平移
    scale=0.5,           # 缩放
    fliplr=0.5,          # 水平翻转
    flipud=0.1,          # 垂直翻转（无人机视角适用）

    # 颜色增强（仅作用于 VI 通道）
    hsv_h=0.015,         # 色调（仅 VI）
    hsv_s=0.5,           # 饱和度（仅 VI）
    hsv_v=0.3,           # 亮度（仅 VI）
    # IR 通道：仅做轻微亮度扰动 ±10%
)
```

**关键检查点**：
- 确认 `augment.py` 中 `RandomHSV` 只作用于前 3 通道（VI）
- Mosaic 拼接时 IR 和 VI 必须使用相同的拼接参数
- MixUp 混合时两对 IR-VI 图像必须同步混合

**预期提升**：mAP50 提升 **1-2%**

---

## 🎯 P2 级：损失函数与标签分配优化（中收益，预期提升 2-5%）

### 为什么 P2 级值得尝试？

损失函数和标签分配策略直接影响模型的优化方向。对于小目标密集、多模态融合的场景，标准的 CIoU + BCE 损失可能不够精细。

---

### P2.1 定位损失函数改进

**ultralytics-pro 支持的定位损失**（在 `LOSS改进系列.md` 中）：

| 损失类型 | 特点 | 适用场景 | 推荐度 |
|---------|------|---------|--------|
| **MPDIoU** | 考虑中心点距离和宽高比 | 小目标、长宽比变化大 | ⭐⭐⭐⭐⭐ |
| **ShapeIoU** | 关注形状相似度 | 车辆等规则形状目标 | ⭐⭐⭐⭐ |
| **Inner-IoU 系列** | 内部辅助框，提升小目标敏感度 | 小目标检测 | ⭐⭐⭐⭐⭐ |
| **Wise-IoU v3** | 动态调整难易样本权重 | 样本不均衡 | ⭐⭐⭐⭐ |
| **NWD** | 归一化高斯 Wasserstein 距离 | 小目标、模糊边界 | ⭐⭐⭐⭐ |

**推荐组合（针对无人机小车辆）**：
```python
# 在 ultralytics/utils/loss.py 的 BboxLoss 类中修改
class BboxLoss(nn.Module):
    def __init__(self, reg_max):
        super().__init__()
        self.reg_max = reg_max

        # 方案 1：Inner-MPDIoU（推荐）
        self.iou_type = 'inner_mpdiou'

        # 方案 2：NWD + CIoU 混合
        self.nwd_loss = True
        self.iou_ratio = 0.7  # CIoU 占 70%，NWD 占 30%
```

**预期提升**：mAP50 提升 **1-3%**，小目标 AP 提升更明显

---

### P2.2 分类损失函数改进

**ultralytics-pro 支持的分类损失**：

| 损失类型 | 特点 | 推荐度 |
|---------|------|--------|
| **VarifocalLoss** | 关注高质量正样本 | ⭐⭐⭐⭐⭐ |
| **QualityFocalLoss** | 联合优化分类和定位质量 | ⭐⭐⭐⭐ |
| **SlideLoss / EMASlideLoss** | 动态调节正负样本系数 | ⭐⭐⭐⭐ |

```python
# 在 ultralytics/utils/loss.py 的 v8DetectionLoss 中
class v8DetectionLoss:
    def __init__(self, model):
        # 使用 VarifocalLoss 替代标准 BCE
        self.use_vfl = True  # Varifocal Loss
        # 或使用 EMASlideLoss
        self.use_ema_slide = True
```

**预期提升**：mAP50 提升 **0.5-1.5%**

---

### P2.3 TAL 标签分配策略调优

**问题描述**：
当前 TAL (Task-Aligned Learning) 的 `topk` 和 `beta` 参数可能不适合小目标场景。

**推荐实验矩阵**：

| 实验 | tal_topk | beta | 说明 |
|------|----------|------|------|
| 基线 | 10 | 6.0 | YOLO11 默认 |
| 实验1 | 13 | 6.0 | 增加候选框 |
| 实验2 | 15 | 5.0 | 更多候选 + 降低惩罚 |
| 实验3 | 10 | 4.0 | 对小目标更宽容 |

```python
# 在 ultralytics/utils/loss.py 中
self.assigner = TaskAlignedAssigner(
    topk=13,      # 从 10 改为 13
    num_classes=self.nc,
    alpha=0.5,
    beta=5.0      # 从 6.0 改为 5.0
)
```

**预期提升**：mAP50 提升 **0.5-2%**

---

## 🔧 P3 级：Neck 与注意力机制改进（探索性，预期提升 1-4%）

### 为什么 P3 级是探索性？

Neck 和注意力改进能带来增益，但需要仔细调试。对于多模态项目，过度融合可能引入噪声。建议在 P0-P2 完成后，根据 baseline 对照结果决定是否尝试。

---

### P3.1 改进 Neck 结构

**ultralytics-pro 中适合多模态的 Neck 方案**：

| Neck 类型 | 配置文件 | 特点 | 多模态适配难度 | 推荐度 |
|----------|---------|------|---------------|--------|
| **BiFPN** | `yolo11-bifpn.yaml` | 加权双向特征融合 | 低 | ⭐⭐⭐⭐⭐ |
| **AFPN** | `yolo11-AFPN-P345.yaml` | 渐近式特征金字塔 | 低 | ⭐⭐⭐⭐ |
| **HSFPN** | `yolo11-HSFPN.yaml` | 高分辨率语义 FPN | 中 | ⭐⭐⭐ |
| **GoldYOLO** | `yolo11-goldyolo.yaml` | 信息聚合增强 | 中 | ⭐⭐⭐⭐ |

**推荐方案：BiFPN 适配到 DMFNet**

BiFPN 的加权融合机制天然适合多模态场景，可以学习 VI 和 IR 在不同尺度上的贡献权重。

```yaml
# 在 DMFNet 的融合后 Neck 部分使用 BiFPN
# 关键思路：
# 1. VI 和 IR 分支各自提取 P3/P4/P5
# 2. 先做模态内的 BiFPN（VI 内部、IR 内部）
# 3. 再做跨模态融合
# 4. 最后统一 BiFPN 输出到检测头
```

**预期提升**：mAP50 提升 **1-3%**

---

### P3.2 检测头改进

**ultralytics-pro 中的高级检测头**：

| 检测头 | 配置文件 | 特点 | 推荐度 |
|--------|---------|------|--------|
| **DyHead** | `yolo11-dyhead.yaml` | 动态注意力检测头 | ⭐⭐⭐⭐ |
| **TADDH** | `yolo11-TADDH.yaml` | 任务对齐解耦检测头 | ⭐⭐⭐⭐⭐ |
| **EfficientHead** | `yolo11-EfficientHead.yaml` | 轻量化检测头 | ⭐⭐⭐ |

**推荐：TADDH（Task-Aligned Decoupled Detection Head）**

TADDH 将分类和回归任务解耦，并引入任务对齐机制，特别适合小目标检测。

```yaml
# 在 DMFNet.yaml 的 head 最后一层
head:
  # ... 前面的融合层
  - [[P3_fused, P4_fused, P5_fused], 1, Detect_TADDH, [nc, 512]]
```

**预期提升**：mAP50 提升 **0.5-2%**

---

## 🔧 P3 级：Neck 与注意力机制改进（探索性，预期提升 1-4%）

### 为什么 P3 级是探索性？

Neck 和注意力改进能带来增益，但需要仔细调试。对于多模态项目，过度融合可能引入噪声。建议在 P0-P2 完成后，根据 baseline 对照结果决定是否尝试。

---

### P3.1 改进 Neck 结构

**ultralytics-pro 中适合多模态的 Neck 方案**：

| Neck 类型 | 配置文件 | 特点 | 多模态适配难度 | 推荐度 |
|----------|---------|------|---------------|--------|
| **BiFPN** | `yolo11-bifpn.yaml` | 加权双向特征融合 | 低 | ⭐⭐⭐⭐⭐ |
| **AFPN** | `yolo11-AFPN-P345.yaml` | 渐近式特征金字塔 | 低 | ⭐⭐⭐⭐ |
| **HSFPN** | `yolo11-HSFPN.yaml` | 高分辨率语义 FPN | 中 | ⭐⭐⭐ |
| **GoldYOLO** | `yolo11-goldyolo.yaml` | 信息聚合增强 | 中 | ⭐⭐⭐⭐ |

**推荐方案：BiFPN 适配到 DMFNet**

BiFPN 的加权融合机制天然适合多模态场景，可以学习 VI 和 IR 在不同尺度上的贡献权重。

```yaml
# 在 DMFNet 的融合后 Neck 部分使用 BiFPN
# 关键思路：
# 1. VI 和 IR 分支各自提取 P3/P4/P5
# 2. 先做模态内的 BiFPN（VI 内部、IR 内部）
# 3. 再做跨模态融合
# 4. 最后统一 BiFPN 输出到检测头
```

**预期提升**：mAP50 提升 **1-3%**

---

### P3.2 检测头改进

**ultralytics-pro 中的高级检测头**：

| 检测头 | 配置文件 | 特点 | 推荐度 |
|--------|---------|------|--------|
| **DyHead** | `yolo11-dyhead.yaml` | 动态注意力检测头 | ⭐⭐⭐⭐ |
| **TADDH** | `yolo11-TADDH.yaml` | 任务对齐解耦检测头 | ⭐⭐⭐⭐⭐ |
| **EfficientHead** | `yolo11-EfficientHead.yaml` | 轻量化检测头 | ⭐⭐⭐ |

**推荐：TADDH（Task-Aligned Decoupled Detection Head）**

TADDH 将分类和回归任务解耦，并引入任务对齐机制，特别适合小目标检测。

```yaml
# 在 DMFNet.yaml 的 head 最后一层
head:
  # ... 前面的融合层
  - [[P3_fused, P4_fused, P5_fused], 1, Detect_TADDH, [nc, 512]]
```

**预期提升**：mAP50 提升 **0.5-2%**

---


### P3.3 注意力机制（谨慎使用）

**问题**：当前 DMFNet 已有 LAEF 和 MDAFP 融合模块，再加注意力可能过度复杂。

**建议策略**：
1. 先做 P0-P2 的改进
2. 如果 baseline 对照显示融合模块有效，再考虑轻量注意力
3. 优先在 backbone 浅层（P2/P3）加注意力，不在深层（P5）

**ultralytics-pro 中适合小目标的注意力**：

| 注意力 | 特点 | 计算开销 | 推荐度 |
|--------|------|---------|--------|
| **SimAM** | 无参数注意力 | 极低 | ⭐⭐⭐⭐⭐ |
| **EMA** | 高效多尺度注意力 | 低 | ⭐⭐⭐⭐ |
| **LSKBlock** | 大核选择注意力 | 中 | ⭐⭐⭐⭐ |
| **MLCA** | 多层级上下文注意力 | 中 | ⭐⭐⭐ |

**使用方式**：
```yaml
# 在 C3k2 模块中插入注意力
# 例如：C3k2-EMA 替换部分 C3k2
backbone:
  - [-1, 2, C3k2, [256, False, 0.25]]  # 原来
  - [-1, 2, C3k2_EMA, [256, False, 0.25]]  # 改进
```

**预期提升**：mAP50 提升 **0.5-1.5%**（但可能增加训练不稳定性）

---

## 🚀 P4 级：Backbone 替换与高级模块（锦上添花，预期提升 0-3%）

### 为什么 P4 级优先级最低？

Backbone 替换通常需要大量调试，且对多模态双分支结构适配困难。只有在 P0-P3 都完成、精度仍不达标时才考虑。

---

### P4.1 轻量化 Backbone（如果推理速度重要）

| Backbone | 配置文件 | 特点 | 推荐度 |
|----------|---------|------|--------|
| **FastNet** | `yolo11-fasternet.yaml` | CVPR2023，速度快 | ⭐⭐⭐⭐ |
| **EfficientViT** | `yolo11-efficientViT.yaml` | 高效 ViT | ⭐⭐⭐ |
| **MobileNetV4** | `yolo11-mobilenetv4.yaml` | 移动端优化 | ⭐⭐⭐ |

**注意**：双分支结构需要两个 backbone，轻量化收益会打折扣。

---

### P4.2 大感受野 Backbone（如果精度优先）

| Backbone | 配置文件 | 特点 | 推荐度 |
|----------|---------|------|--------|
| **UniRepLKNet** | `yolo11-unireplknet.yaml` | 超大卷积核 | ⭐⭐⭐⭐ |
| **LSKNet** | `yolo11-lsknet.yaml` | 大核选择网络 | ⭐⭐⭐⭐ |
| **ConvNeXt V2** | `yolo11-convnextv2.yaml` | 现代卷积网络 | ⭐⭐⭐ |

**预期提升**：mAP50 提升 **0-2%**，但训练时间显著增加

---

### P4.3 高级卷积模块（实验性）

**ultralytics-pro 中的高级卷积**：

| 模块 | 特点 | 推荐度 |
|------|------|--------|
| **C3k2-DySnakeConv** | 动态蛇形卷积，适合细长目标 | ⭐⭐⭐ |
| **C3k2-RFAConv** | 感受野注意力卷积 | ⭐⭐⭐⭐ |
| **C3k2-Faster** | FasterNet 块，速度快 | ⭐⭐⭐⭐ |
| **C3k2-DCNV3** | 可变形卷积 V3 | ⭐⭐⭐ |

**使用建议**：
- 只在 Neck 部分尝试，不要动 backbone
- 优先试 C3k2-Faster（轻量）或 C3k2-RFAConv（精度）

**预期提升**：mAP50 提升 **0.5-1.5%**

---


## 📊 综合实验路线图

### 推荐执行顺序（按阶段）

```
阶段一：排雷（1-2天）                    阶段二：建基线（3-5天）
┌─────────────────────┐                ┌─────────────────────┐
│ P0.1 通道顺序检查    │                │ P0.4 Baseline A:    │
│ P0.2 配对验证        │───────────────▶│      VI-only YOLO   │
│ P0.3 脚本统一        │                │      Baseline B:    │
│                     │                │      IR-only YOLO   │
│                     │                │      Baseline C:    │
│                     │                │      6ch Early Fuse │
│                     │                │      Baseline D:    │
│                     │                │      当前 DMFNet     │
└─────────────────────┘                └──────────┬──────────┘
                                                  │
                                                  ▼
阶段三：高收益改进（5-7天）              阶段四：精细调优（3-5天）
┌─────────────────────┐                ┌─────────────────────┐
│ P1.1 预训练初始化    │                │ P2.1 IoU Loss 改进   │
│ P1.2 提高分辨率      │───────────────▶│ P2.2 分类 Loss 改进  │
│ P1.3 P2 检测头       │                │ P2.3 TAL 调优        │
│ P1.4 AdamW+CosLR    │                │ P1.5 增强策略微调     │
│                     │                │                     │
└─────────────────────┘                └──────────┬──────────┘
                                                  │
                                                  ▼
阶段五：探索性改进（按需）
┌─────────────────────┐
│ P3.1 BiFPN Neck      │
│ P3.2 TADDH 检测头    │
│ P3.3 SimAM/EMA 注意力│
│ P4.x Backbone/卷积   │
└─────────────────────┘
```

---

### 每阶段实验记录模板

每次实验请记录以下信息，便于对比和复现：

| 字段 | 示例 |
|------|------|
| 实验编号 | exp_001 |
| 改进点 | P1.1 预训练初始化 |
| 模型配置 | DMFNet.yaml |
| imgsz | 640 |
| batch | 16 |
| epochs | 200 |
| 优化器 | AdamW |
| lr0 | 0.001 |
| 是否预训练 | Yes (VI backbone) |
| mAP50 | xx.x% |
| mAP50-95 | xx.x% |
| 小目标 AP | xx.x% |
| 每类 AP | car/truck/bus/... |
| 训练时间 | xx h |
| 备注 | ... |

---

### 预期累计提升估算

| 阶段 | 改进项 | 单项预期提升 | 累计预期 mAP50 |
|------|--------|-------------|---------------|
| P0 | 数据/工程修复 | +5~15% | 基线 + 5~15% |
| P1 | 预训练+分辨率+P2头 | +3~10% | 基线 + 10~20% |
| P2 | Loss+TAL 优化 | +2~5% | 基线 + 12~23% |
| P3 | Neck+检测头+注意力 | +1~4% | 基线 + 13~25% |
| P4 | Backbone+高级模块 | +0~3% | 基线 + 13~27% |

> 注意：以上为理论估算，实际提升取决于当前 baseline 水平和数据质量。各项改进之间可能存在重叠或冲突，不能简单累加。

---

## ⚡ 快速行动清单（Top 5 最值得先做的事）

1. **检查 IR/VI 通道顺序**：写一个可视化脚本，确认 6 通道的前 3 通道和后 3 通道分别是什么。如果顺序错了，修复后可能直接涨 10%+。

2. **跑 VI-only baseline**：用标准 YOLO11s + 预训练权重，只用 VI 图像训练。这个数字是你所有改进的参照锚点。

3. **给 DMFNet 加预训练初始化**：用 VI baseline 的 backbone 权重初始化双分支模型，分阶段训练。

4. **把 imgsz 从 640 提到 960**：对小目标检测效果立竿见影。

5. **加 P2 检测头**：参考 `yolo11-p2.yaml`，在 DMFNet 中增加 P2 层级输出。

---

## 📎 ultralytics-pro 改进方案速查索引

### 按改进类型分类

| 类型 | 数量 | 代表方案 | 对应文件位置 |
|------|------|---------|-------------|
| C3k2 变体 | 100+ | C3k2-Faster, C3k2-EMA, C3k2-DCNV2/3/4 | `cfg/models/11/yolo11-C3k2-*.yaml` |
| Neck 改进 | 20+ | BiFPN, AFPN, HSFPN, GoldYOLO, SlimNeck | `cfg/models/11/yolo11-bifpn.yaml` 等 |
| 检测头 | 10+ | DyHead, TADDH, LSCD, EfficientHead | `cfg/models/11/yolo11-dyhead.yaml` 等 |
| Backbone | 15+ | FastNet, EfficientViT, LSKNet, ConvNeXtV2 | `cfg/models/11/yolo11-fasternet.yaml` 等 |
| 注意力 | 25+ | EMA, SimAM, LSKBlock, MLCA, CoordAtt | `nn/extra_modules/attention.py` |
| Loss | 30+ | IoU系列, NWD, SlideLoss, VFL | `LOSS改进系列.md` |
| 上/下采样 | 5+ | DySample, CARAFE, SPDConv, LAWDS | `cfg/models/11/yolo11-dysample.yaml` 等 |

### 按多模态适配难度分类

| 难度 | 改进类型 | 说明 |
|------|---------|------|
| **低（直接用）** | Loss 改进、TAL 调优、数据增强 | 不涉及网络结构，直接修改 loss.py/tal.py |
| **低（直接用）** | 检测头替换 | 只改最后一层，不影响双分支结构 |
| **中（需适配）** | Neck 改进、注意力机制 | 需要在融合后的特征上使用 |
| **高（需重构）** | Backbone 替换 | 双分支都要换，预训练权重需重新映射 |

---

> 本文档基于 ultralytics-pro 项目（400+ 改进方案）和 MMproject 现状分析生成。
> 核心原则：**先排雷 → 建基线 → 高收益改进 → 精细调优 → 探索性改进**。
> 不要跳过 P0 直接做 P3/P4，那是最常见的"改了很多但精度不涨"的原因。
