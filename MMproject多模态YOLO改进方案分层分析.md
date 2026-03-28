# MMproject 多模态 YOLO 改进方案分层分析

## 1. 目标与结论先行

这份文档的目标不是把 `ultralytics-pro` 里的几百个 YOLO 改法逐个罗列一遍，而是结合 `MMproject` 当前的 IR-VI 双模态检测结构，按照 **对 mAP 提升的概率、改动成本、训练风险** 做分层筛选，找出最值得优先落地的路线。

一句话结论：

- 对 `MMproject` 来说，**最先拉 mAP 的不是继续堆新模块**，而是先把实验闭环、预训练初始化、小目标能力和 neck/head 的低风险增强做好。
- `ultralytics-pro` 里真正和当前项目最匹配的改法，优先级最高的是：**P2 小目标头、DySample/CARAFE、BiFPN/AFPN、LQEHead、NWD/IoU 类 loss、TAL 小范围调参**。
- 多 backbone / 多分支融合类方案（如 `MutilBackbone-DAF/HAFB/MSGA`、`CAFMFusion`、`CGAFusion`）对多模态项目很有启发，但它们属于 **第二阶段结构升级项**，不建议在 baseline 还不稳时就直接大改。

---

## 2. 当前 MMproject 的状态复核

结合 `ir_vi_yolo_project_analysis.md` 和当前代码快照，`MMproject` 当前已经具备以下基础：

- `improve_multimodal/DMFNet.yaml`：6 通道输入、双分支 backbone、`LAEF` 对齐、`MDAFP` 多尺度融合、P3/P4/P5 检测头。
- `ultralytics/data/base.py`：按可见光路径读取配对红外图，并拼成 6 通道输入。
- `ultralytics/data/augment.py`：6 通道时区分 RGB 与 IR，RGB 做 HSV，IR 做轻量强度扰动。
- `ultralytics/utils/loss.py`：当前是 `TaskAlignedAssigner(topk=15, alpha=0.5, beta=6.0)` + BCE + CIoU + DFL。
- `train.py`：当前训练配置为 `imgsz=640`、`SGD`、`amp=False`、未显式加载预训练权重。

### 2.1 和旧分析相比，需要修正的一点

`ir_vi_yolo_project_analysis.md` 里最担心的是“通道顺序可能错位”。

但从当前代码快照看，链路已经基本对齐：

- `ultralytics/data/base.py` 当前是先读可见光 `im`，再拼接红外 `ir_im`，即 `[VI, IR]`。
- `ultralytics/nn/AddModules/multimodal.py` 中 `Multiin` 把前 3 通道作为 visible，后 3 通道作为 infrared。
- `ultralytics/data/augment.py` 中 `RandomHSV(order="rgb_ir")` 也默认前 3 通道是 RGB、后 3 通道是 IR。

也就是说，**当前版本看起来已经不是“明显顺序错位”状态**。但这不代表这个问题可以跳过，仍然建议做 50~100 对样本的可视化抽检，因为多模态项目里“样本配对正确”本身就是精度的一部分。

### 2.2 当前仍然最影响 mAP 的问题

目前最值得优先解决的仍然是这 5 个问题：

1. `train.py`、`val.py`、`predict.py` 的模型链路不统一：训练是 `DMFNet`，验证路径却指向 `./results/try/weights/best.pt`，预测又切成了 `RTDETR`。
2. 当前主模型只有 `P3/P4/P5`，对 DroneVehicle 这类无人机小目标场景不够友好。
3. 当前看起来仍是自定义双分支结构直接训练，缺少明确的预训练迁移与阶段训练策略。
4. 训练配置固定在 `640` 分辨率，容易先天吃亏。
5. 当前 loss 和样本分配超参是固定值，没有系统扫参。

---

## 3. `ultralytics-pro` 改进方案，如何映射到 MMproject

`ultralytics-pro` 的 YOLO 改法非常多，但对 `MMproject` 不需要“全都试”，而要按适配度筛选。

| 改进族 | 代表方案 | 对 MMproject 的价值 | 适配难度 | 建议时机 |
|---|---|---:|---:|---|
| 小目标检测层 | `yolo11-p2.yaml`、`yolo11-ASF-P2.yaml`、`AFPN-P2345`、`ReCalibrationFPN-P2345`、`SOEP` | 很高 | 中 | 第一阶段 |
| Neck 多尺度融合 | `yolo11-bifpn.yaml`、`AFPN-P2345`、`HSFPN`、`GDFPN`、`MAFPN`、`EMBSFPN` | 高 | 中 | 第一/第二阶段 |
| 上采样增强 | `yolo11-dysample.yaml`、`yolo11-CARAFE.yaml`、`yolo11-EUCB.yaml` | 高 | 低~中 | 第一阶段 |
| 检测头增强 | `yolo11-LQEHead.yaml`、`yolo11-dyhead.yaml`、`yolo11-atthead.yaml`、`SEAMHead`、`MultiSEAMHead`、`aux` | 中高 | 中 | 第二阶段 |
| Loss / 标签分配 | `NWD`、`MPDIoU`、`Wise-IoU`、`Focal/Varifocal`、`ATSS`、`APT-TAL` | 高 | 低~中 | 第一/第二阶段 |
| 多 backbone / 融合结构 | `MutilBackbone-DAF`、`MutilBackbone-HAFB`、`MutilBackbone-MSGA`、`CAFMFusion`、`CGAFusion` | 高潜力 | 高 | 第二/第三阶段 |
| Backbone 替换 | `HGNetV2`、`FasterNet`、`RepHGNetV2`、`LSKNet`、`RepViT` | 中 | 高 | 第三阶段 |
| 新版 C2PSA / C3k2 / Mamba / 编译类模块 | `SEFN`、`SEFFN`、`EDFFN`、`MALA`、`Mona`、`DCNV3/v4`、`Mamba*` | 有上限但风险高 | 很高 | 最后再试 |

核心判断是：

- **先动 neck / head / loss / P2，比先换整个 backbone 更稳。**
- **先做“小目标友好 + 训练稳定 + 定位质量增强”，比先堆 2025 新模块更容易涨 mAP。**
- 对多模态项目而言，`ultralytics-pro` 里最有价值的不是“花哨 backbone”，而是 **P2、小目标 neck、融合 neck、定位质量头、loss/assigner**。

---

## 4. 按“提升梯度”分层给出建议

下面按 **提升概率从高到低** 分层。

## Layer S：必须先做的基础增益层（提升概率最高）

这层不是“可选项”，而是所有后续改法的前提。

### S1. 统一实验闭环

- 统一 `train.py` / `val.py` / `predict.py`，全部指向同一模型体系与同一实验目录。
- 每次实验固定保存：`args.yaml`、`results.csv`、`best.pt`、每类 AP、混淆矩阵、失败样例。
- 没有这个闭环，后面所有改法都很难比较真假收益。

### S2. 建立 4 个基线

至少先做：

1. `VI-only YOLO11`
2. `IR-only YOLO11`
3. `6ch early fusion YOLO`
4. `当前 DMFNet`

没有这 4 组，后续所有多模态改进都没有参照物。

### S3. 预训练初始化 + 分阶段训练

- 先训好 `VI-only` 基线。
- 把可见光分支初始化到双分支模型。
- IR 分支可先复制初始化，再慢慢适配。
- 先冻结部分 backbone，只训融合层和检测头，再全网解冻微调。

这是当前项目最容易带来真实收益的一项，因为它直接改善“复杂双分支结构不好训”的问题。

### S4. 提高分辨率 + 引入 P2 检测层

当前 `DMFNet` 只有 `P3/P4/P5`，这对无人机车辆检测并不理想。

优先建议：

- 先做 `imgsz=960` 和 `imgsz=1280` 对照。
- 加 `P2` 检测头，优先参考：
  - `ultralytics-pro/ultralytics/cfg/models/11/yolo11-p2.yaml`
  - `ultralytics-pro/ultralytics/cfg/models/11/yolo11-ASF-P2.yaml`
  - `ultralytics-pro/ultralytics/cfg/models/11/yolo11-AFPN-P2345.yaml`

这一层通常比“继续换注意力模块”更容易涨 `mAP50`。

---

## Layer A：低风险高收益层（最适合第一批接入）

这一层是我最推荐你在 `DMFNet` 上优先嫁接的 `ultralytics-pro` 方案。

### A1. 上采样模块：优先 `DySample`，备选 `CARAFE`

对应方案：

- `ultralytics-pro/ultralytics/cfg/models/11/yolo11-dysample.yaml`
- `ultralytics-pro/ultralytics/cfg/models/11/yolo11-CARAFE.yaml`

推荐理由：

- 改动小，容易嵌入当前 head。
- 直接作用在 P5->P4、P4->P3 的特征恢复上。
- 对小目标和边界细节比最近邻上采样更友好。

优先级建议：`DySample > CARAFE`。

### A2. Neck 融合：优先 `BiFPN / AFPN-P2345`

对应方案：

- `ultralytics-pro/ultralytics/cfg/models/11/yolo11-bifpn.yaml`
- `ultralytics-pro/ultralytics/cfg/models/11/yolo11-AFPN-P2345.yaml`

推荐理由：

- 当前 `DMFNet` 已经解决了“跨模态融合”，但对“跨尺度融合”的强化还不够系统。
- `BiFPN/AFPN` 的价值是把已经融合好的多模态特征，在 P2/P3/P4/P5 间再做更高质量的多尺度传播。
- 对小目标召回和尺度鲁棒性更友好。

优先级建议：

- 想稳一点：先试 `BiFPN`
- 想冲小目标：直接试 `AFPN-P2345`

### A3. 检测头：优先 `LQEHead`，轻量对照用 `atthead`

对应方案：

- `ultralytics-pro/ultralytics/cfg/models/11/yolo11-LQEHead.yaml`
- `ultralytics-pro/ultralytics/cfg/models/11/yolo11-atthead.yaml`

推荐理由：

- `LQEHead` 更针对定位质量建模，往往比单纯再堆注意力更贴近 mAP 提升目标。
- `atthead` 改动最小，适合先做轻量对照。

条件使用：

- 如果遮挡很重，再考虑 `SEAMHead` / `MultiSEAMHead`。
- 如果 occlusion 不是主矛盾，不建议一开始先上 `SEAM`。

### A4. Loss / 标签分配：这是第一批必须系统扫的项目

优先顺序建议：

1. `NWD`：对小目标框回归很有帮助。
2. `Wise-IoU / MPDIoU`：针对回归质量做增强。
3. `FocalLoss / VarifocalLoss`：改善难样本分类。
4. `APT-TAL` 或 `ATSS`：改善正负样本分配。
5. `tal_topk / beta` 小范围搜索：`10/13/15` 与 `4/5/6`。

原因很简单：

- 当前 `loss.py` 里 `topk=15, beta=6.0` 是固定死的。
- 多模态 + 小目标数据上，分配器和 box loss 往往比再加一个 attention 更直接。

### A5. 训练策略：从“能训”转成“训得稳”

建议一起纳入第一批实验：

- `AdamW` 对照 `SGD`
- `cos_lr=True`
- `AMP=True`（除非某个特定模块明确要求关掉）
- 高分辨率下使用梯度累积
- `close_mosaic`、`mosaic`、`hsv`、IR 强度扰动做小范围搜索

注意：你当前 `amp=False` 适合保守排错，但如果要上 `960/1280`，后面最好尽量恢复 AMP。

---

## Layer B：中风险结构增强层（Baseline 稳住后再上）

这层的核心不是“继续堆模块”，而是 **把 `ultralytics-pro` 里对多分支/多特征融合有启发的结构，迁移到 MMproject 的双模态场景**。

### B1. 用 `DAF / HAFB / MSGA` 思路替换当前一部分跨模态融合节点

对应方案：

- `yolo11-MutilBackbone-DAF.yaml`
- `yolo11-MutilBackbone-HAFB.yaml`
- `yolo11-MutilBackbone-MSGA.yaml`

适配思路：

- 它们本质上都是“多 backbone / 多分支之间的对齐与融合”。
- 这和 `MMproject` 的 visible / infrared 双分支天然契合。
- 可以考虑先在 `P3/P4` 两个阶段做替换试验，不要一上来把 `LAEF + MDAFP` 全盘推倒。

### B2. 用 `CAFMFusion / CGAFusion` 替代后融合阶段的简单拼接

对应方案：

- `yolo11-CAFMFusion.yaml`
- `yolo11-CGAFusion.yaml`

适配建议：

- 不建议直接替掉最早期模态拆分。
- 更适合替换 `MDAFP` 之后进入 head 前的多尺度融合节点，或者做和 `MDAFP` 的消融对照。

### B3. 检测头进一步升级：`DyHead / Aux / TADDH` 这类再后置

这类 head 的收益可能不错，但它们对训练稳定性更敏感。

建议顺序：

1. `LQEHead`
2. `DyHead`
3. `Aux`
4. `TADDH / LSCD-LQE` 一类更复杂头

换句话说：**先把 P2、neck、loss 做好，再换大 head。**

### B4. 分支内部的 block 微替换，比整根 backbone 替换更适合当前阶段

可优先考虑给 visible / IR 两个分支内部的 `C3k2` 做轻量增强，而不是直接换完整 backbone。

相对更适合先试的 block：

- `C3k2-EMA`
- `C3k2-MSBlock`
- `C3k2-EMSC/EMSCP`
- `C3k2-SCConv/SCcConv`

这类替换的风险，明显低于直接双分支一起换 `HGNetV2`、`LSKNet`、`RepViT`。

---

## Layer C：高风险冲榜层（论文/冲榜阶段再做）

这一层不代表没价值，而是 **不适合第一批就做**。

### C1. 完整 backbone 替换

例如：

- `HGNetV2`
- `FasterNet`
- `RepHGNetV2`
- `LSKNet`
- `RepViT`
- `efficientViT`

问题在于：

- 你的 `MMproject` 是双分支结构，不是单分支标准 YOLO。
- 完整 backbone 替换意味着 visible / IR 两支都要同步适配。
- 同时你还要重新考虑预训练权重映射、分支共享策略、融合层接口。

所以它们更像“上限项”，不是“第一批提分项”。

### C2. 新版 C2PSA / C3k2 大量实验型模块

例如：

- `C2PSA-SEFN`
- `C2PSA-SEFFN`
- `C2PSA-EDFFN`
- `C2PSA-Mona`
- `C2PSA-MALA`
- 大量 2025 年的 `C3k2-*`

这类模块的共同问题是：

- 很新，稳定性和任务泛化不一定好。
- 对你的当前问题并不够“对症”，因为你最核心的问题仍是小目标、多模态训练稳定和工程闭环。

### C3. 编译依赖重、需要关闭 AMP 的模块

例如：

- `DCNV3/DCNV4`
- 一些 Mamba / Transformer / Triton 依赖模块
- 文档里明确提示要关 AMP 的模块

这类模块建议放到最后，因为它们会显著增加实验成本，还可能反过来压缩高分辨率训练空间。

---

## 5. 我认为最适合 MMproject 的一条推荐路线

如果目标是 **尽快把 mAP 提上去**，建议按下面顺序做：

### 第 0 步：先把基线和工程跑通

1. 统一 `train / val / predict`。
2. 做 `VI-only / IR-only / 6ch early fusion / DMFNet` 四组基线。
3. 按白天/夜间、小/中/大目标做分组评估。

### 第 1 步：先拿最稳的结构收益

推荐组合 A：

- `DMFNet + 预训练初始化 + imgsz=960/1280 + P2 head`

这是我认为 **性价比最高** 的第一组合。

### 第 2 步：补 neck 细节恢复

推荐组合 B：

- `组合 A + DySample`

如果 `DySample` 效果一般，再试：

- `组合 A + CARAFE`

### 第 3 步：加强多尺度融合

推荐组合 C：

- `组合 B + BiFPN`

如果你更想冲小目标，再试：

- `组合 B + AFPN-P2345`

### 第 4 步：提升定位质量

推荐组合 D：

- `组合 C + LQEHead`
- `组合 C + NWD`
- `组合 C + tal_topk/beta sweep`

### 第 5 步：再决定是否重做融合模块

只有当上面这些都做完后，再判断要不要进入：

- `DAF/HAFB/MSGA` 替代当前一部分融合节点
- `CAFMFusion/CGAFusion` 对 `MDAFP` 做替代或消融
- `DyHead / Aux / 更复杂 head`

---

## 6. 一份更具体的实验优先级清单

### 第一优先级（马上做）

1. 基线补齐
2. 预训练迁移
3. `imgsz=960/1280`
4. `P2` 检测头
5. `DySample`
6. `NWD + topk/beta sweep`

### 第二优先级（第一轮有效后接着做）

1. `BiFPN`
2. `AFPN-P2345`
3. `LQEHead`
4. `AdamW + cos_lr`
5. `Wise-IoU / MPDIoU / Varifocal`

### 第三优先级（有了稳定强 baseline 再做）

1. `CAFMFusion / CGAFusion`
2. `DAF / HAFB / MSGA`
3. `DyHead / Aux`
4. 分支内部 `C3k2` 微替换

### 第四优先级（冲榜 / 论文阶段）

1. 双分支 backbone 大替换
2. 大量新版 `C2PSA / C3k2` 模块
3. 编译依赖重、需要关 AMP 的模块

---

## 7. 不建议第一批就做的事

- 不建议一开始就把 `LAEF + MDAFP + neck + head + loss` 一起改，无法判断收益来源。
- 不建议先换完整 backbone，因为双分支迁移成本太高。
- 不建议先上 2025 年大量新模块，训练风险和排错成本都过高。
- 不建议在主训练还不稳时就把时间花在 `TTA / soft-NMS / 多模型融合` 上，这些属于后处理加分项。

---

## 8. 最后的建议

对 `MMproject` 来说，最合理的主线不是“继续找更花的融合模块”，而是：

1. **先把工程和基线做标准化**；
2. **先补小目标能力（高分辨率 + P2）**；
3. **先上低风险 neck/head/loss 改法（DySample、BiFPN、LQEHead、NWD）**；
4. **确认这些低风险项已经吃满，再回头重做模态融合结构**。

如果只让我给出一个最推荐的优先组合，那就是：

> **DMFNet + 预训练初始化 + 960/1280 分辨率 + P2 检测头 + DySample + BiFPN + LQEHead + NWD/TAL 小范围扫参**

这条路线，比直接去堆更多 attention / Mamba / 2025 新模块，更符合你这个项目当前的提分逻辑。
