# 多模态 YOLO 已改进记录与训练说明

## 1. 文档目的

这份文档用于记录当前 `MMproject` 已经完成的第一阶段改造，并说明现在如何开始训练、验证和预测。

当前这一轮改造的重点，不是继续堆新的检测模块，而是先解决：

- 模态通道顺序一致性
- IR/VI 文件配对一致性
- train / val / predict 脚本不统一
- baseline 对照组缺失

---

## 2. 当前已经完成的改进内容

## 阶段 1：数据一致性与模态顺序修正

### 2.1 新增多模态共享工具

新增文件：`ultralytics/data/multimodal.py`

已经实现：

- 统一 `channel_order` 解析
- 统一 VI/IR 配对路径解析
- 统一 6 通道拆分与融合
- 抽样检查 IR/VI 配对问题

这一步的意义是：

- 不再把 `images -> image` 这种逻辑散落在多个文件里硬编码
- 后续 train / val / predict / dataset 都用同一套规则

### 2.2 数据加载逻辑已统一到共享规则

已修改文件：`ultralytics/data/base.py`

已经完成：

- 支持 `visible`、`infrared`、`multimodal` 三种输入模式
- 读取 6 通道时不再使用脆弱的手写路径替换，而是走共享解析函数
- 初始化数据集时支持抽样检查 IR/VI 配对与尺寸一致性
- cache 文件名增加模态标记，避免 3 通道 / 6 通道缓存混淆

### 2.3 推理加载逻辑也已同步修正

已修改文件：`ultralytics/data/loaders.py`

已经完成：

- 推理时也按共享规则查找配对红外图
- 推理时的 6 通道拼接顺序和训练保持一致

### 2.4 模态通道顺序已显式固定

已修改文件：

- `data.yaml`
- `ultralytics/cfg/default.yaml`
- `improve_multimodal/DMFNet.yaml`

当前默认统一为：

- `channel_order: vi_ir`

即：

- 前 3 通道 = VI
- 后 3 通道 = IR

### 2.5 `Multiin` 已支持显式顺序参数

已修改文件：`ultralytics/nn/AddModules/multimodal.py`

已经完成：

- `Multiin` 不再默认假设输入永远固定不变
- 现在支持通过 `order` 参数显式指定 `vi_ir` 或 `ir_vi`

### 2.6 `RandomHSV` 已按通道顺序自动匹配 VI 通道

已修改文件：`ultralytics/data/augment.py`

已经完成：

- 当 `channel_order=vi_ir` 时，HSV 作用于前 3 通道
- 当 `channel_order=ir_vi` 时，HSV 自动切到后 3 通道

这一步的意义是：

- 避免把 IR 当成 RGB 去做颜色增强

---

## 阶段 2：训练 / 验证 / 预测脚本统一

### 2.7 新增统一实验配置中心

新增文件：`experiment_config.py`

已经完成：

- 统一管理实验名、模型 YAML、数据路径、输入模态、保存目录、训练超参
- 所有训练相关配置都集中到一个地方维护

当前已注册实验：

- `dmfnet`
- `dmfnet_imgsz960`
- `dmfnet_imgsz1280`
- `dmfnet_adamw`

### 2.8 `train.py` / `val.py` / `predict.py` 已统一到同一配置体系

已修改文件：

- `train.py`
- `val.py`
- `predict.py`

已经完成：

- 三个脚本不再各自写死不同模型逻辑
- 都通过 `experiment_config.py` 读取实验配置
- 默认使用同一个实验：`dmfnet`
- 现在已经支持通过命令行直接切换实验名

这一步的意义是：

- 避免训练一个模型、验证另一个权重、预测又换成第三套逻辑

---

## 阶段 3：Baseline 脚手架已补齐

### 2.9 新增 3 个 baseline 模型 YAML

新增文件：

- `improve_multimodal/yolo11-visible.yaml`
- `improve_multimodal/yolo11-ir.yaml`
- `improve_multimodal/yolo11-earlyfusion.yaml`

含义如下：

- `baseline_visible`：只使用可见光 VI
- `baseline_ir`：只使用红外 IR
- `baseline_early_fusion`：简单 6 通道 early fusion
- `dmfnet`：当前双分支融合方案

### 2.10 train / val 框架已经支持按配置自动选择是否多模态数据集

已修改文件：

- `ultralytics/models/yolo/detect/train.py`
- `ultralytics/models/yolo/detect/val.py`

已经完成：

- 当 `input_modality=multimodal` 时，启用多模态数据集逻辑
- 当 `input_modality=visible` 或 `infrared` 时，按单模态方式走

---

## 阶段 4：检查与辅助脚本

### 2.11 新增多模态检查脚本

新增文件：`tools/verify_multimodal_setup.py`

这个脚本用于：

- 打印当前实验配置
- 演示 VI 路径如何映射到 IR 路径
- 抽样检查 train 集中的 IR/VI 配对问题

### 2.12 新增测试脚手架

新增文件：`tests/test_multimodal_setup.py`

当前包含：

- 配对路径解析测试
- 通道顺序拆分测试
- 缺失文件 / 尺寸不一致问题测试
- 实验配置注册测试

注意：当前环境缺少部分运行依赖，所以这些测试文件已经写好，但本地环境里还没有完成完整 pytest 执行。

---

## 3. 当前默认训练配置

当前 `dmfnet` 默认配置定义在：`experiment_config.py`

关键参数如下：

- 实验名：`DMFNet`
- 模型：`improve_multimodal/DMFNet.yaml`
- 数据：`data.yaml`
- 输入模式：`multimodal`
- 通道顺序：`vi_ir`
- 训练尺寸：`640`
- epoch：`150`
- batch：`16`
- optimizer：`SGD`
- AMP：`False`
- 项目保存目录：`results/DMFNet`

---

## 4. 现在如何开始训练

## 4.1 训练前先做数据检查

先运行：

```bash
python tools/verify_multimodal_setup.py
```

作用：

- 查看当前默认实验是不是 `dmfnet`
- 查看当前通道顺序是不是 `vi_ir`
- 抽样检查配对图像是否存在缺失、尺寸不一致

如果这里就报错，先不要训练，先把数据问题修掉。

---

## 4.2 开始训练当前主模型 DMFNet

直接运行：

```bash
python train.py
```

当前 `train.py` 默认训练的是：

- `dmfnet`

训练产物默认会保存在：

```text
results/DMFNet/
```

其中最重要的权重是：

```text
results/DMFNet/weights/best.pt
```

---

## 4.3 训练完成后做验证

直接运行：

```bash
python val.py
```

当前 `val.py` 默认会读取：

```text
results/DMFNet/weights/best.pt
```

并按 `experiment_config.py` 中 `dmfnet` 的配置进行验证。

---

## 4.4 训练完成后做预测

直接运行：

```bash
python predict.py
```

当前 `predict.py` 默认会：

- 读取 `results/DMFNet/weights/best.pt`
- 预测 `datasets/images/val`
- 自动保存预测结果

---

## 5. 现在如何直接开始改进实验

由于单模态 `IR-only`、`VI-only` 和 `baseline_early_fusion` 你已经跑过了，这一轮配置已经做了清理，`experiment_config.py` 不再保留这些旧 baseline，只保留当前真正要继续推进的改进实验。

### 5.1 查看当前可运行实验

```bash
python train.py --list
```

或者：

```bash
python val.py --list
```

### 5.2 当前可以直接跑的改进实验

#### `dmfnet`

当前多模态主模型，对应：

```bash
python train.py dmfnet
```

#### `dmfnet_imgsz960`

优先推荐的第一组改进实验：只提高分辨率，观察小目标收益。

```bash
python train.py dmfnet_imgsz960
```

#### `dmfnet_imgsz1280`

更激进的高分辨率实验。

```bash
python train.py dmfnet_imgsz1280
```

#### `dmfnet_adamw`

优化器改进实验，用于对比 `SGD` 与 `AdamW`。

```bash
python train.py dmfnet_adamw
```

---

## 6. 推荐训练顺序

既然单模态和 early fusion 你已经跑过了，那当前建议直接按下面顺序进入改进实验：

1. `dmfnet`
2. `dmfnet_imgsz960`
3. `dmfnet_adamw`
4. `dmfnet_imgsz1280`

判断逻辑建议是：

- 如果 `imgsz960` 明显优于 `dmfnet`，说明当前主要瓶颈确实是小目标分辨率不足
- 如果 `adamw` 优于 `dmfnet`，说明当前训练优化方式还有明显空间
- 如果 `imgsz1280` 继续提升，下一阶段优先做 `P2 head`

---

## 7. 当前这一阶段还没有做的事

这一轮还没有开始动下面这些提分项：

- P2 小目标检测头
- DySample / CARAFE
- BiFPN / AFPN
- LQEHead
- NWD / TAL 系统扫参
- 预训练初始化迁移

也就是说，**当前完成的是 P0 工程基础修正阶段**，不是最终提分阶段。

---

## 8. 当前阶段的结论

当前已经完成的工作，可以概括成一句话：

> 已经把 `MMproject` 的多模态输入顺序、IR/VI 配对检查、训练/验证/预测入口、以及 baseline 脚手架统一到了同一个配置体系中，为后续真正做 mAP 提升实验打好了第一层基础。

现在你可以先做两件事：

1. 先运行 `python tools/verify_multimodal_setup.py dmfnet` 检查数据
2. 再运行 `python train.py dmfnet_imgsz960` 开始第一组改进实验

如果你下一步需要，我建议继续做：

- **在当前配置基础上继续加 `P2 head`**
- 然后再继续做 **DySample** 的第一版提分改造
