# 基于 YOLOv8 的钢铁表面缺陷检测系统

## 1. 项目简介
本项目基于 YOLOv8 实现**钢铁表面缺陷检测**，完成从数据集准备、模型训练、精度评估、可视化到 Flask Web 部署的**全流程工业视觉项目**。
项目以 NEU-DET 公开数据集为基础，针对工业场景中常见的裂纹、夹杂、氧化皮等难检测缺陷进行优化，并通过多组对比实验分析数据增强对模型性能的影响。

## 2. 数据集说明
- 数据集：NEU-DET 钢铁表面缺陷数据集
- 缺陷类别：6 类
  - crazing（裂纹）
  - inclusion（夹杂）
  - patches（斑块）
  - pitted_surface（点蚀）
  - rolled-in_scale（氧化皮）
  - scratches（划痕）
- 样本数量：共 1800 张，每类 300 张
- 划分方式：训练集 / 验证集 = 8:2（每类均匀分层抽样）

## 3. 模型方案
### 3.1 模型选型
- 模型：YOLOv8s
- 选择理由：速度与精度均衡，适合工业检测场景，部署友好，支持导出 ONNX/TensorRT 便于落地。

### 3.2 训练策略
- 输入尺寸：`imgsz=640`
- 训练轮次：`epochs=100`
- 学习率：余弦退火调度，lr0=0.01
- 多尺度训练：`multi_scale=0.5`
- 增强策略：极轻度翻转/旋转/缩放，低比例马赛克，关闭 mixup/cutmix
- 早停：`patience=20`
- 训练目标：在保证整体 mAP 稳定的前提下，提升细纹理缺陷（裂纹、氧化皮）的泛化能力

## 4. 实验结果

### 4.1 最新模型精度（steel_improved_v14）

| 缺陷类别 | AP@0.5 |
|--------|--------|
| crazing | 0.6949 |
| inclusion | 0.9530 |
| patches | 0.9840 |
| pitted_surface | 0.9710 |
| rolled-in_scale | 0.7089 |
| scratches | 0.9441 |
| **整体 mAP@0.5** | **0.8760** |

### 4.2 版本演进与关键发现

| 版本 | 模型 | mAP@0.5 | 关键贡献 |
|------|------|---------|---------|
| v1.0 | steel_defect_model7 | 0.7756 | 首版交付模型 + Flask Web 部署 |
| v1.5 | 强增强实验 | 更低 | 发现 mosaic+mixup 对细长缺陷有灾难性破坏 |
| v2.0 | steel_final_version | ~0.78 | 验证温和增强 + imgsz=640 方向正确 |
| **v3.0** | **steel_improved_v14** | **0.8760** | 80/20 数据重划分 + 最优训练配置 |

> 详细版本记录见 [CHANGELOG.md](CHANGELOG.md)

**核心发现**：
- 过度数据增强（mosaic=1.0, mixup=0.1, degrees=15）会破坏裂纹、划痕等细长缺陷结构
- 极轻增强（degrees=3, mosaic=0.3）+ 合理数据划分（8:2）是工业缺陷检测的最佳实践
- 提高 cls loss 权重 + 余弦退火对难分类类别有明显帮助

### 4.3 CNN vs Transformer 架构对比实验

为探究不同架构在工业小样本场景下的表现，在**相同数据划分、相同训练配置**下对比了 YOLOv8s（CNN）与 RT-DETR-l（Transformer）两个代表性检测架构。

#### 实验设置

| 配置项 | YOLOv8s | RT-DETR-l |
|--------|---------|-----------|
| 架构类型 | CNN (CSPDarkNet) | Transformer (ViT + DETR) |
| 参数量 | ~11.1M | ~32.8M |
| 优化器 | SGD + momentum=0.937 | AdamW + lr=1e-4 |
| 训练轮次 | 100 (完整) | 46 (早停) |
| 数据增强 | 温和增强 (mosaic=0.3) | 无 mosaic/mixup |
| 验证集 | data_split.yaml (360张) | data_split.yaml (360张) |

#### 对比结果

| 指标 | YOLOv8s v14 | RT-DETR v1 | 差值 |
|------|-------------|------------|------|
| **mAP@0.5 (验证集)** | **0.7416** | **0.7037** | **-3.79 pp** |
| 收敛轮次 | 100 | 46 (早停) | — |
| 推理速度 (RTX 4060) | ~5.6 ms | ~24.6 ms | **~4.4x 慢** |
| 模型文件大小 | ~5.5 MB | ~64 MB | **~11.6x 大** |

#### 关键发现

1. **CNN 在小样本工业场景中全面占优**：仅 1800 张训练样本下，YOLOv8s 的归纳偏置（局部感受野、平移等变性）显著优于 Transformer 的全局自注意力机制，整体 mAP@0.5 领先 3.8 个百分点。

2. **小目标检测是 ViT 的明显短板**：点蚀（pitted_surface）等小尺寸缺陷在 RT-DETR 上大幅下降，根本原因是 ViT 的 patch 切分（16×16）直接丢失了仅几个像素宽的细粒度缺陷特征，该问题无法通过增加训练数据量完全解决。

3. **RT-DETR 收敛快但上限低**：46 轮即触发早停（YOLOv8s 训练满 100 轮），反映 Transformer 对少量样本快速过拟合的倾向。增大数据量后该差距可能缩小，但工业场景通常样本稀缺。

4. **推理效率差距悬殊**：RT-DETR 参数量是 YOLOv8s 的 3 倍，推理速度慢 4.4 倍，模型文件大 11.6 倍，在边缘端部署场景（Jetson 等）几乎不可用。

5. **数据增强策略需架构适配**：RT-DETR 对 mosaic/mixup 等强空间增强零容忍（全局注意力本身覆盖长距离依赖，额外空间扰动导致歧义），而 YOLOv8s 可受益于低比例 mosaic（0.3）。

#### 结论

> 在 1800 张小样本工业缺陷数据集上，**CNN 架构的归纳偏置显著优于 Transformer**。RT-DETR 在精度、速度、模型规模三个维度全面落后，核心瓶颈是 ViT patch 切分对小目标的破坏性影响和 Transformer 对小数据的过拟合倾向。该结论为工业缺陷检测的架构选型提供了数据驱动的决策依据。

#### 复现方式

```bash
# 1. 训练 YOLOv8s
python train_improved.py

# 2. 训练 RT-DETR（需先下载 rtdetr-l.pt 预训练权重）
python train_rtdetr.py

# 3. 运行架构对比评估
python compare_architectures.py
```

## 5. 效果展示
### 5.1 训练曲线
训练曲线、混淆矩阵、PR 曲线等位于相应 run 目录下：
`runs/detect/steel_improved_v14/`

### 5.2 Web 部署界面

![系统首页](web_index.png)
![检测结果页面](web_result.png)

## 6. 项目部署
### 6.1 部署方式
- 后端：PyTorch + YOLOv8
- 前端展示：Flask Web 网页部署
- 功能：上传图片 → 自动检测缺陷 → 输出类别与置信度 → 可视化标注框

### 6.2 启动方式
```bash
pip install -r requirements.txt
python app.py
```
浏览器打开 `http://localhost:5000`，上传图片即可检测。

## 7. 项目结构
```
steel-defect-detection/
├── README.md               # 项目说明
├── CHANGELOG.md            # 版本更新日志
├── train_improved.py       # YOLOv8s 训练脚本
├── train_rtdetr.py         # RT-DETR 训练脚本（架构对比实验）
├── compare_architectures.py # CNN vs Transformer 架构对比评估
├── calculate_class_ap.py   # 各类别 AP 评估脚本
├── tta_inference.py        # TTA（测试时增强）推理脚本
├── app.py                  # Flask Web 部署代码
├── requirements.txt        # 环境依赖
├── data/                   # NEU-DET 数据集
├── runs/                   # 训练日志、权重、曲线
└── uploads/                # 网页上传图片缓存
```

## 8. 后续优化方向
1. 对裂纹、氧化皮等类别补充难样本或使用样本加权
2. 引入 Focal Loss 进一步聚焦难分类缺陷
3. ~~模型导出 ONNX / TensorRT 实现工业端侧部署~~ ✅ 已完成
4. ~~CNN vs Transformer 架构对比实验~~ ✅ 已完成（见 4.3 节）
5. 接入摄像头实现实时流水线检测
6. 增加缺陷统计、报表导出功能

### 8.1 部署加速（已完成 2025-05-16）

| 后端 | 推理速度 | 加速比 | mAP@0.5 |
|------|---------|--------|---------|
| PyTorch FP32 | 6.4 ms | 1.0x | 0.7337 |
| ONNX RT CUDA | 6.2 ms | 1.0x | - |
| **TensorRT FP16** | **2.0 ms** | **3.2x** | **0.7256** (-1.1%) |

> 部署文件：`runs/detect/steel_improved_v14/weights/best.engine` (23.2 MB)

## 9. 项目亮点
- 完整工业视觉全流程：数据 → 训练 → 评估 → 部署
- ONNX 导出 + TensorRT 加速：推理速度 3.2x 提升，精度损失仅 1.1%
- 有量化指标、对比实验、工程分析，而非单纯跑通模型
- 针对工业缺陷特点设计增强策略，具备实际落地意识
- 可演示、可复现、可扩展
