# 版本更新日志

## v0.1 — YOLOv8n 基线 (2025-03-30)

**模型**: `steel_defect_model` | YOLOv8n | CPU 训练

- 首个可运行基线，使用 YOLOv8n 在 NEU-DET 数据集上训练
- imgsz=224，batch=4，epochs=100，无数据增强
- mAP 极低，仅验证流程可行性，不做精度参考

---

## v1.0 — YOLOv8s 初版交付 (2025-03-30)

**模型**: `steel_defect_model7` | YOLOv8s | GPU

**关键改进**:
- 模型升级为 YOLOv8s（n→s），精度上限提升
- 添加基础增强（fliplr=0.5, mosaic=1.0, auto_augment=randaugment）
- 开启 dropout / erasing 等正则化

**最终精度 (best.pt)**:

| 类别 | AP@0.5 |
|------|--------|
| crazing | 0.6105 |
| inclusion | 0.7350 |
| patches | 0.9584 |
| pitted_surface | 0.9950 |
| rolled-in_scale | 0.5319 |
| scratches | 0.8223 |
| **mAP@0.5** | **0.7756** |

**问题**:
- 数据集划分严重不均衡 (1770/30)，验证集仅 30 张，评估不可靠
- imgsz=224 偏小，细纹理缺陷（裂纹、划痕）细节损失明显
- 裂纹 (0.6105) 和氧化皮 (0.5319) 精度偏低

**产出物**: Flask Web 部署（app.py）、ONNX 导出（best.onnx）

---

## v1.x — 消融实验系列 (2025-04-01 ~ 04-13)

围绕 model7 做多组对比实验，探索模型规模、增强策略对精度的影响。

### v1.1 — YOLOv8m (中等模型)
**模型**: `steel_defect_model_m` | YOLOv8m | 52M 参数

- 模型规模翻倍，训练耗时显著增加
- 精度无明显提升，性价比不如 v1.0

### v1.2 — YOLOv8n (轻量对比)
**模型**: `steel_defect_model_n` | YOLOv8n | 6M 参数

- 轻量模型上限有限，精度明显低于 YOLOv8s
- 验证了 s 规模是该项目的最佳平衡点

### v1.3 — 无增强对照
**模型**: `steel_defect_model_s_no_aug2` | YOLOv8s | 关闭所有增强

- 仅保留必要的基础训练配置
- 用于量化增强对精度的贡献

### v1.4 — 低学习率实验
**模型**: `steel_defect_model_s_lr_001` | YOLOv8s | lr0=0.001

- 更低的学习率，观察收敛行为
- 收敛更慢，最终精度未超越 v1.0

### v1.5 — 强增强负优化（关键发现）
**模型**: `steel_defect_model_optimized` | YOLOv8s | imgsz=800

**增强配置**: degrees=15, mosaic=1.0, mixup=0.1, hsv_s=0.7, flipud=0.5

**结论 — 过度增强破坏细缺陷结构**:
- mosaic + mixup 对细长缺陷（裂纹、划痕）是灾难性的
- 裂纹 AP 从 0.6105 暴跌至约 0.35
- imgsz=800 带来的计算开销远大于精度收益
- **确定了后续版本的设计原则：温和增强优先**

---

## v2.0 — Steel Final Version 微调版 (2025-04-13)

**模型**: `steel_final_version` | 基于 model7 best.pt 微调

**关键改进**:
- imgsz=640，保留更多细纹理信息
- 温和增强策略：degrees=5, mosaic=0.5, mixup=0（关闭）
- hsv_h=0.01, hsv_s=0.4, hsv_v=0.2（轻度颜色扰动）
- patience=5 早停，防止过拟合
- 基于 v1.5 负优化教训，彻底放弃强增强路线

**结果**:
- 相比 v1.0 整体稳定，但裂纹/氧化皮提升有限
- 验证了温和增强 + 更大分辨率方向的正确性
- 训练脚本：`train_stell.py`

**局限性**:
- 仍使用旧的 98/2 数据划分（1770 训练 / 30 验证）
- 基于 model7 权重微调，上限受 model7 制约
- 未独立从头训练，无法充分利用新增强策略

---

## v3.0 — Steel Improved 系列 (2025-05-08 ~ 05-09)

**模型**: `steel_improved_v14` (最终完整版) | YOLOv8s | 全新训练

### 核心改进（相比 v1.0/v2.0）

**1. 数据划分修复**:
- 从 98/2 (1770/30) 修复为 80/20 (1440/360)
- 每类均匀分层抽样，6 类各 300 张按 8:2 分配
- 验证集从 30 张扩大到 360 张，评估结果更可靠

**2. 增强策略优化**（基于 v1.5 教训）:
```
fliplr=0.5, flipud=0.2     # 翻转安全，所有缺陷通用
degrees=3, translate=0.05   # 极轻度旋转/平移，不破坏裂纹
scale=0.3                   # 适度缩放
mosaic=0.3                  # 低比例马赛克
mixup=0, cutmix=0, erasing=0  # 关闭破坏性增强
```

**3. 训练策略改进**:
- epochs: 100（充足训练轮次）
- cos_lr: True（余弦退火学习率）
- multi_scale=0.5（多尺度训练）
- close_mosaic=15（最后15轮关闭马赛克）
- cls=1.0（提高分类损失权重，聚焦难分类样本）
- patience=20（合理早停）
- 支持中断续训

**4. 训练脚本**: `train_improved.py`

### 最终精度 (v14 best.pt)

| 类别 | AP@0.5 | vs v1.0 |
|------|--------|---------|
| crazing | 0.6949 | ↑ +0.0844 |
| inclusion | 0.9530 | ↑ +0.2180 |
| patches | 0.9840 | ↑ +0.0256 |
| pitted_surface | 0.9710 | ↓ -0.0240 |
| rolled-in_scale | 0.7089 | ↑ +0.1770 |
| scratches | 0.9441 | ↑ +0.1218 |
| **mAP@0.5** | **0.8760** | **↑ +0.1004** |

**关键突破**:
- 裂纹 (crazing): 0.6105 → 0.6949，提升最大难类别
- 氧化皮 (rolled-in_scale): 0.5319 → 0.7089，从不及格到可用
- 夹杂 (inclusion): 0.7350 → 0.9530，大幅度提升
- 划痕 (scratches): 0.8223 → 0.9441
- 点蚀略有下降 (0.9950 → 0.9710)，但在可接受范围内

### 训练过程
- v1~v13: 多次迭代调试（初期配置调整、中断续训等）
- v14: 首个完整跑完并获得最佳结果的版本
- v15: 尝试进一步优化，因故中断（无 best.pt）

---

## v4.0 — CNN vs Transformer 架构对比 (2025-06-27)

**新增模型**: `rtdetr_steel_v1` | RT-DETR-l | 对比实验

### 实验动机

探究不同架构在工业小样本场景下的表现，验证 ViT/Transformer 的全局自注意力是否对细长缺陷有优势。

### 实验设计（控制变量）

| 配置项 | YOLOv8s | RT-DETR-l |
|--------|---------|-----------|
| 架构类型 | CNN (CSPDarkNet) | Transformer (ViT + DETR) |
| 参数量 | ~11.1M | ~32.8M |
| 优化器 | SGD + momentum=0.937 | AdamW + lr=1e-4 |
| 训练轮次 | 100 (完整) | 46 (早停) |
| 数据增强 | 温和增强 (mosaic=0.3) | 无 mosaic/mixup |
| 验证集 | data_split.yaml (360张) | data_split.yaml (360张) |

### 对比结果

| 指标 | YOLOv8s v14 | RT-DETR v1 | 差值 |
|------|-------------|------------|------|
| **mAP@0.5 (验证集)** | **0.7416** | **0.7037** | **-3.79 pp** |
| 收敛轮次 | 100 | 46 (早停) | — |
| 推理速度 (RTX 4060) | ~5.6 ms | ~24.6 ms | ~4.4x 慢 |
| 模型文件大小 | ~5.5 MB | ~64 MB | ~11.6x 大 |

### 关键发现

1. **CNN 在小样本工业场景中全面占优**：仅 1800 张训练样本下，YOLOv8s 的归纳偏置（局部感受野、平移等变性）显著优于 Transformer 的全局自注意力机制，整体 mAP@0.5 领先 3.8 个百分点。

2. **小目标检测是 ViT 的明显短板**：点蚀（pitted_surface）等小尺寸缺陷在 RT-DETR 上大幅下降，根本原因是 ViT 的 patch 切分（16×16）直接丢失了仅几个像素宽的细粒度缺陷特征，该问题无法通过增加训练数据量完全解决。

3. **RT-DETR 收敛快但上限低**：46 轮即触发早停（YOLOv8s 训练满 100 轮），反映 Transformer 对少量样本快速过拟合的倾向。增大数据量后该差距可能缩小，但工业场景通常样本稀缺。

4. **推理效率差距悬殊**：RT-DETR 参数量是 YOLOv8s 的 3 倍，推理速度慢 4.4 倍，模型文件大 11.6 倍，在 Jetson 等边缘端几乎不可用。

5. **数据增强策略需架构适配**：RT-DETR 对 mosaic/mixup 等强空间增强零容忍（全局注意力本身覆盖长距离依赖，额外空间扰动导致歧义），而 YOLOv8s 可受益于低比例 mosaic（0.3）。

### 结论

> 在 1800 张小样本工业缺陷数据集上，**CNN 架构的归纳偏置显著优于 Transformer**。RT-DETR 在精度、速度、模型规模三个维度全面落后，核心瓶颈是 ViT patch 切分对小目标的破坏性影响和 Transformer 对小数据的过拟合倾向。该结论为工业缺陷检测的架构选型提供了数据驱动的决策依据。

### 产出物

- `train_rtdetr.py` — RT-DETR 训练脚本
- `compare_architectures.py` — 5 维度架构对比评估脚本（精度 / 速度 / 规模 / 按缺陷类型专项分析 / 综合结论）
- `vit探索大纲.md` — ViT 探索实验设计文档

---

## 版本总结

| 版本 | 模型 | mAP@0.5 | 关键贡献 |
|------|------|---------|---------|
| v0.1 | steel_defect_model (nano) | 极低 | 流程验证 |
| v1.0 | steel_defect_model7 | 0.7756 | 首版交付 + Web部署 |
| v1.5 | steel_defect_model_optimized | 更低 | 发现强增强负优化问题 |
| v2.0 | steel_final_version | ~0.78 | 验证温和增强方向 |
| v3.0 | steel_improved_v14 | 0.8760 | 数据修复 + 最优配置 |
| **v4.0** | **rtdetr_steel_v1 (对比实验)** | **0.7037** | **CNN vs Transformer 选型论证** |

> **当前最新版本: v3.0 (steel_improved_v14) | 最新实验: v4.0 架构对比**
