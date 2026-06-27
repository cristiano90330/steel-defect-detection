"""
RT-DETR 训练脚本 — ViT 探索实验
目标：在 NEU-DET 钢铁缺陷数据集上训练 RT-DETR，与 YOLOv8s 做控制变量对比

关键设计：
1. 复用与 YOLOv8s 完全相同的数据划分 (data_split.yaml)，保证公平对比
2. 使用 rtdetr-l.pt（~32M 参数），对应 YOLOv8s 的 11.1M
3. RT-DETR 使用 AdamW 优化器 + 低学习率，与 CNN 的 SGD 不同
4. 不启用 mosaic/mixup（Transformer 本身有全局注意力，强增强有害）
5. 保持温和的几何/颜色增强，控制过拟合风险（1800 张对 Transformer 偏少）

用法：
    python train_rtdetr.py          # 全新训练
    python train_rtdetr.py --resume # 从最新 checkpoint 续训
"""
from ultralytics import RTDETR
import os
from pathlib import Path

DATA_YAML = r"D:\ultralytics\steel-defect-detection\data\NEU-DET\data_split.yaml"
RUNS_DIR = Path(r"D:\ultralytics\steel-defect-detection\runs\detect")


def find_latest_checkpoint():
    """查找最新的 RT-DETR checkpoint，支持中断后复训"""
    if not RUNS_DIR.exists():
        return None, None
    checkpoints = sorted(
        RUNS_DIR.glob("rtdetr_steel_*/weights/last.pt"),
        key=lambda p: p.stat().st_mtime
    )
    if checkpoints:
        ckpt = checkpoints[-1]
        run_name = ckpt.parent.parent.name
        return str(ckpt), run_name
    return None, None


def get_next_run_name():
    """获取下一个可用的 RT-DETR run 名称"""
    existing = list(RUNS_DIR.glob("rtdetr_steel_*"))
    indices = []
    for p in existing:
        try:
            idx = int(p.name.replace("rtdetr_steel_v", ""))
            indices.append(idx)
        except ValueError:
            pass
    next_idx = max(indices) + 1 if indices else 1
    return f"rtdetr_steel_v{next_idx}"


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="从最新 checkpoint 续训")
    args = parser.parse_args()

    # 验证数据配置存在
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"数据配置文件不存在: {DATA_YAML}\n"
            "请先运行 train_improved.py 生成数据划分。"
        )

    ckpt_path, resumed_name = find_latest_checkpoint()

    if args.resume and ckpt_path:
        print(f"\n{'='*60}")
        print(f"从 checkpoint 续训: {ckpt_path}")
        print(f"Run: {resumed_name}")
        print(f"{'='*60}\n")
        model = RTDETR(ckpt_path)
        run_name = resumed_name
    elif args.resume and not ckpt_path:
        print("未找到 checkpoint，将开始全新训练。")
        model = RTDETR("rtdetr-l.pt")
        run_name = get_next_run_name()
    else:
        print(f"\n{'='*60}")
        print("RT-DETR 全新训练 — ViT 探索实验")
        print(f"{'='*60}\n")
        model = RTDETR("rtdetr-l.pt")
        run_name = get_next_run_name()

    model.train(
        # ==================== 数据配置 ====================
        data=DATA_YAML,        # 与 YOLOv8s 完全相同的数据划分
        epochs=100,            # 与 YOLOv8s 相同
        imgsz=640,             # 与 YOLOv8s 相同
        batch=4,               # RT-DETR-l 参数多 (~32M)，显存有限降为 4
        device=0,
        workers=0,
        name=run_name,
        amp=True,              # RT-DETR 支持 AMP 混合精度，节省显存

        # ==================== 数据增强（温和策略） ====================
        # 与 YOLOv8s 保持一致的增强参数，控制变量
        fliplr=0.5,            # 水平翻转
        flipud=0.2,            # 垂直翻转
        degrees=3,             # 随机旋转 ±3°
        translate=0.05,        # 平移 ±5%
        scale=0.3,             # 缩放 ±30%
        hsv_h=0.01,            # 色相微调
        hsv_s=0.3,             # 饱和度
        hsv_v=0.2,             # 明度

        # RT-DETR 不使用 mosaic/mixup（Transformer 全局注意力已覆盖）
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        erasing=0.0,           # 随机擦除可能破坏细纹理缺陷
        auto_augment=None,
        shear=0.0,
        perspective=0.0,

        # ==================== 损失权重 ====================
        cls=1.0,               # 分类损失
        box=7.5,               # 边界框损失
        dfl=1.5,               # DFL 损失

        # ==================== 优化器参数 ====================
        # RT-DETR 默认使用 AdamW，学习率比 SGD 低两个数量级
        lr0=1e-4,              # AdamW 初始学习率 (vs SGD 的 0.01)
        lrf=0.01,              # 最终 lr = 1e-4 × 0.01 = 1e-6
        weight_decay=1e-4,     # AdamW 权重衰减
        warmup_epochs=3,       # 学习率预热
        cos_lr=True,           # 余弦退火
        patience=15,           # 15 轮无提升早停（比 YOLO 更激进，防过拟合）

        # ==================== 其他 ====================
        pretrained=True,       # COCO 预训练权重
        save=True,
        plots=True,
        resume=args.resume and bool(ckpt_path),
    )

    print("\n" + "=" * 60)
    print("RT-DETR 训练完成！")
    print(f"模型保存在: {RUNS_DIR / run_name / 'weights' / 'best.pt'}")
    print("接下来运行 compare_architectures.py 进行 YOLOv8s vs RT-DETR 对比评估。")
    print("=" * 60)
