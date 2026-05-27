"""
改进版训练脚本 — 针对低精度缺陷类别优化
关键改进:
1. 正确的80/20数据划分 (1440/360 vs 原1770/30)
2. imgsz=640 保留细纹理缺陷细节
3. 针对裂纹/氧化皮的温和增强策略
4. 提高 cls loss 权重聚焦难分类样本
5. 多尺度训练 + 合理早停
"""
from ultralytics import YOLO
import os
import shutil
import random
from pathlib import Path

random.seed(42)

# ===================== 0. 数据集重划分 =====================
def resplit_dataset():
    """将 NEU-DET 从 98/2 重新划分为 80/20，保证每类均匀分布"""
    src_train = Path(r"D:\ultralytics\steel-defect-detection\data\NEU-DET\train")
    src_valid = Path(r"D:\ultralytics\steel-defect-detection\data\NEU-DET\valid")

    all_images = list(src_train.glob("images/*.jpg")) + list(src_valid.glob("images/*.jpg"))
    all_labels = list(src_train.glob("labels/*.txt")) + list(src_valid.glob("labels/*.txt"))

    classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    class_images = {c: [] for c in classes}
    for img in all_images:
        for c in classes:
            if c in img.stem:
                class_images[c].append(img)
                break

    new_train_dir = Path(r"D:\ultralytics\steel-defect-detection\data\NEU-DET\split_train")
    new_val_dir = Path(r"D:\ultralytics\steel-defect-detection\data\NEU-DET\split_val")

    for d in [new_train_dir / "images", new_train_dir / "labels",
              new_val_dir / "images", new_val_dir / "labels"]:
        d.mkdir(parents=True, exist_ok=True)

    for cls_name, imgs in class_images.items():
        random.shuffle(imgs)
        n_train = int(len(imgs) * 0.8)
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:]

        for img in train_imgs:
            label = img.parent.parent / "labels" / (img.stem + ".txt")
            shutil.copy2(img, new_train_dir / "images" / img.name)
            if label.exists():
                shutil.copy2(label, new_train_dir / "labels" / label.name)

        for img in val_imgs:
            label = img.parent.parent / "labels" / (img.stem + ".txt")
            shutil.copy2(img, new_val_dir / "images" / img.name)
            if label.exists():
                shutil.copy2(label, new_val_dir / "labels" / label.name)

        print(f"{cls_name}: {len(train_imgs)} train / {len(val_imgs)} val")

    print("数据集80/20重划分完成!")
    return str(new_train_dir), str(new_val_dir)


DATA_YAML = r"D:\ultralytics\steel-defect-detection\data\NEU-DET\data_split.yaml"

def create_data_yaml():
    train_path, val_path = resplit_dataset()
    yaml_content = f"""path: D:/ultralytics/steel-defect-detection/data/NEU-DET
train: {train_path}/images
val: {val_path}/images
nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
"""
    with open(DATA_YAML, 'w') as f:
        f.write(yaml_content)
    print(f"数据配置文件已创建: {DATA_YAML}")


def find_latest_checkpoint():
    """查找最新训练的last.pt，用于中断后复训"""
    runs_dir = Path(r"D:\ultralytics\steel-defect-detection\runs\detect")
    if not runs_dir.exists():
        return None, None

    checkpoints = sorted(runs_dir.glob("steel_improved_*/weights/last.pt"),
                         key=lambda p: p.stat().st_mtime)
    if checkpoints:
        ckpt = checkpoints[-1]
        run_name = ckpt.parent.parent.name
        return str(ckpt), run_name
    return None, None


def get_next_run_name():
    """获取下一个可用的run名称"""
    runs_dir = Path(r"D:\ultralytics\steel-defect-detection\runs\detect")
    existing = list(runs_dir.glob("steel_improved_*"))
    indices = []
    for p in existing:
        try:
            idx = int(p.name.replace("steel_improved_v", ""))
            indices.append(idx)
        except ValueError:
            pass
    next_idx = max(indices) + 1 if indices else 1
    return f"steel_improved_v{next_idx}"


if __name__ == '__main__':
    if not os.path.exists(DATA_YAML):
        create_data_yaml()

    ckpt_path, resumed_name = find_latest_checkpoint()

    if ckpt_path:
        print(f"\n{'='*60}")
        print(f"检测到中断的训练: {ckpt_path}")
        print(f"将从 checkpoint 继续训练 (run: {resumed_name})")
        print(f"{'='*60}\n")
        model = YOLO(ckpt_path)
        run_name = resumed_name
    else:
        print(f"\n{'='*60}")
        print("未检测到checkpoint，开始全新训练")
        print(f"{'='*60}\n")
        model = YOLO("yolov8s.yaml")
        run_name = get_next_run_name()

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        name=run_name,
        amp=False,

        fliplr=0.5,
        flipud=0.2,
        degrees=3,
        translate=0.05,
        scale=0.3,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,

        mosaic=0.3,
        mixup=0.0,
        cutmix=0.0,
        erasing=0.0,
        auto_augment=None,
        shear=0.0,
        perspective=0.0,

        cls=1.0,
        box=7.5,
        dfl=1.5,

        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        patience=20,

        multi_scale=0.5,
        close_mosaic=15,

        pretrained=True,
        save=True,
        plots=True,

        resume=bool(ckpt_path),
    )

    print("\n训练完成！运行 calculate_class_ap.py 评估各类别精度。")
