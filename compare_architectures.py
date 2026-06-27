"""
架构对比脚本 — YOLOv8s vs RT-DETR (ViT 探索实验)
在 NEU-DET 验证集上全面对比 CNN 与 Transformer 检测架构

对比维度:
  1. 精度 — 整体 mAP + 6 类逐类 AP
  2. 速度 — PyTorch FP32 纯推理时间
  3. 规模 — 参数量 / 模型文件大小
  4. 专项分析 — 细长缺陷 (裂纹+划痕) / 小目标 (点蚀) / 模糊边界 (氧化皮)

用法:
    python compare_architectures.py
"""
import time
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# ==================== 路径配置 ====================
DATA_YAML = r"D:\ultralytics\steel-defect-detection\data\NEU-DET\data_split.yaml"
RUNS_DIR = Path(r"D:\ultralytics\steel-defect-detection\runs\detect")

# 自动查找最新模型
def find_best_model(pattern: str) -> str | None:
    """查找匹配 pattern 的最新 best.pt"""
    candidates = sorted(
        RUNS_DIR.glob(f"{pattern}/weights/best.pt"),
        key=lambda p: p.stat().st_mtime
    )
    return str(candidates[-1]) if candidates else None

YOLO_MODEL = find_best_model("steel_improved_v*")
RTDETR_MODEL = find_best_model("rtdetr_steel_*")

# ==================== 辅助函数 ====================
def get_param_count(model_path: str) -> int:
    """获取模型参数量（通过 ultralytics API，安全且准确）"""
    from ultralytics import YOLO
    m = YOLO(model_path)
    params = sum(p.numel() for p in m.model.parameters())
    return params


def get_model_size_mb(model_path: str) -> float:
    """模型文件大小 (MB)"""
    return os.path.getsize(model_path) / (1024 * 1024)


def benchmark_pytorch(model_path: str, warmup: int = 30, runs: int = 300):
    """PyTorch FP32 纯推理时间测量 (与 benchmark_final.py 同方法)"""
    import torch
    import cv2
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    # 取一张验证集图片
    test_dir = Path(r"D:\ultralytics\steel-defect-detection\data\NEU-DET\split_val\images")
    imgs = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if not imgs:
        return None, None
    img_path = str(test_dir / imgs[0])

    img = cv2.imread(img_path)
    preproc = LetterBox(new_shape=640, auto=False, stride=32)
    im = preproc(image=img)
    im = np.transpose(im, (2, 0, 1))[::-1]
    im = np.ascontiguousarray(im).astype(np.float32) / 255.0
    im_tensor = torch.from_numpy(np.expand_dims(im, axis=0)).cuda()

    model = YOLO(model_path).model.cuda()
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(im_tensor)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(im_tensor)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


# ==================== 主流程 ====================
if __name__ == '__main__':
    print("=" * 80)
    print("架构对比实验 — YOLOv8s (CNN) vs RT-DETR (Transformer)")
    print("=" * 80)

    # ----- 检查模型可用性 -----
    available = {}
    if YOLO_MODEL and os.path.exists(YOLO_MODEL):
        available["YOLOv8s"] = YOLO_MODEL
        print(f"\n[✓] YOLOv8s: {YOLO_MODEL}")
    else:
        print(f"\n[✗] YOLOv8s: 未找到模型 (搜索 steel_improved_v*)")

    if RTDETR_MODEL and os.path.exists(RTDETR_MODEL):
        available["RT-DETR"] = RTDETR_MODEL
        print(f"[✓] RT-DETR: {RTDETR_MODEL}")
    else:
        print(f"[✗] RT-DETR: 未找到模型 (需先运行 train_rtdetr.py 训练)")

    if len(available) < 2:
        print("\n⚠️  需要进行对比的两个模型不全，请先完成训练。")
        print("   YOLOv8s: train_improved.py")
        print("   RT-DETR: train_rtdetr.py")
        sys.exit(0)

    # ==================== 1. 精度对比 ====================
    print("\n" + "=" * 80)
    print("1. 精度对比 (mAP)")
    print("=" * 80)

    ap_results = {}
    for name, model_path in available.items():
        print(f"\n--- {name} ---")
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model.val(data=DATA_YAML, imgsz=640, workers=0, verbose=False)

        ap_results[name] = {
            "map50": results.box.map50,
            "map": results.box.map,
            "ap50_per_class": {results.names[i]: results.box.ap50[i] for i in range(len(results.names))},
        }

        print(f"  mAP@0.5:     {results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"  各类别 AP@0.5:")
        for i, cls_name in results.names.items():
            ap = results.box.ap50[i]
            print(f"    {cls_name:20s} {ap:.4f}")

    # ----- 精度对比表格 -----
    print("\n" + "-" * 80)
    print("逐类别 AP@0.5 对比")
    print("-" * 80)

    if len(available) >= 2:
        names_list = list(available.keys())
        classes = list(ap_results[names_list[0]]["ap50_per_class"].keys())
        print(f"{'类别':<20s} {names_list[0]:>10s} {names_list[1]:>10s} {'差值':>10s}")
        print("-" * 55)

        for cls_name in classes:
            ap1 = ap_results[names_list[0]]["ap50_per_class"][cls_name]
            ap2 = ap_results[names_list[1]]["ap50_per_class"][cls_name]
            diff = ap2 - ap1
            winner = "←" if diff < -0.01 else ("→" if diff > 0.01 else "≈")
            print(f"{cls_name:<20s} {ap1:>10.4f} {ap2:>10.4f} {diff:>+9.4f} {winner}")

        map1 = ap_results[names_list[0]]["map50"]
        map2 = ap_results[names_list[1]]["map50"]
        diff = map2 - map1
        print("-" * 55)
        print(f"{'mAP@0.5 (整体)':<20s} {map1:>10.4f} {map2:>10.4f} {diff:>+9.4f}")

    # ==================== 2. 专项分析 ====================
    print("\n" + "=" * 80)
    print("2. 专项分析 — 按缺陷类型")
    print("=" * 80)

    # 缺陷分类
    slender_defects = ["crazing", "scratches"]        # 细长缺陷 → Transformer 优势?
    small_defects = ["pitted_surface"]                  # 小目标 → ViT patch 切分风险?
    boundary_blur = ["rolled-in_scale"]                 # 模糊边界
    regular = ["inclusion", "patches"]                  # 常规缺陷

    groups = {
        "细长缺陷 (裂纹+划痕)": slender_defects,
        "小目标 (点蚀)": small_defects,
        "模糊边界 (氧化皮)": boundary_blur,
        "常规缺陷 (夹杂+斑块)": regular,
    }

    if len(available) >= 2:
        for group_name, defect_list in groups.items():
            print(f"\n--- {group_name} ---")
            ap1_sum = sum(ap_results[names_list[0]]["ap50_per_class"][d] for d in defect_list)
            ap2_sum = sum(ap_results[names_list[1]]["ap50_per_class"][d] for d in defect_list)
            ap1_avg = ap1_sum / len(defect_list)
            ap2_avg = ap2_sum / len(defect_list)
            diff = ap2_avg - ap1_avg
            print(f"  {names_list[0]}: {ap1_avg:.4f}  →  {names_list[1]}: {ap2_avg:.4f}  (Δ {diff:+.4f})")
            for d in defect_list:
                ap1 = ap_results[names_list[0]]["ap50_per_class"][d]
                ap2 = ap_results[names_list[1]]["ap50_per_class"][d]
                print(f"    {d:20s}: {ap1:.4f} → {ap2:.4f}  ({ap2-ap1:+.4f})")

    # ==================== 3. 推理速度对比 ====================
    print("\n" + "=" * 80)
    print("3. 推理速度对比 (PyTorch FP32, RTX 4060)")
    print("=" * 80)

    speed_results = {}
    for name, model_path in available.items():
        print(f"\n--- {name} ---")
        mean_ms, std_ms = benchmark_pytorch(model_path)
        speed_results[name] = (mean_ms, std_ms)
        print(f"  推理时间: {mean_ms:.2f} ± {std_ms:.2f} ms")
        if mean_ms:
            print(f"  FPS:      {1000/mean_ms:.1f}")

    if len(speed_results) >= 2:
        t1, _ = speed_results[names_list[0]]
        t2, _ = speed_results[names_list[1]]
        if t1 and t2:
            ratio = t2 / t1
            print(f"\n  速度比: {names_list[1]} / {names_list[0]} = {ratio:.2f}x")
            print(f"  {'RT-DETR 慢 ' + str(ratio) + 'x' if ratio > 1 else 'RT-DETR 快 ' + str(1/ratio) + 'x'}")

    # ==================== 4. 模型规模对比 ====================
    print("\n" + "=" * 80)
    print("4. 模型规模对比")
    print("=" * 80)

    for name, model_path in available.items():
        size_mb = get_model_size_mb(model_path)
        print(f"\n--- {name} ---")
        print(f"  文件大小: {size_mb:.1f} MB")

        try:
            params = get_param_count(model_path)
            if params > 0:
                print(f"  参数量:   {params/1e6:.1f}M")
        except Exception as e:
            print(f"  参数量:   获取失败 ({e})")

    # ==================== 5. 综合结论 ====================
    print("\n" + "=" * 80)
    print("5. 综合结论")
    print("=" * 80)

    if len(available) >= 2:
        map1 = ap_results[names_list[0]]["map50"]
        map2 = ap_results[names_list[1]]["map50"]
        diff_map = map2 - map1
        t1 = speed_results[names_list[0]][0]
        t2 = speed_results[names_list[1]][0]

        # 细长缺陷对比
        slender1 = np.mean([ap_results[names_list[0]]["ap50_per_class"][d] for d in slender_defects])
        slender2 = np.mean([ap_results[names_list[1]]["ap50_per_class"][d] for d in slender_defects])
        slender_diff = slender2 - slender1

        # 小目标对比
        small1 = ap_results[names_list[0]]["ap50_per_class"]["pitted_surface"]
        small2 = ap_results[names_list[1]]["ap50_per_class"]["pitted_surface"]

        print(f"\n  📊 整体精度:")
        print(f"     {names_list[0]}: mAP@0.5 = {map1:.4f}")
        print(f"     {names_list[1]}: mAP@0.5 = {map2:.4f}")
        print(f"     Δ = {diff_map:+.4f} → {'RT-DETR 更优' if diff_map > 0.01 else ('YOLOv8s 更优' if diff_map < -0.01 else '持平')}")

        print(f"\n  🔍 细长缺陷 (裂纹+划痕): Δ AP = {slender_diff:+.4f}")
        if slender_diff > 0.02:
            print(f"     → RT-DETR self-attention 对长距离依赖有优势 ✓ (大纲预期)")

        print(f"\n  🎯 小目标 (点蚀): Δ AP = {small2-small1:+.4f}")
        if small2 < small1 - 0.02:
            print(f"     → ViT patch 切分可能丢失小目标 ⚠️ (大纲预期)")

        print(f"\n  ⚡ 推理速度:")
        print(f"     {names_list[0]}: {t1:.2f} ms")
        print(f"     {names_list[1]}: {t2:.2f} ms ({t2/t1:.1f}x)")

        # 生成简历可用的结论文本
        print("\n" + "-" * 80)
        print("📝 简历结论建议 (可直接用于 resume_text.json):")
        print("-" * 80)

        if diff_map > 0.01:
            print(f"\n  \"RT-DETR 在裂纹检测上优于 YOLOv8s（AP {slender_diff:+.1%}），"
                  f"self-attention 对细长缺陷的全局建模能力弥补了 CNN 感受野局限。\n"
                  f" 但推理速度慢 {t2/t1:.1f}x，工业部署需权衡精度-效率。\"")
        elif diff_map < -0.01:
            print(f"\n  \"对比 RT-DETR，YOLOv8s 在 1800 张小样本工业数据集上泛化更优"
                  f"（mAP +{abs(diff_map):.1%}），小数据场景下 CNN 的归纳偏置仍有优势。\"")
        else:
            print(f"\n  \"RT-DETR 在细长缺陷（裂纹）上更优（AP +{max(0, slender_diff):.1%}），"
                  f"YOLOv8s 在推理速度上占优（{t2/t1:.1f}x），工业部署应场景化选型。\"")

    print("\n" + "=" * 80)
    print("对比实验完成！")
    print("=" * 80)
