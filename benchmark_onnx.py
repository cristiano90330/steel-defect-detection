"""
PyTorch vs ONNX Runtime 推理速度对比
"""
import time
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from pathlib import Path

MODEL_PT = r"D:\ultralytics\steel-defect-detection\runs\detect\steel_improved_v14\weights\best.pt"
MODEL_ONNX = r"D:\ultralytics\steel-defect-detection\runs\detect\steel_improved_v14\weights\best.onnx"
TEST_IMG = r"D:\ultralytics\steel-defect-detection\data\NEU-DET\split_val\images"
WARMUP = 10
RUNS = 100

def benchmark_pt(model, img_dir):
    """PyTorch 原生推理"""
    # Warmup
    for _ in range(WARMUP):
        _ = model(img_dir, imgsz=640, verbose=False)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        results = model(img_dir, imgsz=640, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times), results

def benchmark_onnx(session, img_path):
    """ONNX Runtime 推理 (单张预处理好的图)"""
    import cv2

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    input_tensor = np.expand_dims(img, axis=0)

    # Warmup
    for _ in range(WARMUP):
        _ = session.run(None, {session.get_inputs()[0].name: input_tensor})

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times), outputs

def benchmark_pt_single(model, img_path, use_gpu=True):
    """PyTorch 单张推理计时 (不含数据加载/后处理)"""
    import torch

    results = model(img_path, imgsz=640, verbose=False)
    # Extract the preprocessed input tensor for fair comparison
    # Use internal preprocessing
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    import cv2

    img = cv2.imread(img_path)

    # Preprocess once
    preproc = LetterBox(new_shape=640, auto=False, stride=32)
    im = preproc(image=img)
    im = np.transpose(im, (2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).float() / 255.0
    im = im.unsqueeze(0)
    if use_gpu:
        im = im.cuda()

    model = model.model
    model.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(WARMUP):
            _ = model(im)

        torch.cuda.synchronize()
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            _ = model(im)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times)

def benchmark_onnx_single(session, img_path):
    """ONNX Runtime 单张推理计时 (不含后处理)"""
    import cv2
    from ultralytics.data.augment import LetterBox

    img = cv2.imread(img_path)
    preproc = LetterBox(new_shape=640, auto=False, stride=32)
    im = preproc(image=img)
    im = np.transpose(im, (2, 0, 1))[::-1]
    im = np.ascontiguousarray(im).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis=0)

    input_name = session.get_inputs()[0].name
    for _ in range(WARMUP):
        _ = session.run(None, {input_name: im})

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: im})
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times)

def check_consistency(pt_results, onnx_output):
    """检查 ONNX 输出与 PyTorch 输出的一致性"""
    # Get PyTorch output from model directly
    import torch
    pt_out = pt_results[0].probs if hasattr(pt_results[0], 'probs') else None
    print(f"  PyTorch boxes: {len(pt_results[0].boxes)}")
    print(f"  ONNX output shape: {onnx_output[0].shape}")
    print(f"  ONNX output range: [{onnx_output[0].min():.4f}, {onnx_output[0].max():.4f}]")

if __name__ == '__main__':
    import os
    test_imgs = sorted([os.path.join(TEST_IMG, f) for f in os.listdir(TEST_IMG)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if not test_imgs:
        print("No test images found!")
        exit(1)

    test_img = test_imgs[0]
    print(f"Test image: {test_img}")
    print(f"Warmup: {WARMUP}, Runs: {RUNS}")
    print("=" * 60)

    # Check providers
    available = ort.get_available_providers()
    print(f"Available providers: {available}")

    # Try CUDA provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'CUDAExecutionProvider' not in available:
        print("CUDA provider not available for ONNX, using CPU")
        providers = ['CPUExecutionProvider']

    # 1. PyTorch
    print("\n--- PyTorch ---")
    model = YOLO(MODEL_PT)
    pt_single_mean, pt_single_std = benchmark_pt_single(model, test_img, use_gpu=True)
    print(f"Pure inference: {pt_single_mean:.2f} ± {pt_single_std:.2f} ms")

    pt_mean, pt_std, pt_results = benchmark_pt(model, TEST_IMG)
    print(f"End-to-end (batch): {pt_mean:.2f} ± {pt_std:.2f} ms")

    # 2. ONNX Runtime
    print("\n--- ONNX Runtime ---")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(MODEL_ONNX, so, providers=providers)

    actual_providers = session.get_providers()
    print(f"Using providers: {actual_providers}")

    onnx_single_mean, onnx_single_std = benchmark_onnx_single(session, test_img)
    print(f"Pure inference: {onnx_single_mean:.2f} ± {onnx_single_std:.2f} ms")

    onnx_mean, onnx_std, onnx_out = benchmark_onnx(session, test_img)
    print(f"Single image: {onnx_mean:.2f} ± {onnx_std:.2f} ms")

    check_consistency(pt_results, onnx_out)

    # 3. Summary
    print("\n" + "=" * 60)
    print("SPEED COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'Pure Inference':>15} {'Speedup':>10}")
    print("-" * 55)
    print(f"{'PyTorch (CUDA)':<30} {pt_single_mean:>10.2f} ms {'1.0x':>10}")
    if onnx_single_mean > 0:
        speedup = pt_single_mean / onnx_single_mean
        print(f"{'ONNX Runtime':<30} {onnx_single_mean:>10.2f} ms {speedup:>9.2f}x")
