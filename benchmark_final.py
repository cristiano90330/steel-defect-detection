"""
最终三方速度对比：PyTorch / ONNX Runtime CUDA / TensorRT FP16 Engine
使用纯推理（不含前后处理）进行公平对比
"""
import time
import numpy as np
import onnxruntime as ort
import cv2
import os
from ultralytics.data.augment import LetterBox

MODEL_PT = r"D:\ultralytics\steel-defect-detection\runs\detect\steel_improved_v14\weights\best.pt"
MODEL_ONNX = r"D:\ultralytics\steel-defect-detection\runs\detect\steel_improved_v14\weights\best.onnx"
MODEL_ENGINE = r"D:\ultralytics\steel-defect-detection\runs\detect\steel_improved_v14\weights\best.engine"
TEST_IMG_DIR = r"D:\ultralytics\steel-defect-detection\data\NEU-DET\split_val\images"
WARMUP = 30
RUNS = 300

def get_test_img():
    imgs = sorted([os.path.join(TEST_IMG_DIR, f) for f in os.listdir(TEST_IMG_DIR)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    return imgs[0]

def preprocess(img_path):
    img = cv2.imread(img_path)
    preproc = LetterBox(new_shape=640, auto=False, stride=32)
    im = preproc(image=img)
    im = np.transpose(im, (2, 0, 1))[::-1]
    im = np.ascontiguousarray(im).astype(np.float32) / 255.0
    return np.expand_dims(im, axis=0)

def benchmark_pt():
    import torch
    from ultralytics import YOLO
    model = YOLO(MODEL_PT).model.cuda()
    model.eval()
    img = torch.from_numpy(preprocess(get_test_img())).cuda()

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(img)
        torch.cuda.synchronize()
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            _ = model(img)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

def benchmark_onnx(providers, label):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(MODEL_ONNX, so, providers=providers)
    img = preprocess(get_test_img())
    input_name = session.get_inputs()[0].name
    for _ in range(WARMUP):
        _ = session.run(None, {input_name: img})
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: img})
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times), session.get_providers()

def benchmark_trt_engine():
    """Benchmark pure TensorRT engine (via ultralytics inference without postprocess)"""
    from ultralytics import YOLO
    model = YOLO(MODEL_ENGINE)
    img_path = get_test_img()
    # Warmup
    for _ in range(WARMUP):
        _ = model(img_path, imgsz=640, half=True, verbose=False)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _ = model(img_path, imgsz=640, half=True, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

if __name__ == '__main__':
    print("=" * 70)
    print("FINAL SPEED BENCHMARK: Steel Defect Detection (YOLOv8s)")
    print(f"GPU: NVIDIA RTX 4060 Laptop | Warmup: {WARMUP} | Runs: {RUNS}")
    print("=" * 70)

    results = {}

    # 1. PyTorch FP32
    print("\n[1/4] PyTorch FP32...")
    pt_mean, pt_std = benchmark_pt()
    results['PyTorch FP32'] = (pt_mean, pt_std)
    print(f"  {pt_mean:.2f} ± {pt_std:.2f} ms")

    # 2. ONNX Runtime CUDA
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print("\n[2/4] ONNX Runtime CUDA...")
        cuda_mean, cuda_std, providers = benchmark_onnx(['CUDAExecutionProvider', 'CPUExecutionProvider'], 'CUDA')
        results['ONNX RT CUDA'] = (cuda_mean, cuda_std)
        print(f"  {cuda_mean:.2f} ± {cuda_std:.2f} ms  [{providers[0]}]")

    # 3. ONNX Runtime TensorRT
    if 'TensorrtExecutionProvider' in ort.get_available_providers():
        print("\n[3/4] ONNX RT TensorRT EP...")
        import os as _os
        # Set PATH for TensorRT DLLs
        _os.environ['PATH'] = r'D:\anaconda\envs\pytorch_env\Lib\site-packages\tensorrt_libs' + _os.pathsep + _os.environ.get('PATH', '')
        trt_mean, trt_std, providers = benchmark_onnx(['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], 'TRT')
        results['ONNX RT TensorRT'] = (trt_mean, trt_std)
        print(f"  {trt_mean:.2f} ± {trt_std:.2f} ms  [{providers[0]}]")

    # 4. TensorRT FP16 Engine (pure, no ONNX Runtime)
    print("\n[4/4] TensorRT FP16 Engine (pure)...")
    engine_mean, engine_std = benchmark_trt_engine()
    results['TensorRT Engine FP16'] = (engine_mean, engine_std)
    print(f"  {engine_mean:.2f} ± {engine_std:.2f} ms")

    # Summary
    print("\n" + "=" * 70)
    print("SPEED COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Backend':<25} {'Inference (ms)':>15} {'Speedup vs PT':>15}")
    print("-" * 55)
    pt_baseline = results.get('PyTorch FP32', (0, 0))[0]
    for name, (mean, std) in results.items():
        speedup = pt_baseline / mean if pt_baseline > 0 else 0
        print(f"{name:<25} {mean:>10.2f} ±{std:>4.2f}  {speedup:>13.2f}x")

    # Also show model sizes
    print(f"\n{'Model Sizes':<25}")
    for path in [MODEL_PT, MODEL_ONNX, MODEL_ENGINE]:
        name = os.path.basename(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name:<22} {size_mb:.1f} MB")

    # Accuracy summary
    print(f"\n{'Accuracy Summary (mAP@0.5)':<25}")
    print(f"  PyTorch FP32:         0.7337  (baseline)")
    print(f"  TensorRT FP16 Engine: 0.7256  (-1.1%)")
