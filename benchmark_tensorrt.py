"""
ONNX Runtime + TensorRT Execution Provider 推理速度测试
"""
import time
import numpy as np
import onnxruntime as ort
import cv2
import os
from ultralytics.data.augment import LetterBox

MODEL_ONNX = r"D:\ultralytics\steel-defect-detection\runs\detect\steel_improved_v14\weights\best.onnx"
TEST_IMG_DIR = r"D:\ultralytics\steel-defect-detection\data\NEU-DET\split_val\images"
WARMUP = 30
RUNS = 200

def preprocess(img_path):
    img = cv2.imread(img_path)
    preproc = LetterBox(new_shape=640, auto=False, stride=32)
    im = preproc(image=img)
    im = np.transpose(im, (2, 0, 1))[::-1]
    im = np.ascontiguousarray(im).astype(np.float32) / 255.0
    return np.expand_dims(im, axis=0)

def benchmark(providers, label):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # For TensorRT: set optimization level higher and enable graph capture
    if 'TensorrtExecutionProvider' in providers:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    try:
        session = ort.InferenceSession(MODEL_ONNX, so, providers=providers)
    except Exception as e:
        print(f"  Failed to create session with {providers}: {e}")
        return None

    actual = session.get_providers()
    print(f"  Using: {actual}")

    test_imgs = sorted([os.path.join(TEST_IMG_DIR, f) for f in os.listdir(TEST_IMG_DIR)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    img = preprocess(test_imgs[0])
    input_name = session.get_inputs()[0].name

    # Warmup (ORT TensorRT needs more warmup iterations)
    print(f"  Warming up ({WARMUP} iterations)...")
    for i in range(WARMUP):
        _ = session.run(None, {input_name: img})

    print(f"  Benchmarking ({RUNS} iterations)...")
    times = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: img})
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times)

if __name__ == '__main__':
    available = ort.get_available_providers()
    print(f"Available: {available}\n")

    # Test 1: CPU (baseline)
    print("=" * 50)
    print("CPU (baseline)")
    res_cpu = benchmark(['CPUExecutionProvider'], 'CPU')

    # Test 2: CUDA
    if 'CUDAExecutionProvider' in available:
        print("=" * 50)
        print("CUDA")
        res_cuda = benchmark(['CUDAExecutionProvider', 'CPUExecutionProvider'], 'CUDA')
    else:
        res_cuda = None

    # Test 3: TensorRT
    if 'TensorrtExecutionProvider' in available:
        print("=" * 50)
        print("TensorRT")
        res_trt = benchmark(['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], 'TensorRT')
    else:
        res_trt = None

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SPEED COMPARISON (Pure Inference)")
    print("=" * 70)
    print(f"{'Backend':<25} {'Avg (ms)':>12} {'Std (ms)':>12} {'Speedup vs CPU':>15}")
    print("-" * 70)

    results = {}
    if res_cpu:
        results['CPU'] = res_cpu
    if res_cuda:
        results['CUDA'] = res_cuda
    if res_trt:
        results['TensorRT'] = res_trt

    cpu_mean = results.get('CPU', (0, 0))[0]
    for name, (mean, std) in results.items():
        speedup = cpu_mean / mean if cpu_mean > 0 else 0
        print(f"{name:<25} {mean:>10.2f}  {std:>10.2f}  {speedup:>13.2f}x")

    if 'CUDA' in results and 'TensorRT' in results:
        trt_vs_cuda = results['CUDA'][0] / results['TensorRT'][0]
        print(f"\nTensorRT vs CUDA speedup: {trt_vs_cuda:.2f}x")

    import torch
    print(f"\nPyTorch reference: ~5.44 ms (from previous benchmark)")
