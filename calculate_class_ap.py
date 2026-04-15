from ultralytics import YOLO

if __name__ == '__main__':
    # 加载你刚训练好的最终版模型
    model = YOLO(r"D:\ultralytics\runs\detect\steel_final_version\weights\best.pt")

    # 验证
    results = model.val(
        data=r"D:\ultralytics\works\NEU-DET\data.yaml",
        imgsz=640,
        workers=0,
        verbose=True
    )

    # 输出最终精度
    print("\n===== 最终模型各类别 AP@0.5 =====")
    for i, name in results.names.items():
        print(f"{name:18s} {results.box.ap50[i]:.4f}")

    print(f"\n整体 mAP@0.5: {results.box.map50:.4f}")