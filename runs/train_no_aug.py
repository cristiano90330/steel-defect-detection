from ultralytics import YOLO

# 加载 yolov8s 模型
model = YOLO("yolov8s.yaml")

# 开始训练（完全关闭数据增强）
model.train(
    data=r"D:\ultralytics\works\NEU-DET\data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    name="steel_defect_model_s_no_aug",
    amp=False,  # 关闭网络检查
    workers=0,  # Windows必加

    # 以下 = 全部关闭数据增强
    augment=False,
    hsv_h=0,
    hsv_s=0,
    hsv_v=0,
    degrees=0,
    translate=0,
    scale=0,
    shear=0,
    perspective=0,
    flipud=0,
    fliplr=0,
    mosaic=0,
    mixup=0
)